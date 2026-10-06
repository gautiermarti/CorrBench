"""Select augmentation classifiers on validation sites, then evaluate once on test."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from corrbench.neuro import sha256

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def features(matrices):
    x = np.asarray(matrices, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != x.shape[-2] or not np.isfinite(x).all():
        raise ValueError("Require finite square matrix batches")
    if not np.allclose(x, x.swapaxes(-1, -2), atol=1e-6, rtol=0):
        raise ValueError("Asymmetric generated matrices")
    if not np.allclose(np.diagonal(x, axis1=-2, axis2=-1), 1, atol=1e-6):
        raise ValueError("Nonunit matrix diagonal")
    if np.linalg.eigvalsh(x).min() <= 0:
        raise ValueError("Utility requires valid PD generated correlations")
    i, j = np.triu_indices(x.shape[-1], 1)
    return np.arctanh(np.clip(x[:, i, j], -1 + 1e-7, 1 - 1e-7))


def metrics(y, probabilities):
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probabilities >= 0.5)),
    }


def stratified_bootstrap_difference(y, candidate, reference, *, draws=1000, seed=7201):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    differences = {name: [] for name in ["roc_auc", "balanced_accuracy"]}
    for _ in range(draws):
        selected = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        c, r = (
            metrics(y[selected], candidate[selected]),
            metrics(y[selected], reference[selected]),
        )
        for name in differences:
            differences[name].append(c[name] - r[name])
    return {
        name: np.quantile(values, [0.025, 0.975]).tolist()
        for name, values in differences.items()
    }


def select(data, output, generated_manifest):
    if output.exists():
        raise FileExistsError(output)
    arrays = {}
    for split in ["train", "validation"]:
        with np.load(data / f"{split}.npz") as a:
            arrays[split] = {k: a[k] for k in ["matrices", "labels", "subject_ids"]}
    train, val = arrays["train"], arrays["validation"]
    if set(train["subject_ids"]) & set(val["subject_ids"]):
        raise ValueError("Overlapping subjects")
    scaler = StandardScaler().fit(features(train["matrices"]))
    xtrain = scaler.transform(features(train["matrices"]))
    xval = scaler.transform(features(val["matrices"]))
    per_class = (len(xtrain) + 1) // 2
    external = json.loads(generated_manifest.read_text()) if generated_manifest else []
    arms = [{"name": "real_only", "seed": 2027, "kind": "real"}]
    arms += [
        {"name": f"replay_{seed}", "seed": seed, "kind": "replay"}
        for seed in range(2027, 2032)
    ]
    arms += external
    if len({a["name"] for a in arms}) != len(arms):
        raise ValueError("Duplicate arm names")
    output.mkdir(parents=True)
    joblib.dump(scaler, output / "feature_scaler.joblib")
    results = []
    for arm in arms:
        name, seed = arm["name"], int(arm["seed"])
        if not name.replace("_", "").isalnum():
            raise ValueError("Unsafe arm name")
        x, y = xtrain, train["labels"]
        if arm["kind"] == "replay":
            rng = np.random.default_rng(seed)
            chosen = np.concatenate(
                [
                    rng.choice(np.flatnonzero(y == c), per_class, replace=True)
                    for c in [0, 1]
                ]
            )
            x, y = np.concatenate([x, x[chosen]]), np.concatenate([y, y[chosen]])
        elif arm["kind"] in {"generated", "synthetic_only"}:
            path = Path(arm["path"])
            if sha256(path) != arm["sha256"]:
                raise ValueError("Generated artifact changed")
            with np.load(path) as a:
                matrices, labels = a["matrices"], a["labels"]
            if matrices.shape[1:] != train["matrices"].shape[1:]:
                raise ValueError("Generated matrix size differs")
            if labels.shape != (2 * per_class,) or any(
                (labels == c).sum() != per_class for c in [0, 1]
            ):
                raise ValueError("Augmentation counts differ from declared budget")
            generated = scaler.transform(features(matrices))
            if arm["kind"] == "synthetic_only":
                x, y = generated, labels
            else:
                x, y = np.concatenate([x, generated]), np.concatenate([y, labels])
        elif arm["kind"] != "real":
            raise ValueError("Unknown arm kind")
        history, best, selected_model = [], -np.inf, None
        for c in C_GRID:
            model = LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x, y)
            score = metrics(val["labels"], model.predict_proba(xval)[:, 1])
            history.append({"C": c, **score})
            if score["roc_auc"] > best:
                best, selected_model = score["roc_auc"], model
        joblib.dump(selected_model, output / f"{name}.joblib")
        results.append(
            {
                **arm,
                "C": float(selected_model.C),
                "validation": history,
                "training_rows": len(y),
            }
        )
        print(
            json.dumps(
                {"arm": name, "selected_C": selected_model.C, "validation_auc": best}
            ),
            flush=True,
        )
    record = {
        "protocol": "fmri_pilot_protocol_20261005",
        "source_sha256": {s: sha256(data / f"{s}.npz") for s in arrays},
        "code_sha256": sha256(Path(__file__)),
        "test_opened": False,
        "augmentation_per_class": per_class,
        "C_grid": C_GRID,
        "selection": "maximum validation-site ROC-AUC, first/smallest C tie",
        "feature_scale": "real training subjects only, shared across all arms",
        "arms": results,
        "model_sha256": {p.name: sha256(p) for p in output.glob("*.joblib")},
    }
    (output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")


def evaluate(data, output):
    report_path = output / "test_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    record = json.loads((output / "selection.json").read_text())
    if record["code_sha256"] != sha256(Path(__file__)):
        raise ValueError("Utility code changed after selection")
    for name, expected in record["model_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError("Selected model changed")
    for split, expected in record["source_sha256"].items():
        if sha256(data / f"{split}.npz") != expected:
            raise ValueError("Training/validation data changed")
    with np.load(data / "test.npz") as a:
        matrices, y, subjects = a["matrices"], a["labels"], a["subject_ids"]
    for split in ["train", "validation"]:
        with np.load(data / f"{split}.npz") as a:
            if set(subjects) & set(a["subject_ids"]):
                raise ValueError("Test subject leakage")
    scaler = joblib.load(output / "feature_scaler.joblib")
    x = scaler.transform(features(matrices))
    predictions = {
        arm["name"]: joblib.load(output / f"{arm['name']}.joblib").predict_proba(x)[
            :, 1
        ]
        for arm in record["arms"]
    }
    reference = predictions["real_only"]
    rows = []
    for arm in record["arms"]:
        name = arm["name"]
        score = metrics(y, predictions[name])
        paired = stratified_bootstrap_difference(y, predictions[name], reference)
        rows.append(
            {"name": name, "C": arm["C"], **score, "paired_difference_95pct": paired}
        )
        print(json.dumps(rows[-1]), flush=True)
    np.savez_compressed(
        output / "test_predictions.npz", labels=y, subject_ids=subjects, **predictions
    )
    report = {
        "test_sha256": sha256(data / "test.npz"),
        "selection_sha256": sha256(output / "selection.json"),
        "test_subjects": len(y),
        "threshold": 0.5,
        "bootstrap": "1000 diagnosis-stratified paired subject draws, seed 7201",
        "scope": (
            "Research proxy on one held-out site; "
            "no clinical or across-site generalization claim"
        ),
        "arms": rows,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "evaluate"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--generated-manifest", type=Path)
    args = p.parse_args()
    if args.stage == "select":
        select(args.data, args.output, args.generated_manifest)
    else:
        evaluate(args.data, args.output)


if __name__ == "__main__":
    main()
