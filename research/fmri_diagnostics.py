"""Measure anonymous structure and named-ROI fidelity for the frozen ABIDE pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corrbench.metrics import (
    classifier_two_sample_auc,
    energy_distance,
    manifold_precision_recall,
    nearest_train_diagnostic,
    sliced_wasserstein,
    structural_features,
    validity_summary,
)
from corrbench.neuro import sha256


def roi_features(matrices):
    i, j = np.triu_indices(matrices.shape[-1], 1)
    return np.arctanh(np.clip(matrices[:, i, j], -1 + 1e-7, 1 - 1e-7))


def distances(real, generated, scale):
    return {
        "swd": sliced_wasserstein(
            real, generated, scale=scale, seed=2027, device="cpu"
        ),
        "energy": energy_distance(real / scale, generated / scale, seed=2027),
        "mean_absolute_z_error": float(
            np.mean(np.abs(real.mean(axis=0) - generated.mean(axis=0)) / scale)
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--generated-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    with np.load(args.data / "train.npz") as a:
        train, train_labels = a["matrices"], a["labels"]
    with np.load(args.data / "test.npz") as a:
        test, test_labels = a["matrices"], a["labels"]
    train_structure, names = structural_features(train)
    train_roi = roi_features(train)
    scales = {
        "structure": np.maximum(train_structure.std(axis=0, ddof=1), 1e-6),
        "roi": np.maximum(train_roi.std(axis=0, ddof=1), 1e-6),
    }
    # Balance the pooled comparison; test-site prevalence is not the generator target.
    per_class = min(int((test_labels == c).sum()) for c in [0, 1])
    rng = np.random.default_rng(2027)
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(test_labels == c), per_class, replace=False)
            for c in [0, 1]
        ]
    )
    real = test[selected]
    real_labels = test_labels[selected]
    real_structure, _ = structural_features(real)
    real_roi = roi_features(real)
    manifest = json.loads(args.generated_manifest.read_text())
    arms = [
        {"name": f"replay_{seed}", "kind": "replay", "seed": seed}
        for seed in range(2027, 2032)
    ] + [a for a in manifest if a["kind"] == "generated"]
    rows = []
    for arm in arms:
        seed = arm["seed"]
        if arm["kind"] == "replay":
            rng = np.random.default_rng(seed)
            indices = np.concatenate(
                [
                    rng.choice(np.flatnonzero(train_labels == c), 175, replace=True)
                    for c in [0, 1]
                ]
            )
            matrices, labels = train[indices], train_labels[indices]
            artifact_hash = None
        else:
            path = Path(arm["path"])
            artifact_hash = sha256(path)
            assert artifact_hash == arm["sha256"]
            with np.load(path) as a:
                matrices, labels = a["matrices"], a["labels"]
        all_structure, _ = structural_features(matrices)
        all_roi = roi_features(matrices)
        rng = np.random.default_rng(2027)
        chosen = np.concatenate(
            [
                rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
                for c in [0, 1]
            ]
        )
        generated_structure, generated_roi = all_structure[chosen], all_roi[chosen]
        assert np.array_equal(labels[chosen], real_labels)
        classes = {}
        for c in [0, 1]:
            mask = real_labels == c
            classes[str(c)] = {
                "structure": distances(
                    real_structure[mask], generated_structure[mask], scales["structure"]
                ),
                "roi": distances(real_roi[mask], generated_roi[mask], scales["roi"]),
            }
        test_effect = real_roi[real_labels == 1].mean(0) - real_roi[
            real_labels == 0
        ].mean(0)
        generated_effect = all_roi[labels == 1].mean(0) - all_roi[labels == 0].mean(0)
        a, b = test_effect / scales["roi"], generated_effect / scales["roi"]
        row = {
            "name": arm["name"],
            "artifact_sha256": artifact_hash,
            "validity": validity_summary(matrices),
            "by_class": classes,
            "pooled_structure": distances(
                real_structure, generated_structure, scales["structure"]
            ),
            "pooled_roi": distances(real_roi, generated_roi, scales["roi"]),
            "macro_class_structure_swd": float(
                np.mean([x["structure"]["swd"] for x in classes.values()])
            ),
            "macro_class_roi_swd": float(
                np.mean([x["roi"]["swd"] for x in classes.values()])
            ),
            "structure_manifold": manifold_precision_recall(
                real_structure / scales["structure"],
                generated_structure / scales["structure"],
                seed=2027,
            ),
            "structure_classifier": classifier_two_sample_auc(
                real_structure / scales["structure"],
                generated_structure / scales["structure"],
                seed=2027,
            ),
            "roi_nearest_train": nearest_train_diagnostic(
                train_roi, real_roi, all_roi, scale=scales["roi"], seed=2027
            ),
            "diagnosis_mean_contrast": {
                "cosine_with_test": float(
                    a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
                ),
                "standardized_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                "generated_norm": float(np.linalg.norm(b)),
                "test_norm": float(np.linalg.norm(a)),
                "interpretation": "Descriptive, noisy site-specific effect; not causal",
            },
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "name",
                        "macro_class_structure_swd",
                        "macro_class_roi_swd",
                    ]
                }
            ),
            flush=True,
        )
    report = {
        "source_sha256": {s: sha256(args.data / f"{s}.npz") for s in ["train", "test"]},
        "code_sha256": sha256(Path(__file__)),
        "test_indices": selected.tolist(),
        "per_class": per_class,
        "feature_names": names,
        "scales": "Real train standard deviation, ddof=1, floor=1e-6",
        "scope": (
            "Descriptive diagnostics defined after test utility was started; "
            "not a new selection rule or confirmatory hypothesis test"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
