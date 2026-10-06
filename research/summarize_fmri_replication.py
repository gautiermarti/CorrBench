"""Summarize all five unchanged ABIDE training replications, never a partial set."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from corrbench.neuro import sha256

METHODS = {
    "diffeocfm": "DiffeoCFM selected",
    "diffeocfm_last": "DiffeoCFM final",
    "nodeflow": "Node-aware CorrFlow",
    "anonymous_flow": "Anonymous CorrFlow",
}


def read(path):
    return json.loads(path.read_text())


def bootstrap_auc(labels, probability):
    negative, positive = [np.flatnonzero(labels == value) for value in [0, 1]]
    rng = np.random.default_rng(7201)
    nw, pw = [], []
    for _ in range(1000):
        nw.append(
            np.bincount(
                rng.choice(len(negative), len(negative)), minlength=len(negative)
            )
        )
        pw.append(
            np.bincount(
                rng.choice(len(positive), len(positive)), minlength=len(positive)
            )
        )
    nw, pw = np.asarray(nw), np.asarray(pw)
    difference = probability[:, positive, None] - probability[:, None, negative]
    concordance = (difference > 0) + 0.5 * (difference == 0)
    auc = np.einsum("bi,sij,bj->bs", pw, concordance, nw, optimize=True)
    auc /= len(positive) * len(negative)
    indices = np.concatenate([np.repeat(negative, nw[0]), np.repeat(positive, pw[0])])
    for index in [0, 1, len(probability) - 1]:
        np.testing.assert_allclose(
            auc[0, index],
            roc_auc_score(labels[indices], probability[index, indices]),
            atol=1e-14,
        )
    return auc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--table", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.bench_root
    directories = {"original": root / "artifacts/abide_complete_utility_20261005"}
    directories.update(
        {
            family: root / "artifacts/abide_replication_utility_20261005" / family
            for family in ["diffeocfm", "nodeflow", "anonymous_flow"]
        }
    )
    data_hash = sha256(root / "artifacts/abide_ho_20261005/test.npz")
    reports, predictions, sources = {}, {}, {}
    for key, directory in directories.items():
        report = read(directory / "test_report.json")
        audit = read(directory / "input_audit.json")
        assert audit["status"] == "passed"
        assert report["test_sha256"] == data_hash
        assert report["selection_sha256"] == sha256(directory / "selection.json")
        assert audit["selection_sha256"] == report["selection_sha256"]
        reports[key] = {r["name"]: r for r in report["arms"]}
        with np.load(directory / "test_predictions.npz") as a:
            predictions[key] = {name: a[name] for name in a.files}
        sources[key] = {
            name: sha256(directory / name)
            for name in [
                "test_report.json",
                "test_predictions.npz",
                "selection.json",
                "input_audit.json",
            ]
        }
    labels = predictions["original"]["labels"]
    reference = predictions["original"]["real_only"]
    for key in directories:
        np.testing.assert_array_equal(labels, predictions[key]["labels"])
        np.testing.assert_array_equal(
            predictions["original"]["subject_ids"], predictions[key]["subject_ids"]
        )
        np.testing.assert_array_equal(reference, predictions[key]["real_only"])
    ref_auc = float(roc_auc_score(labels, reference))
    rows, probabilities, metrics = [], [reference], {}
    for method in METHODS:
        metrics[method] = []
        for seed in range(1701, 1706):
            values = []
            for draw in range(2027, 2032):
                if seed == 1701:
                    group, name = "original", f"{method}_{draw}"
                else:
                    group = "diffeocfm" if method.startswith("diffeocfm") else method
                    name = f"{method}_train{seed}_draw{draw}"
                probability = predictions[group][name]
                auc = float(roc_auc_score(labels, probability))
                balanced = float(balanced_accuracy_score(labels, probability >= 0.5))
                np.testing.assert_allclose(
                    auc, reports[group][name]["roc_auc"], atol=1e-14
                )
                np.testing.assert_allclose(
                    balanced, reports[group][name]["balanced_accuracy"], atol=1e-14
                )
                probabilities.append(probability)
                values.append(
                    {"generation_seed": draw, "auc": auc, "balanced_accuracy": balanced}
                )
            metrics[method].append(
                {
                    "training_seed": seed,
                    "draws": values,
                    "auc_mean": float(np.mean([r["auc"] for r in values])),
                    "auc_generation_sd": float(
                        np.std([r["auc"] for r in values], ddof=1)
                    ),
                    "balanced_accuracy_mean": float(
                        np.mean([r["balanced_accuracy"] for r in values])
                    ),
                }
            )
    boot = bootstrap_auc(labels, np.stack(probabilities))
    training_indices = np.random.default_rng(7202).integers(0, 5, (1000, 5))
    for index, (method, label) in enumerate(METHODS.items()):
        section = boot[:, 1 + index * 25 : 1 + (index + 1) * 25].reshape(1000, 5, 5)
        training_mean = section.mean(axis=2)
        subject_only = training_mean.mean(axis=1) - boot[:, 0]
        mixed = training_mean[np.arange(1000)[:, None], training_indices].mean(axis=1)
        mixed -= boot[:, 0]
        aucs = [r["auc_mean"] for r in metrics[method]]
        rows.append(
            {
                "method": method,
                "label": label,
                "training_seeds": 5,
                "generation_seeds_per_training": 5,
                "auc_mean": float(np.mean(aucs)),
                "auc_training_sd": float(np.std(aucs, ddof=1)),
                "balanced_accuracy_mean": float(
                    np.mean([r["balanced_accuracy_mean"] for r in metrics[method]])
                ),
                "auc_difference": float(np.mean(aucs) - ref_auc),
                "paired_subject_95pct": np.quantile(
                    subject_only, [0.025, 0.975]
                ).tolist(),
                "training_and_paired_subject_95pct": np.quantile(
                    mixed, [0.025, 0.975]
                ).tolist(),
                "per_training_seed": metrics[method],
            }
        )
    original = read(root / "docs/fmri_pilot_results.json")
    report = {
        "scope": "Five fixed training replications; previously inspected NYU site",
        "reference": original["reference"],
        "controls": [
            r for r in original["methods"] if r["method"] in ["gaussian", "replay"]
        ],
        "methods": rows,
        "dataset_sha256": data_hash,
        "test_subjects": len(labels),
        "source_sha256": sources,
        "code_sha256": sha256(Path(__file__)),
        "uncertainty": "Average AUC over five draws within each training seed, then "
        "mean/sample SD across five training seeds. Paired subject "
        "and joint training/subject bootstraps are conditional on "
        "this site, exploratory and unadjusted for multiplicity. "
        "Five generation draws are not independent training fits.",
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# ABIDE five-training-seed replication — 2026-10-05",
        "",
        report["scope"] + ".",
        "",
        f"Real-only AUC: {ref_auc:.4f}.",
        "",
        "| Method | AUC mean ± training SD | Δ AUC | Joint bootstrap 95% range |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lo, hi = row["training_and_paired_subject_95pct"]
        lines.append(
            f"| {row['label']} | {row['auc_mean']:.4f} ± "
            f"{row['auc_training_sd']:.4f} | {row['auc_difference']:+.4f} "
            f"| [{lo:+.4f}, {hi:+.4f}] |"
        )
    lines += [
        "",
        report["uncertainty"],
        "",
        "The initial one-seed report remains preserved. No test-driven "
        "configuration change was made for this expansion.",
        "All 100 learned augmentation outcomes and all per-training-seed "
        "results are retained in the machine-readable report.",
        "Source audits verify common data, validation selection, complete "
        "budgets, prior parity and ROI ordering.",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['label']}: per-training-seed results",
            "",
            "| Training seed | Mean AUC | Generation SD |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {r['training_seed']} | {r['auc_mean']:.4f} | "
            f"{r['auc_generation_sd']:.4f} |"
            for r in row["per_training_seed"]
        ]
        lines.append("")
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    if args.table:
        tex = [
            r"\begin{table}[t]",
            r"\caption{ABIDE replication: five training seeds, with five generation "
            r"draws averaged within each fit. SD is over training seeds. Difference "
            r"ranges jointly resample training seeds and subjects at the same "
            r"previously inspected site; they are exploratory and unadjusted.}",
            r"\label{tab:fmri_replication}",
            r"\centering\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Augmentation & AUC \(\pm\) training SD & \(\Delta\) AUC [95\% range] \\",
            r"\midrule",
        ]
        for row in rows:
            lo, hi = row["training_and_paired_subject_95pct"]
            tex.append(
                f"{row['label']} & \\({row['auc_mean']:.4f}\\pm "
                f"{row['auc_training_sd']:.4f}\\) & \\({row['auc_difference']:+.4f}"
                f"\\ [{lo:+.4f},{hi:+.4f}]\\)" + r" \\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(tex) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "method",
                        "auc_mean",
                        "auc_training_sd",
                        "training_and_paired_subject_95pct",
                    ]
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
