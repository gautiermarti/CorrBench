"""Create reproducible tables and figures from the completed frozen ABIDE pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from corrbench.neuro import sha256

METHODS = {
    "replay": "Training replay",
    "gaussian": "Transformed Gaussian",
    "diffeocfm": "DiffeoCFM selected",
    "diffeocfm_last": "DiffeoCFM final",
    "nodeflow": "Node-aware CorrFlow pilot",
    "anonymous_flow": "Anonymous CorrFlow pilot",
}


def bootstrap_auc_difference(labels, candidate, reference):
    """Paired subject bootstrap of mean per-generation-seed AUC, not an ensemble."""
    negative, positive = [np.flatnonzero(labels == c) for c in [0, 1]]
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
    scores = np.concatenate([reference[None], candidate])
    comparisons = scores[:, positive, None] - scores[:, None, negative]
    concordance = (comparisons > 0) + 0.5 * (comparisons == 0)
    auc = np.einsum("bi,sij,bj->bs", pw, concordance, nw, optimize=True)
    auc /= len(positive) * len(negative)
    # Independent implementation cross-check against sklearn on the same resample.
    indices = np.concatenate([np.repeat(negative, nw[0]), np.repeat(positive, pw[0])])
    for s, probabilities in enumerate(scores):
        np.testing.assert_allclose(
            auc[0, s],
            roc_auc_score(labels[indices], probabilities[indices]),
            atol=1e-14,
        )
    differences = auc[:, 1:].mean(axis=1) - auc[:, 0]
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paper-output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    utility = json.loads((args.results / "test_report.json").read_text())
    diagnostics = json.loads((args.results / "diagnostics.json").read_text())
    audit = json.loads((args.results / "input_audit.json").read_text())
    assert audit["status"] == "passed"
    assert audit["selection_sha256"] == utility["selection_sha256"]
    scores = {r["name"]: r for r in utility["arms"]}
    fidelity = {r["name"]: r for r in diagnostics["rows"]}
    baseline = scores["real_only"]
    summary = []
    with np.load(args.results / "test_predictions.npz") as predictions:
        for key, label in METHODS.items():
            names = [f"{key}_{seed}" for seed in range(2027, 2032)]
            aucs = [scores[n]["roc_auc"] for n in names]
            balanced = [scores[n]["balanced_accuracy"] for n in names]
            row = {
                "method": key,
                "label": label,
                "training_seeds": 0 if key in {"replay", "gaussian"} else 1,
                "generation_seeds": 5,
                "auc_mean": float(np.mean(aucs)),
                "auc_generation_sd": float(np.std(aucs, ddof=1)),
                "balanced_accuracy_mean": float(np.mean(balanced)),
                "balanced_accuracy_generation_sd": float(np.std(balanced, ddof=1)),
                "auc_difference": float(np.mean(aucs) - baseline["roc_auc"]),
                "auc_difference_paired_subject_95pct": bootstrap_auc_difference(
                    predictions["labels"],
                    np.stack([predictions[n] for n in names]),
                    predictions["real_only"],
                ),
                "structure_swd_mean": float(
                    np.mean([fidelity[n]["macro_class_structure_swd"] for n in names])
                ),
                "roi_swd_mean": float(
                    np.mean([fidelity[n]["macro_class_roi_swd"] for n in names])
                ),
                "valid_fraction": min(
                    fidelity[n]["validity"]["valid_fraction"] for n in names
                ),
            }
            summary.append(row)
    report = {
        "scope": "One generator training seed and one test site; five generation seeds",
        "uncertainty": (
            "SD across generation draws; paired bootstrap CI over test subjects "
            "conditional on fitted models/site, unadjusted for multiple comparisons"
        ),
        "reference": baseline,
        "methods": summary,
        "synthetic_only": [r for n, r in scores.items() if "synthetic_only" in n],
        "sources": {
            n: sha256(args.results / n)
            for n in [
                "selection.json",
                "test_report.json",
                "test_predictions.npz",
                "diagnostics.json",
                "input_audit.json",
            ]
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# ABIDE pilot results — 2026-10-05",
        "",
        "Completed local research pilot; no clinical validation or public release.",
        "One training seed per generator; five generation seeds; one test site.",
        f"Real-only ROC-AUC: {baseline['roc_auc']:.4f}; "
        f"balanced accuracy: {baseline['balanced_accuracy']:.4f}.",
        "",
        "| Augmentation | AUC mean ± generation SD | Δ AUC, subject-bootstrap 95% CI "
        "| Structure SWD ↓ | Named-ROI SWD ↓ |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in summary:
        lo, hi = r["auc_difference_paired_subject_95pct"]
        lines.append(
            f"| {r['label']} | {r['auc_mean']:.4f} ± {r['auc_generation_sd']:.4f} "
            f"| {r['auc_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"| {r['structure_swd_mean']:.3f} | {r['roi_swd_mean']:.3f} |"
        )
    lines += [
        "",
        "Every generated matrix is PD. All bootstrap intervals are conditional on the",
        "fitted models and NYU test site, and are unadjusted for multiple comparisons.",
        "Generation seeds are not independent training replications. Structural and",
        "named-ROI SWD have different representations; their scales differ.",
        "Fidelity uses 70 test subjects per diagnosis; utility uses all 160.",
        "",
        "No method establishes improved augmentation utility in this pilot. Better",
        "anonymous structural fidelity is insufficient for named-ROI task fidelity.",
        "The node-aware architecture improves its named-ROI distance relative to the",
        "anonymous ablation, but the small feasibility configuration is not yet a",
        "validated neuroscience generator. Retain the simple Gaussian baseline.",
        "",
        "Synthetic-only diagnostics (one generation seed):",
        "",
        "| Generator | ROC-AUC | Balanced accuracy |",
        "| --- | --- | --- |",
    ]
    for row in report["synthetic_only"]:
        lines.append(
            f"| {row['name']} | {row['roc_auc']:.4f} | {row['balanced_accuracy']:.4f} |"
        )
    lines += [
        "",
        "Protocol: docs/fmri_pilot_protocol.md. Dataset: 349 training subjects,",
        "92 validation subjects from UM_1/UM_2, and 160 NYU test subjects; 111 ROIs.",
        "Inputs are official ABIDE PCP preprocessed regional BOLD series. Numerical",
        "exclusions, OAS estimator, site-dependent observation lengths, and source",
        "permissions accompany interpretation. No test-driven retraining followed.",
        "Descriptive fidelity diagnostics were added after utility evaluation began;",
        "they are not retrospective checkpoint selection criteria.",
    ]
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    if args.paper_output:
        rows = [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Augmentation & AUC & $\Delta$ AUC 95\% CI & F-SWD & ROI-SWD \\",
            r"\midrule",
            f"Real only & {baseline['roc_auc']:.3f} & --- & --- & --- " + r"\\",
        ]
        for r in summary:
            lo, hi = r["auc_difference_paired_subject_95pct"]
            rows.append(
                f"{r['label']} & {r['auc_mean']:.3f}"
                + r"\,$\pm$\,"
                + f"{r['auc_generation_sd']:.3f} & [{lo:+.3f}, {hi:+.3f}] & "
                + f"{r['structure_swd_mean']:.3f} & {r['roi_swd_mean']:.3f} "
                + r"\\"
            )
        rows += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{ABIDE pilot: one training seed and five generation seeds,",
            r"one test site. AUC is mean $\pm$ generation SD. Paired bootstrap",
            r"intervals for mean AUC minus real-only condition on fitted models and",
            r"site; they are not adjusted for multiple comparisons. F-SWD summarizes",
            r"anonymous matrix structure; ROI-SWD retains named edges. Both are",
            r"macro-averaged over diagnoses; smaller is better within each column.",
            r"No augmentation method establishes an AUC gain in this pilot.}",
            r"\label{tab:fmriutility}",
            r"\end{table}",
        ]
        args.paper_output.parent.mkdir(parents=True, exist_ok=True)
        args.paper_output.write_text("\n".join(rows) + "\n")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = plt.cm.tab10(np.arange(len(summary)))
    for i, (row, color) in enumerate(zip(summary, colors, strict=True)):
        lo, hi = row["auc_difference_paired_subject_95pct"]
        delta = row["auc_difference"]
        axes[0].errorbar(
            delta, i, xerr=[[delta - lo], [hi - delta]], fmt="o", color=color, capsize=3
        )
        axes[1].scatter(
            row["structure_swd_mean"],
            row["roi_swd_mean"],
            color=color,
            s=65,
            label=row["label"],
        )
    axes[0].set(
        yticks=np.arange(len(summary)),
        yticklabels=[r["label"] for r in summary],
        xlabel="AUC difference from real-only (paired subject 95% CI)",
        title="No established augmentation gain",
    )
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].invert_yaxis()
    axes[1].set(
        xlabel="Anonymous structural SWD ↓",
        ylabel="Named-ROI SWD ↓",
        title="Different fidelity questions",
    )
    fig.suptitle(
        "ABIDE: one training seed, five generation seeds, one held-out site",
        fontsize=12,
    )
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(args.output / f"fmri_pilot.{suffix}", dpi=180)
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
