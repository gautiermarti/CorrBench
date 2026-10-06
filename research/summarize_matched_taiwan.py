"""Render the complete audited categorical comparison without selecting results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from corrbench.reference import sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "corrflow": "CorrFlow",
    "diffeocfm": "DiffeoCFM",
    "gaussian": "Transformed Gaussian",
    "replay": "Training replay",
    "ccorrgan": "cCorrGAN reconstruction",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads(args.audit.read_text())
    assert full["states"]["corrflow"]["complete"]
    assert full["states"]["diffeocfm_evaluation"]["complete"]
    for row in full["summary"]:
        seeds = [
            r
            for r in full["rows"]
            if (r["method"], r["weights"]) == (row["method"], row["weights"])
        ]
        assert len(seeds) == 5
        for key, values in row["metrics"].items():
            observed = np.array([r[key] for r in seeds])
            np.testing.assert_allclose(
                [observed.mean(), observed.std(ddof=1)],
                [values["mean"], values["sd"]],
                rtol=0,
                atol=1e-14,
            )
    public = {
        key: full[key]
        for key in [
            "protocol",
            "scope",
            "training_budgets",
            "equal_compute",
            "case_selection",
            "timing_limit",
            "seed_uncertainty_limit",
            "states",
            "source_data_sha256",
            "validation_selection_checks",
            "rows",
            "summary",
        ]
    }
    public["audit_sha256"] = sha256(args.audit)
    public["render_code_sha256"] = sha256(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matched_taiwan_results.json").write_text(
        json.dumps(public, indent=2) + "\n"
    )
    selected = [r for r in full["summary"] if r["weights"] != "last"]
    selected.sort(key=lambda r: list(LABELS).index(r["method"]))
    gan_complete = full["states"]["ccorrgan"]["complete"]
    gan_budget = " and cCorrGAN 200" if gan_complete else ""
    validation_error = max(
        abs(r["training_validation_swd"] - r["reproduced_native_validation_swd"])
        for r in full["validation_selection_checks"]
    )
    tex = [
        r"\begin{table}[t]",
        r"\caption{Matched categorical Taiwan comparison. "
        r"Identical 1,500 test cases, 500 per regime. Means and sample standard "
        r"deviations over five training/paired generation seeds for neural models; "
        r"five draws from the same fitted training data for controls. Selected "
        r"weights; CorrFlow 20 epochs, DiffeoCFM 200"
        + gan_budget
        + r", not equal compute.}",
        r"\label{tab:matchedcategorical}",
        r"\centering\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & F-SWD $\downarrow$ & Precision & Recall & Portfolio SWD "
        r"$\downarrow$ & C2ST AUC \\",
        r"\midrule",
    ]
    md = [
        "# Audited categorical Taiwan comparison — 2026-10-05",
        "",
        "Compared with DiffeoCFM, CorrFlow has lower structural and portfolio "
        "distances under this fixed protocol. DiffeoCFM has slightly higher "
        "structural coverage; both remain "
        "readily distinguishable from real data. This is not universal superiority.",
        "",
        "| Method | Structural SWD ↓ | Precision | Recall | Portfolio SWD ↓ "
        "| C2ST AUC (.5 ideal) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        m = row["metrics"]
        label = LABELS[row["method"]]
        f, p = m["feature_swd"], m["portfolio_swd"]
        tex.append(
            f"{label} & \\({f['mean']:.3f}\\pm{f['sd']:.3f}\\) & "
            f"{m['precision']['mean']:.3f} & {m['recall']['mean']:.3f} & "
            f"\\({p['mean']:.3f}\\pm{p['sd']:.3f}\\) & "
            f"{m['classifier_auc']['mean']:.3f}" + r" \\"
        )
        md.append(
            f"| {label} | {f['mean']:.4f} ± {f['sd']:.4f} | "
            f"{m['precision']['mean']:.4f} | {m['recall']['mean']:.4f} | "
            f"{p['mean']:.4f} ± {p['sd']:.4f} | "
            f"{m['classifier_auc']['mean']:.4f} |"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    args.table.write_text("\n".join(tex) + "\n")
    md += [
        "",
        (
            "All evaluated matrices pass validity checks after required image-model "
            "repair; this is not a raw-generator validity claim. "
            if gan_complete
            else "All evaluated generated matrices pass validity checks. "
        )
        + "The audit "
        "verifies frozen source/data/output hashes, all completed budgets, identical "
        "case indices and labels, replay membership, validation-minimum selection "
        "and reproduction of all "
        f"{len(full['validation_selection_checks'])} native validation scores "
        f"(maximum absolute difference {validation_error:.3g}).",
        "",
        "CorrFlow selected epochs: 1, 3, 18, 6, 4. DiffeoCFM: 170, 190, 140, 200, "
        "130. Final-weight structural SWD is .8363 ± .0679 for CorrFlow and "
        ".7980 ± .0221 for DiffeoCFM. The final-weight ordering therefore differs "
        "from the predeclared validation-selected comparison; preserve both.",
        "",
        "DiffeoCFM improves structural distance over its fitted Gaussian prior, "
        "but its portfolio distance worsens (.8620 versus .8107) and its recall "
        "falls (.3549 versus .5043). Prior-versus-endpoint reporting matters.",
        "",
        "Ordered GFT distance favors DiffeoCFM (.1082 versus .3348), while "
        "permutation-orbit GFT distance favors CorrFlow (.0767 versus .0856). "
        "The former retains Ward coordinate order; the latter compares independently "
        "relabelled matrices. Neither metric alone certifies useful generation.",
        "",
        "The nearest-training diagnostic searches a sampled training reference, "
        "so its approximate exact-match fraction is not full-corpus membership. "
        "The separate audit confirms every replay matrix is an exact training draw. "
        "No privacy claim follows from neural outputs having no exact match there.",
        "",
        "Limits: previously inspected Taiwan periods; categorical conditions "
        "only; unequal model-specific training budgets and source laws; one paired "
        "generation seed per neural training seed. Seed SD is not uncertainty over "
        "independent market episodes. Concurrent timings do not establish speed "
        "rankings. "
        + (
            "The complete five-seed cCorrGAN reconstruction is included. Completing "
            "its declared budget does not establish faithful original training "
            "or recover the original published fidelity."
            if gan_complete
            else "The full cCorrGAN suite is still running and is excluded from "
            "this completed two-flow comparison."
        ),
        "",
    ]
    (args.output / "matched_taiwan_results.md").write_text("\n".join(md))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.7))
    for ax, metric, label in zip(
        axes,
        ["feature_swd", "recall", "portfolio_swd"],
        ["Structural SWD ↓", "Coverage recall ↑", "Portfolio SWD ↓"],
        strict=True,
    ):
        values = [r["metrics"][metric]["mean"] for r in selected]
        errors = [r["metrics"][metric]["sd"] for r in selected]
        ax.barh(
            np.arange(len(selected)),
            values,
            xerr=errors,
            height=0.55,
            color=["#177c75", "#587487", "#86aea6", "#bd803d", "#8d6f9e"][
                : len(selected)
            ],
            capsize=3,
        )
        ax.set_yticks(
            np.arange(len(selected)),
            [LABELS[r["method"]] for r in selected] if ax is axes[0] else [],
        )
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, max(v + e for v, e in zip(values, errors, strict=True)) * 1.14)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "A matched comparison gives a qualified answer",
        x=0.03,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.04,
        "Taiwan · categorical conditions · 1,500 identical test cases "
        "· five seeds · validation-selected checkpoints\n"
        "Whiskers: seed SD. Previously inspected periods; unequal training budgets. "
        "Both flow models have C2ST AUC above .996.",
        fontsize=8.5,
        color="#52616b",
    )
    fig.tight_layout(rect=[0, 0.15, 1, 0.92], w_pad=2)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(
            args.output / f"matched_taiwan.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print("Audited comparison summary, table and figures written")


if __name__ == "__main__":
    main()
