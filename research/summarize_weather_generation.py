"""Audit the frozen weather experiment and render its complete, limited result."""

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
    "gaussian": "Transformed Gaussian",
    "diffeocfm": "DiffeoCFM (selected)",
    "diffeocfm_last": "DiffeoCFM (final)",
    "replay": "Region / quarter replay",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    plan_path = args.root / "plan.json"
    plan = json.loads(plan_path.read_text())
    ledger = json.loads((args.root / "ledger.json").read_text())
    assert ledger["plan_sha256"] == sha256(plan_path)
    for filename, digest in plan["hashes"].items():
        assert sha256(Path(filename)) == digest, filename
    for job in plan["jobs"]:
        state = ledger["jobs"][job["name"]]
        assert state["status"] == "complete" and state["returncode"] == 0
        for filename, digest in state["artifacts"].items():
            assert sha256(Path(filename)) == digest, filename
    report_path = args.root / "evaluation/report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed" and report["completed_epochs"] == 200
    assert report["generation_seeds"] == list(range(2027, 2032))
    assert report["per_condition"] == 200 and report["training_seed"] == 1701
    for name, artifact in report["generated"].items():
        assert sha256(args.root / f"evaluation/{name}.npz") == artifact["sha256"]
        assert artifact["validity"]["valid_fraction"] == 1
    rows = []
    for method, label in LABELS.items():
        items = [
            next(row for row in report["rows"] if row["name"] == f"{method}_{seed}")
            for seed in report["generation_seeds"]
        ]
        utility = np.array([row["aggregate_w1"] for row in items])
        structure = np.array([row["macro_condition_structure_swd"] for row in items])
        summary = report["summary"][method]
        np.testing.assert_allclose(
            [utility.mean(), utility.std(ddof=1), structure.mean()],
            [
                summary["aggregate_w1_mean"],
                summary["generation_sd"],
                summary["structure_swd_mean"],
            ],
            rtol=0,
            atol=1e-14,
        )
        rows.append(
            {
                "method": method,
                "label": label,
                **summary,
                "structure_generation_sd": float(structure.std(ddof=1)),
            }
        )
    controls = {
        row["name"]: row["aggregate_w1"]
        for row in report["rows"]
        if row["name"] in ["conditional_mean", "pooled_grid_mean", "spatial_kernel"]
    }
    result = {
        "status": "passed",
        "protocol": plan["protocol"],
        "scope": report["scope"],
        "selected_epoch": report["selected_epoch"],
        "completed_epochs": 200,
        "training_seed": 1701,
        "generation_seeds": report["generation_seeds"],
        "per_condition": 200,
        "source_sha256": report["source_sha256"],
        "report_sha256": sha256(report_path),
        "plan_sha256": sha256(plan_path),
        "summary_code_sha256": sha256(Path(__file__)),
        "rows": rows,
        "deterministic_controls": controls,
        "valid_fraction_all_generated": 1.0,
        "test_year_groups": 9,
        "limits": report["limitations"],
        "finding": (
            "DiffeoCFM improves on its Gaussian prior but does not beat "
            "replay or the conditional mean on aggregate-variance W1."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "weather_generation_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    text = [
        "# Exploratory weather generation result — 2026-10-05",
        "",
        result["finding"],
        "",
        "The frozen 200-epoch run selected epoch "
        + str(report["selected_epoch"])
        + " using validation structural SWD. "
        "All generated matrices pass validity checks.",
        "",
        "| Method | Aggregate-variance W1 ↓ | Generation SD | Structural SWD ↓ |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['label']} | {row['aggregate_w1_mean']:.6f} | "
            f"{row['generation_sd']:.6f} | {row['structure_swd_mean']:.6f} |"
        )
    text += [
        "",
        f"Conditional mean W1: {controls['conditional_mean']:.6f}.",
        "",
        "One fitted model seed, five generation seeds, 200 matrices "
        "per region/quarter context. "
        "The final checkpoint is a declared sensitivity; it does not "
        "replace validation selection. "
        "Replay was recomputed at the same draw count (the initial "
        "baseline study used 1,000).",
        "",
        "This extension was declared after the baseline test results were known. "
        "It uses the same nine test years, with 12 dependent region/quarter contexts. "
        "Generation SD is Monte Carlo variability conditional on the "
        "fit, not uncertainty "
        "over training or future years. Structural SWD subsamples only "
        "nine matrices per "
        "condition; the utility calculation uses all 200 generated "
        "draws. No forecasting, "
        "extreme-event, energy-production or spatial-transfer benefit follows.",
        "",
        "The summary validates every frozen source and output hash, "
        "complete-job status, "
        "generated artifact hash, and numerical aggregation. Full "
        "per-context values and "
        "sampling diagnostics remain in the hashed original report.",
        "",
    ]
    (args.output / "weather_generation_results.md").write_text("\n".join(text))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, metric, error, title in zip(
        axes,
        ["aggregate_w1_mean", "structure_swd_mean"],
        ["generation_sd", "structure_generation_sd"],
        ["Aggregate-variance W1 ↓", "Conditional structural SWD ↓"],
        strict=True,
    ):
        y = np.arange(len(rows))
        values = [r[metric] for r in rows]
        ax.barh(
            y,
            values,
            xerr=[r[error] for r in rows],
            height=0.55,
            color=["#587487", "#177c75", "#86aea6", "#bd803d"],
            capsize=3,
        )
        ax.set_yticks(y, [r["label"] for r in rows] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values) * 1.2)
        for i, value in enumerate(values):
            ax.text(
                value + max(values) * 0.025, i, f"{value:.3f}", va="center", fontsize=9
            )
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].axvline(
        controls["conditional_mean"],
        color="#555",
        linestyle=":",
        label="Conditional mean",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Learning beyond a fitted prior is not enough",
        x=0.03,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.03,
        0.045,
        "64 cells · 1 training seed · 5 generation seeds · previously "
        "inspected 2016–2024 periods\n"
        "Bars show means; whiskers show generation SD. No forecast benefit is claimed.",
        fontsize=8.5,
        color="#52616b",
    )
    figure.tight_layout(rect=[0, 0.15, 1, 0.93], w_pad=2)
    for suffix in ["png", "svg", "pdf"]:
        figure.savefig(
            args.output / f"weather_generation.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    if args.table:
        tex = [
            r"\begin{table}[t]",
            r"\caption{Exploratory weather generation, one training "
            r"seed and five generation seeds. "
            r"Uncertainty is generation SD, conditional on the fitted "
            r"model. Lower is better. "
            r"Conditional mean aggregate-variance distance is .0824.}",
            r"\label{tab:weathergeneration}",
            r"\centering\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Method & Aggregate-variance W1 & Structural SWD \\",
            r"\midrule",
        ]
        for row in rows:
            tex.append(
                f"{row['label']} & \\({row['aggregate_w1_mean']:.4f} "
                f"\\pm {row['generation_sd']:.4f}\\) & "
                f"\\({row['structure_swd_mean']:.3f} "
                f"\\pm {row['structure_generation_sd']:.3f}\\)" + r" \\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        args.table.write_text("\n".join(tex) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
