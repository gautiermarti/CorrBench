"""Render weather baseline evidence; no model fitting or test selection."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "identity": "Independent locations",
    "spatial_kernel": "Spatial kernel",
    "pooled_grid_mean": "Pooled matrix mean",
    "conditional_mean": "Region / quarter mean",
    "conditional_replay": "Region / quarter replay",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharex=True)
    for ax, split, title in zip(
        axes,
        ["test", "spatial_transfer"],
        ["Later years, known regions", "Later years, South America"],
        strict=True,
    ):
        rows = report["summary"][split]
        keys = list(rows)
        values = [rows[k]["normalized_aggregate_w1"] for k in keys]
        bars = ax.barh(
            np.arange(len(keys)),
            values,
            height=0.57,
            color=["#177c75" if "replay" in k else "#587487" for k in keys],
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 0.022,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(np.arange(len(keys)), [LABELS[k] for k in keys])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=16)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("Normalized aggregate-variance W1 ↓", labelpad=12)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    figure.suptitle(
        "Dependence changes spatial aggregation",
        x=0.03,
        ha="left",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.03,
        0.04,
        "64 grid cells · 2016–2024 · 9 shared year groups\n"
        "Unit local variance; empirical quarterly targets. "
        "No learned weather generator or forecast benefit is claimed.",
        fontsize=8.5,
        color="#52616b",
    )
    figure.tight_layout(rect=[0, 0.14, 1, 0.91], w_pad=2.6)
    for suffix in ["png", "svg", "pdf"]:
        figure.savefig(
            args.output / f"weather_pilot.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    if args.table:
        rows = [
            r"\begin{table}[t]",
            r"\caption{Weather pilot: normalized aggregate-variance distribution "
            r"distance (lower is better), over 2016--2024. Regional replay/means "
            r"have no fitted South America distribution. These are baseline "
            r"and dependence-sensitivity results, not learned-generation gains.}",
            r"\label{tab:weatherutility}",
            r"\centering\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Baseline & Known regions & South America \\",
            r"\midrule",
        ]
        for key, label in LABELS.items():
            known = report["summary"]["test"][key]["normalized_aggregate_w1"]
            other = report["summary"]["spatial_transfer"].get(key)
            transfer = f"{other['normalized_aggregate_w1']:.4f}" if other else "---"
            rows.append(f"{label} & {known:.4f} & {transfer}" + r" \\")
        rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
