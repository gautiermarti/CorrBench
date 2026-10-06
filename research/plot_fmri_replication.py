"""Plot a complete five-fit brain replication; preserve uncertainty units."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert len(report["methods"]) == 4 and report["test_subjects"] == 160
    reference = report["reference"]["roc_auc"]
    for row in report["methods"]:
        assert row["training_seeds"] == row["generation_seeds_per_training"] == 5
        assert [r["training_seed"] for r in row["per_training_seed"]] == list(
            range(1701, 1706)
        )
        for fit in row["per_training_seed"]:
            assert [r["generation_seed"] for r in fit["draws"]] == list(
                range(2027, 2032)
            )
            np.testing.assert_allclose(
                np.mean([r["auc"] for r in fit["draws"]]),
                fit["auc_mean"],
                rtol=0,
                atol=1e-14,
            )
        auc = [r["auc_mean"] for r in row["per_training_seed"]]
        np.testing.assert_allclose(
            [np.mean(auc), np.std(auc, ddof=1)],
            [row["auc_mean"], row["auc_training_sd"]],
            rtol=0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            row["auc_mean"] - reference, row["auc_difference"], rtol=0, atol=1e-14
        )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    colors = ["#587487", "#8d6f9e", "#177c75", "#bd803d"]
    labels = [r["label"] for r in report["methods"]]
    offsets = np.linspace(-0.10, 0.10, 5)
    for index, (row, color) in enumerate(zip(report["methods"], colors, strict=True)):
        auc = [r["auc_mean"] for r in row["per_training_seed"]]
        axes[0].errorbar(
            row["auc_mean"],
            index,
            xerr=row["auc_training_sd"],
            fmt="D",
            color=color,
            capsize=4,
            markersize=6,
            zorder=3,
        )
        axes[0].scatter(auc, index + offsets, color=color, alpha=0.6, s=25, zorder=2)
        lo, hi = row["training_and_paired_subject_95pct"]
        axes[1].plot([lo, hi], [index, index], color=color, linewidth=2)
        axes[1].scatter([row["auc_difference"]], [index], color=color, marker="D", s=35)
        axes[1].plot([lo, lo], [index - 0.08, index + 0.08], color=color)
        axes[1].plot([hi, hi], [index - 0.08, index + 0.08], color=color)
    axes[0].axvline(
        reference,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Real only: {reference:.4f}",
    )
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    axes[0].set_yticks(range(4), labels)
    axes[0].set(
        xlabel="Test AUC · five draws averaged within each fit",
        title="Five fits and their mean ± training SD",
        ylim=(3.7, -0.5),
    )
    axes[1].axvline(0, color="#333333", linestyle="--", linewidth=1.2)
    axes[1].set_yticks(range(4), [""] * 4)
    axes[1].set(
        xlabel="AUC difference from real only",
        title="Joint training / subject bootstrap",
        ylim=(3.7, -0.5),
    )
    for ax in axes:
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Brain connectivity: does augmentation help across repeated fits?",
        fontsize=15,
        fontweight="bold",
        x=0.03,
        ha="left",
    )
    fig.text(
        0.03,
        0.025,
        "ABIDE · same 160 NYU subjects · five training seeds × five generation seeds\n"
        "Right: exploratory 95% paired ranges, unadjusted for multiplicity; "
        "one inspected site, not uncertainty over future sites.",
        fontsize=8.5,
        color="#52616b",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.91], w_pad=3)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for extension in ["png", "svg", "pdf"]:
        path = args.output / f"fmri_replication.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n"
            )
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    audit = {
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": hashes,
        "all_fit_and_draw_counts_and_aggregates_checked": True,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
