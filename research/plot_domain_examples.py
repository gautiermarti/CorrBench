"""Show the first training matrix in each pilot on a common correlation scale."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finance", type=Path, required=True)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = [
        ("Finance", "80 stock returns", args.finance, 80),
        ("Brain connectivity", "111 regional BOLD signals", args.brain, 111),
        ("Spatial weather", "64 temperature anomalies", args.weather, 64),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3))
    records = {}
    for axis, (domain, description, path, size) in zip(axes, panels, strict=True):
        with np.load(path) as arrays:
            matrix = np.asarray(arrays["matrices"][0], dtype=np.float64)
        assert matrix.shape == (size, size) and np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6, rtol=0)
        np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-6, rtol=0)
        minimum = float(np.linalg.eigvalsh((matrix + matrix.T) / 2).min())
        assert minimum > 0
        picture = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_title(domain, fontsize=13, fontweight="bold", pad=12)
        axis.set_xlabel(description, labelpad=9)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        records[domain] = {
            "source_sha256": sha(path),
            "row": 0,
            "dimension": size,
            "minimum_eigenvalue": minimum,
            "node_order": "unchanged",
        }
    fig.subplots_adjust(left=0.035, right=0.9, bottom=0.19, top=0.76, wspace=0.2)
    bar = fig.colorbar(picture, cax=fig.add_axes([0.92, 0.23, 0.018, 0.50]))
    bar.set_label("Correlation")
    bar.set_ticks([-1, 0, 1])
    fig.text(
        0.035,
        0.93,
        "CorrBench: shared geometry, different observations",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.83,
        "One observed training matrix from each implemented pilot",
        fontsize=11,
        color="#52616b",
    )
    fig.text(
        0.035,
        0.035,
        "First stored training row · original node order · common color scale\n"
        "Brain and weather targets use OAS regularization. "
        "These examples do not measure generator quality.",
        fontsize=8.5,
        color="#52616b",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for extension in ["png", "svg", "pdf"]:
        path = args.output / f"domain_examples.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n"
            )
        hashes[path.name] = sha(path)
    plt.close(fig)
    report = {
        "scope": "Illustration, not a distributional or utility result",
        "selection": "First stored training row in each frozen dataset",
        "panels": records,
        "plot_code_sha256": sha(Path(__file__)),
        "files": hashes,
        "publicly_distributed": False,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
