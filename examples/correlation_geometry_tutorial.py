"""Executable 3x3 correlation lesson; optional publication-quality figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corrbench.geometry import corr_to_gft, gft_to_corr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    bad = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
    weights = np.array([-1, 1, 1])
    bivariate = torch.tensor([[[1, 0.6], [0.6, 1]]], dtype=torch.float64)
    gamma2 = corr_to_gft(bivariate)
    coordinates = torch.tensor([[0.7, -0.4, 0.2]], dtype=torch.float64)
    good, info = gft_to_corr(
        coordinates, tolerance=1e-10, max_iterations=1000, return_info=True
    )
    if not bool(torch.all(info["converged"])):
        raise RuntimeError("Tutorial inverse failed to converge")
    roundtrip = corr_to_gft(good)
    report = {
        "invalid_matrix_eigenvalues": np.linalg.eigvalsh(bad).tolist(),
        "invalid_matrix_determinant": float(np.linalg.det(bad)),
        "impossible_portfolio_variance": float(weights @ bad @ weights),
        "bivariate_gft": float(gamma2.item()),
        "atanh_0_6": float(np.arctanh(0.6)),
        "gft_coordinates": coordinates.tolist(),
        "valid_generated_matrix": good[0].numpy().tolist(),
        "generated_minimum_eigenvalue": float(torch.linalg.eigvalsh(good).min()),
        "inverse_iterations": int(info["iterations"]),
        "roundtrip_maximum_error": float(torch.max(torch.abs(roundtrip - coordinates))),
        "scope": "Geometry demonstration with fixed coordinates, no trained generator",
    }
    print(json.dumps(report, indent=2))
    if args.figures is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    grid = np.linspace(-1, 1, 501)
    a, b = np.meshgrid(grid, grid)
    determinant = 1 + 2 * 0.6 * a * b - 0.6**2 - a**2 - b**2
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.contourf(
        a, b, determinant >= 0, levels=[-0.5, 0.5, 1.5], colors=["#fae8e2", "#d5e9e3"]
    )
    ax.contour(a, b, determinant, levels=[0], colors=["#245e55"], linewidths=2)
    ax.plot([0.9], [-0.9], "x", color="#a72b30", markersize=10, markeredgewidth=2)
    ax.text(0.15, -0.85, "Impossible combination", color="#a72b30")
    ax.text(-0.17, 0.28, "Valid\ncorrelations", color="#245e55", ha="center")
    ax.set(
        xlabel=r"$\rho_{13}$",
        ylabel=r"$\rho_{23}$",
        title=r"Allowed correlations when $\rho_{12}=0.6$",
        xlim=(-1, 1),
        ylim=(-1, 1),
        aspect="equal",
    )
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.figures / f"elliptope_slice.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, matrix, title in [
        (axes[0], bad, "Plausible entries, invalid matrix"),
        (axes[1], good[0].numpy(), "A valid GFT inverse"),
    ]:
        plot = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[i, j]) > 0.65 else "black",
                )
        ax.set(title=title, xticks=range(3), yticks=range(3))
    fig.colorbar(plot, ax=axes, shrink=0.8, label="Correlation")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.figures / f"invalid_and_gft.{ext}", dpi=180)
    plt.close(fig)
    (args.figures / "geometry_numbers.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
