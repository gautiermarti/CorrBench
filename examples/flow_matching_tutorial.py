"""An exact one-coordinate flow-matching lesson, without neural training.

The target is a two-Gaussian mixture in Fisher coordinates. Its independent
linear-bridge conditional velocity is analytic. Numerical ODE sampling can thus
be checked against the exact marginal CDF before decoding rho = tanh(z).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

WEIGHTS = np.array([0.45, 0.55])
MEANS = np.array([-1.2, 1.0])
SIGMAS = np.array([0.25, 0.2])


def bridge_parameters(t):
    return t * MEANS, (1 - t) ** 2 + t**2 * SIGMAS**2


def log_components(z, t):
    mean, variance = bridge_parameters(t)
    return (
        np.log(WEIGHTS)
        - 0.5 * np.log(2 * np.pi * variance)
        - 0.5 * (np.asarray(z)[..., None] - mean) ** 2 / variance
    )


def velocity(z, t):
    mean, variance = bridge_parameters(t)
    logp = log_components(z, t)
    responsibility = np.exp(logp - logsumexp(logp, axis=-1, keepdims=True))
    coefficient = (t * SIGMAS**2 - (1 - t)) / variance
    component_velocity = MEANS + coefficient * (np.asarray(z)[..., None] - mean)
    return np.sum(responsibility * component_velocity, axis=-1)


def cdf(z, t):
    mean, variance = bridge_parameters(t)
    return np.sum(
        WEIGHTS * ndtr((np.asarray(z)[..., None] - mean) / np.sqrt(variance)), axis=-1
    )


def density(z, t):
    return np.exp(logsumexp(log_components(z, t), axis=-1))


def integrate(source, steps, keep_paths=False):
    z = source.copy()
    paths = [z.copy()]
    dt = 1 / steps
    for index in range(steps):
        t = index * dt
        first = velocity(z, t)
        second = velocity(z + dt * first, t + dt)
        z += 0.5 * dt * (first + second)
        if keep_paths:
            paths.append(z.copy())
    return np.array(paths) if keep_paths else z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-3, 3, 1001)
    np.testing.assert_allclose(velocity(grid, 0), WEIGHTS @ MEANS - grid, atol=1e-14)
    np.testing.assert_allclose(velocity(grid, 1), grid, atol=1e-14)
    # A local continuity-equation check is independent of ODE discretization.
    h = 1e-5
    residuals = []
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        time_derivative = (density(grid, t + h) - density(grid, t - h)) / (2 * h)
        space_derivative = (
            density(grid + h, t) * velocity(grid + h, t)
            - density(grid - h, t) * velocity(grid - h, t)
        ) / (2 * h)
        residuals.append(float(np.max(np.abs(time_derivative + space_derivative))))
    assert max(residuals) < 1e-7
    probabilities = (np.arange(4000) + 0.5) / 4000
    source = ndtri(probabilities)
    convergence = []
    for steps in [10, 25, 100, 400]:
        endpoint = integrate(source, steps)
        assert np.all(np.diff(endpoint) > 0), "One-dimensional flow crossed paths"
        rho = np.tanh(endpoint)
        convergence.append(
            {
                "heun_steps": steps,
                "vector_field_calls": 2 * steps,
                "maximum_target_cdf_error": float(
                    np.max(np.abs(cdf(endpoint, 1) - probabilities))
                ),
                "minimum_decoded_eigenvalue": float(np.min(1 - np.abs(rho))),
            }
        )
    assert all(
        b["maximum_target_cdf_error"] < a["maximum_target_cdf_error"]
        for a, b in zip(convergence[:-1], convergence[1:], strict=True)
    )
    assert convergence[-1]["maximum_target_cdf_error"] < 1e-5
    report = {
        "scope": "Analytic scalar Fisher-coordinate flow; "
        "not a trained CorrFlow result",
        "source": "N(0,1)",
        "target": {
            "weights": WEIGHTS.tolist(),
            "means": MEANS.tolist(),
            "standard_deviations": SIGMAS.tolist(),
        },
        "bridge": "Independent Z0 and Z1; Zt=(1-t)Z0+tZ1",
        "maximum_continuity_residual": max(residuals),
        "quantile_grid_size": len(source),
        "convergence": convergence,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cdf_error_scope": "Deterministic midpoint source-quantile grid; numerical "
        "transport error, not an empirical two-sample score",
    }
    (args.figures / "flow_matching_numbers.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    colors = ["#587487", "#bd803d", "#177c75"]
    z = np.linspace(-3.5, 3.5, 1200)
    for t, color in zip([0, 0.5, 1], colors, strict=True):
        axes[0, 0].plot(z, density(z, t), color=color, label=f"t = {t:g}")
    axes[0, 0].set(
        xlabel="Fisher coordinate z",
        ylabel="Density",
        title="The distribution moves, not a matched row",
    )
    axes[0, 0].legend(frameon=False)
    rng = np.random.default_rng(7201)
    z0 = rng.normal(size=20)
    classes = rng.choice(2, size=20, p=WEIGHTS)
    z1 = rng.normal(MEANS[classes], SIGMAS[classes])
    for first, last in zip(z0, z1, strict=True):
        axes[0, 1].plot([0, 1], [first, last], color="#587487", alpha=0.4)
    axes[0, 1].set(
        xlabel="Bridge time",
        ylabel="z",
        title="Independent straight bridges used in training",
    )
    starting = ndtri(np.linspace(0.02, 0.98, 25))
    paths = integrate(starting, 400, keep_paths=True)
    for index in range(paths.shape[1]):
        axes[1, 0].plot(
            np.linspace(0, 1, len(paths)),
            paths[:, index],
            color="#177c75",
            alpha=0.65,
            linewidth=1,
        )
    axes[1, 0].set(
        xlabel="ODE time",
        ylabel="z",
        title="The conditional mean field gives curved paths",
    )
    rho_grid = np.linspace(-0.995, 0.995, 1500)
    for t, color in zip([0, 0.5, 1], colors, strict=True):
        transformed = density(np.arctanh(rho_grid), t) / (1 - rho_grid**2)
        axes[1, 1].plot(rho_grid, transformed, color=color, label=f"t = {t:g}")
    axes[1, 1].set(
        xlabel="Correlation ρ = tanh(z)",
        ylabel="Density",
        title="Decode into valid 2 × 2 correlations",
    )
    fig.suptitle(
        "Flow matching: bridges, a velocity field, and valid correlations",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.015,
        "Analytic toy distribution · no neural training or financial data · "
        "Heun paths: 400 steps",
        color="#52616b",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.96], h_pad=2.5, w_pad=2.5)
    for extension in ["png", "svg", "pdf"]:
        path = args.figures / f"flow_matching.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
