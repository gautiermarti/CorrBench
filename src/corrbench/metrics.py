"""Distributional and financial-structure metrics for correlation generators."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corrbench.geometry import lower_to_symmetric, symmetric_to_lower

FloatArray = NDArray[np.floating]


def structural_features(correlations: FloatArray) -> tuple[FloatArray, list[str]]:
    """Extract compact permutation-invariant dependence and spectrum summaries."""
    correlations = np.asarray(correlations, dtype=np.float64)
    n = correlations.shape[-1]
    lower_indices = np.tril_indices(n, k=-1)
    lower = correlations[:, lower_indices[0], lower_indices[1]]
    eigenvalues = np.linalg.eigvalsh(correlations)
    clipped = np.clip(eigenvalues, 1e-12, None)
    probabilities = clipped / clipped.sum(axis=1, keepdims=True)

    corr_quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    eigen_quantiles = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95]
    columns = [
        lower.mean(axis=1),
        lower.std(axis=1),
        *[np.quantile(lower, quantile, axis=1) for quantile in corr_quantiles],
        *[np.quantile(eigenvalues, quantile, axis=1) for quantile in eigen_quantiles],
        eigenvalues[:, -1],
        eigenvalues[:, -5:].sum(axis=1),
        np.exp(-(probabilities * np.log(probabilities)).sum(axis=1)),
        clipped.sum(axis=1) ** 2 / (clipped**2).sum(axis=1),
        np.log(clipped).sum(axis=1),
        np.log(clipped[:, -1] / clipped[:, 0]),
    ]
    names = (
        ["corr_mean", "corr_std"]
        + [f"corr_q{int(100 * value):02d}" for value in corr_quantiles]
        + [f"eigen_q{int(100 * value):02d}" for value in eigen_quantiles]
        + [
            "largest_eigenvalue",
            "top5_eigenvalue_sum",
            "effective_rank_entropy",
            "effective_rank_participation",
            "log_determinant",
            "log_condition_number",
        ]
    )
    return np.column_stack(columns), names


def portfolio_functionals(
    correlations: FloatArray,
) -> tuple[FloatArray, list[str]]:
    """Return permutation-invariant minimum-variance portfolio diagnostics.

    Unit marginal volatilities isolate the information in the correlation
    scenario. Gross leverage and concentration make near-singular or otherwise
    economically implausible outputs visible even when their entries look
    reasonable.
    """
    correlations = np.asarray(correlations, dtype=np.float64)
    n = correlations.shape[-1]
    ones = np.ones((correlations.shape[0], n), dtype=np.float64)
    inverse_ones = np.linalg.solve(correlations, ones[..., None])[..., 0]
    denominator = np.sum(inverse_ones, axis=1)
    weights = inverse_ones / denominator[:, None]
    values = np.column_stack(
        [
            np.sqrt(1.0 / np.maximum(denominator, 1e-12)),
            np.sum(np.abs(weights), axis=1),
            np.sum(weights**2, axis=1),
            np.max(np.abs(weights), axis=1),
        ]
    )
    return values, [
        "gmv_volatility",
        "gmv_gross_leverage",
        "gmv_weight_concentration",
        "gmv_max_absolute_weight",
    ]


def classifier_two_sample_auc(
    real: FloatArray,
    generated: FloatArray,
    *,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-validated logistic two-sample test in a supplied representation."""
    count = min(real.shape[0], generated.shape[0])
    rng = np.random.default_rng(seed)
    real = real[rng.choice(real.shape[0], count, replace=False)]
    generated = generated[rng.choice(generated.shape[0], count, replace=False)]
    features = np.concatenate([real, generated])
    labels = np.concatenate(
        [np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)]
    )
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        cv=split,
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(labels, probabilities))
    return {"auc": auc, "separation": float(abs(auc - 0.5) * 2)}


def nearest_train_diagnostic(
    train: FloatArray,
    real_test: FloatArray,
    generated: FloatArray,
    *,
    scale: FloatArray | float = 1.0,
    max_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Compare generated-to-train and held-out-to-train nearest distances."""
    rng = np.random.default_rng(seed)
    train = (
        train[
            rng.choice(train.shape[0], min(max_samples, train.shape[0]), replace=False)
        ]
        / scale
    )
    real_test = (
        real_test[
            rng.choice(
                real_test.shape[0], min(max_samples, real_test.shape[0]), replace=False
            )
        ]
        / scale
    )
    generated = (
        generated[
            rng.choice(
                generated.shape[0], min(max_samples, generated.shape[0]), replace=False
            )
        ]
        / scale
    )
    model = NearestNeighbors(n_neighbors=1).fit(train)
    real_distance = model.kneighbors(real_test, return_distance=True)[0][:, 0]
    generated_distance = model.kneighbors(generated, return_distance=True)[0][:, 0]
    real_median = float(np.median(real_distance))
    generated_median = float(np.median(generated_distance))
    return {
        "real_test_median_distance": real_median,
        "generated_median_distance": generated_median,
        "generated_to_real_ratio": generated_median / max(real_median, 1e-12),
        "generated_exact_match_fraction": float(np.mean(generated_distance < 1e-10)),
    }


def sliced_wasserstein(
    real: FloatArray,
    generated: FloatArray,
    *,
    projections: int = 128,
    seed: int = 0,
    scale: float | FloatArray = 1.0,
    device: str = "cuda",
) -> float:
    """Mean one-dimensional W2 distance over fixed random projections."""
    count = min(real.shape[0], generated.shape[0])
    rng = np.random.default_rng(seed)
    real_indices = rng.choice(real.shape[0], size=count, replace=False)
    generated_indices = rng.choice(generated.shape[0], size=count, replace=False)
    real_tensor = torch.as_tensor(
        real[real_indices] / scale, dtype=torch.float32, device=device
    )
    generated_tensor = torch.as_tensor(
        generated[generated_indices] / scale, dtype=torch.float32, device=device
    )
    generator = torch.Generator(device).manual_seed(seed)
    direction = torch.randn(
        real.shape[1], projections, device=device, generator=generator
    )
    direction /= torch.linalg.vector_norm(direction, dim=0, keepdim=True)
    real_projected = torch.sort(real_tensor @ direction, dim=0).values
    generated_projected = torch.sort(generated_tensor @ direction, dim=0).values
    distances = torch.sqrt(
        torch.mean((real_projected - generated_projected) ** 2, dim=0)
    )
    return float(distances.mean())


def orbit_sliced_wasserstein(
    real: FloatArray,
    generated: FloatArray,
    *,
    projections: int = 128,
    seed: int = 0,
    scale: float | FloatArray = 1.0,
    device: str = "cuda",
) -> float:
    """Sliced Wasserstein after independent uniform asset permutations."""
    rng = np.random.default_rng(seed)

    def randomize(values: FloatArray) -> FloatArray:
        matrices = lower_to_symmetric(values)
        randomized = np.empty_like(matrices)
        for index, matrix in enumerate(matrices):
            permutation = rng.permutation(matrix.shape[-1])
            randomized[index] = matrix[permutation][:, permutation]
        return symmetric_to_lower(randomized)

    return sliced_wasserstein(
        randomize(real),
        randomize(generated),
        projections=projections,
        seed=seed,
        scale=scale,
        device=device,
    )


def energy_distance(
    real: FloatArray,
    generated: FloatArray,
    *,
    max_samples: int = 1500,
    seed: int = 0,
) -> float:
    """Multivariate energy distance with a deterministic sample cap."""
    rng = np.random.default_rng(seed)
    real_indices = rng.choice(
        real.shape[0], min(max_samples, real.shape[0]), replace=False
    )
    real = real[real_indices]
    generated = generated[
        rng.choice(
            generated.shape[0],
            min(max_samples, generated.shape[0]),
            replace=False,
        )
    ]
    cross = cdist(real, generated).mean()
    within_real = cdist(real, real).mean()
    within_generated = cdist(generated, generated).mean()
    return float(2 * cross - within_real - within_generated)


def validity_summary(
    correlations: FloatArray, *, tolerance: float = 1e-5
) -> dict[str, float]:
    """Summarize all hard correlation-matrix constraints."""
    symmetry = np.abs(correlations - np.swapaxes(correlations, -1, -2)).max(axis=(1, 2))
    diagonal = np.abs(np.diagonal(correlations, axis1=-2, axis2=-1) - 1).max(axis=1)
    minimum = np.linalg.eigvalsh(correlations)[:, 0]
    return {
        "valid_fraction": float(
            np.mean((symmetry <= tolerance) & (diagonal <= tolerance) & (minimum > 0))
        ),
        "max_symmetry_error": float(symmetry.max()),
        "max_diagonal_error": float(diagonal.max()),
        "minimum_eigenvalue": float(minimum.min()),
        "median_minimum_eigenvalue": float(np.median(minimum)),
    }


def manifold_precision_recall(
    real: FloatArray,
    generated: FloatArray,
    *,
    neighbors: int = 5,
    max_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate fidelity/coverage from adaptive k-nearest-neighbor manifolds."""
    rng = np.random.default_rng(seed)
    real = real[
        rng.choice(real.shape[0], min(max_samples, real.shape[0]), replace=False)
    ]
    generated = generated[
        rng.choice(
            generated.shape[0],
            min(max_samples, generated.shape[0]),
            replace=False,
        )
    ]
    if min(real.shape[0], generated.shape[0]) <= neighbors:
        raise ValueError("Not enough samples for the requested neighbor count.")

    real_model = NearestNeighbors(n_neighbors=neighbors + 1).fit(real)
    generated_model = NearestNeighbors(n_neighbors=neighbors + 1).fit(generated)
    real_radius = real_model.kneighbors(real, return_distance=True)[0][:, -1]
    generated_radius = generated_model.kneighbors(generated, return_distance=True)[0][
        :, -1
    ]

    generated_to_real_distance, generated_to_real_index = real_model.kneighbors(
        generated, n_neighbors=1, return_distance=True
    )
    real_to_generated_distance, real_to_generated_index = generated_model.kneighbors(
        real, n_neighbors=1, return_distance=True
    )
    precision = np.mean(
        generated_to_real_distance[:, 0] <= real_radius[generated_to_real_index[:, 0]]
    )
    recall = np.mean(
        real_to_generated_distance[:, 0]
        <= generated_radius[real_to_generated_index[:, 0]]
    )
    return {"precision": float(precision), "recall": float(recall)}
