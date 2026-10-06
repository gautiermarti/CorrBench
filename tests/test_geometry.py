import numpy as np
import pytest
import torch

from corrbench.geometry import (
    corr_to_gft,
    corr_to_normalized_cholesky,
    gft_to_corr,
    is_correlation_matrix,
    lower_to_symmetric,
    nearest_correlation_torch,
    normalized_cholesky_to_corr,
    symmetric_to_lower,
)


def random_correlation(batch: int, dimension: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(batch, dimension, dimension + 2))
    covariance = factors @ np.swapaxes(factors, -1, -2) + 0.2 * np.eye(dimension)
    scale = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return covariance / (scale[..., :, None] * scale[..., None, :])


def test_lower_vectorization_round_trip_numpy() -> None:
    matrix = np.arange(32, dtype=float).reshape(2, 4, 4)
    matrix = np.tril(matrix, -1)
    matrix = matrix + np.swapaxes(matrix, -1, -2)
    assert np.array_equal(lower_to_symmetric(symmetric_to_lower(matrix)), matrix)


def test_lower_vectorization_round_trip_torch() -> None:
    matrix = torch.arange(32, dtype=torch.float64).reshape(2, 4, 4)
    matrix = torch.tril(matrix, diagonal=-1)
    matrix = matrix + matrix.transpose(-1, -2)
    assert torch.equal(lower_to_symmetric(symmetric_to_lower(matrix)), matrix)


def test_gft_round_trip_numpy() -> None:
    corr = random_correlation(batch=3, dimension=6)
    coordinates = corr_to_gft(corr)
    recovered, info = gft_to_corr(coordinates, return_info=True)
    assert np.all(info["converged"])
    assert np.allclose(recovered, corr, atol=2e-7, rtol=2e-7)
    assert is_correlation_matrix(recovered)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gft_round_trip_torch(dtype: torch.dtype) -> None:
    corr = torch.as_tensor(random_correlation(2, 5), dtype=dtype)
    coordinates = corr_to_gft(corr, return_matrix=True)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-9
    recovered, info = gft_to_corr(coordinates, tolerance=tolerance, return_info=True)
    assert bool(torch.all(info["converged"]))
    assert torch.allclose(recovered, corr, atol=2e-5, rtol=2e-5)
    assert is_correlation_matrix(recovered, atol=2e-5)


def test_gft_is_permutation_equivariant() -> None:
    corr = torch.as_tensor(random_correlation(2, 7), dtype=torch.float64)
    permutation = torch.tensor([3, 0, 6, 2, 4, 1, 5])
    permuted = corr[:, permutation][:, :, permutation]
    transformed = corr_to_gft(corr, return_matrix=True)
    transformed_permuted = corr_to_gft(permuted, return_matrix=True)
    expected = transformed[:, permutation][:, :, permutation]
    assert torch.allclose(transformed_permuted, expected, atol=1e-10, rtol=1e-10)


def test_nearest_correlation_repairs_an_indefinite_matrix() -> None:
    matrix = torch.tensor(
        [[1.0, 1.2, -0.8], [1.2, 1.0, 0.9], [-0.8, 0.9, 1.0]],
        dtype=torch.float64,
    )
    repaired, info = nearest_correlation_torch(matrix, return_info=True)
    assert bool(info["converged"])
    assert is_correlation_matrix(repaired)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_normalized_cholesky_round_trip(backend: str) -> None:
    corr = random_correlation(3, 6)
    if backend == "torch":
        corr = torch.from_numpy(corr)
    coordinates = corr_to_normalized_cholesky(corr)
    recovered = normalized_cholesky_to_corr(coordinates)
    if backend == "torch":
        assert torch.allclose(recovered, corr, atol=1e-10, rtol=1e-10)
    else:
        np.testing.assert_allclose(recovered, corr, atol=1e-10, rtol=1e-10)
