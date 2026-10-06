"""Geometry and exact coordinates for full-rank correlation matrices.

The generalized Fisher transform (GFT) of a full-rank correlation matrix C is
the strictly lower triangular part of log(C). Archakov and Hansen (2021) prove
that this is a bijection between the open elliptope and R^{n(n-1)/2}. The inverse
below implements their fixed-point iteration for the missing diagonal of log(C).

References
----------
I. Archakov and P. R. Hansen, "A New Parametrization of Correlation Matrices,"
Econometrica, 89(4), 2021. https://doi.org/10.3982/ECTA16910
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import expm

Array = NDArray[np.floating]


def _matrix_dim(vector_dim: int) -> int:
    """Infer n from vector_dim = n(n-1)/2, rejecting invalid dimensions."""
    n = int((1 + math.sqrt(1 + 8 * vector_dim)) / 2)
    if n * (n - 1) // 2 != vector_dim:
        raise ValueError(
            f"Vector length {vector_dim} is not triangular: expected n(n-1)/2."
        )
    return n


def symmetric_to_lower(matrix: Array | torch.Tensor) -> Array | torch.Tensor:
    """Vectorize the strict lower triangle of one matrix or a batch of matrices."""
    n = matrix.shape[-1]
    if matrix.shape[-2] != n:
        raise ValueError("Expected square matrices in the last two dimensions.")
    if isinstance(matrix, torch.Tensor):
        idx = torch.tril_indices(n, n, offset=-1, device=matrix.device)
        return matrix[..., idx[0], idx[1]]
    idx = np.tril_indices(n, k=-1)
    return matrix[..., idx[0], idx[1]]


def lower_to_symmetric(
    vector: Array | torch.Tensor,
) -> Array | torch.Tensor:
    """Convert strict-lower vectors to hollow symmetric matrices."""
    n = _matrix_dim(vector.shape[-1])
    batch_shape = vector.shape[:-1]
    if isinstance(vector, torch.Tensor):
        matrix = torch.zeros(
            (*batch_shape, n, n), dtype=vector.dtype, device=vector.device
        )
        idx = torch.tril_indices(n, n, offset=-1, device=vector.device)
        matrix[..., idx[0], idx[1]] = vector
        return matrix + matrix.transpose(-1, -2)
    matrix = np.zeros((*batch_shape, n, n), dtype=vector.dtype)
    idx = np.tril_indices(n, k=-1)
    matrix[..., idx[0], idx[1]] = vector
    return matrix + np.swapaxes(matrix, -1, -2)


def _sym_function_torch(matrix: torch.Tensor, function: Callable) -> torch.Tensor:
    """Apply a scalar function spectrally to symmetric matrices."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    transformed = function(eigenvalues)
    return (eigenvectors * transformed.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def corr_to_gft(
    corr: Array | torch.Tensor,
    *,
    min_eigenvalue: float = 1e-10,
    return_matrix: bool = False,
) -> Array | torch.Tensor:
    """Map full-rank correlation matrices to unrestricted GFT coordinates.

    Parameters
    ----------
    corr:
        A matrix or batch with shape (..., n, n).
    min_eigenvalue:
        Numerical floor. Inputs below this floor are regularized before taking
        their logarithm; callers should record when this is needed.
    return_matrix:
        Return the hollow symmetric off-log matrix rather than its strict-lower
        vectorization.
    """
    if corr.shape[-1] != corr.shape[-2]:
        raise ValueError("Expected square matrices in the last two dimensions.")

    if isinstance(corr, torch.Tensor):
        corr = 0.5 * (corr + corr.transpose(-1, -2))
        log_corr = _sym_function_torch(
            corr, lambda values: torch.log(values.clamp_min(min_eigenvalue))
        )
        off_log = log_corr - torch.diag_embed(
            torch.diagonal(log_corr, dim1=-2, dim2=-1)
        )
    else:
        corr = np.asarray(corr)
        corr = 0.5 * (corr + np.swapaxes(corr, -1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        log_corr = (eigenvectors * np.log(eigenvalues)[..., None, :]) @ np.swapaxes(
            eigenvectors, -1, -2
        )
        diagonal = np.zeros_like(log_corr)
        idx = np.arange(corr.shape[-1])
        diagonal[..., idx, idx] = log_corr[..., idx, idx]
        off_log = log_corr - diagonal

    return off_log if return_matrix else symmetric_to_lower(off_log)


def _gft_to_corr_numpy_single(
    off_log: Array,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, int, bool]:
    matrix = np.array(off_log, dtype=np.float64, copy=True)
    n = matrix.shape[0]
    diag_idx = np.diag_indices(n)
    diagonal = np.zeros(n, dtype=matrix.dtype)

    for iteration in range(max_iterations):
        corr = expm(matrix)
        delta = np.log(np.diag(corr))
        if np.linalg.norm(delta) <= math.sqrt(n) * tolerance:
            corr = 0.5 * (corr + corr.T)
            np.fill_diagonal(corr, 1.0)
            return corr, iteration, True
        diagonal -= delta
        matrix[diag_idx] = diagonal

    corr = expm(matrix)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr, max_iterations, False


def gft_to_corr(
    gft: Array | torch.Tensor,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    return_info: bool = False,
) -> Array | torch.Tensor | tuple[Array | torch.Tensor, dict[str, object]]:
    """Invert GFT coordinates to full-rank correlation matrices.

    The input may be strict-lower vectors (..., n(n-1)/2) or hollow symmetric
    matrices (..., n, n). The torch path is differentiable through its fixed
    number of spectral operations, though generation normally does not require
    gradients through this inverse.
    """
    is_matrix = gft.ndim >= 2 and gft.shape[-1] == gft.shape[-2]
    off_log = gft if is_matrix else lower_to_symmetric(gft)

    if isinstance(off_log, torch.Tensor):
        matrix = 0.5 * (off_log + off_log.transpose(-1, -2))
        n = matrix.shape[-1]
        diagonal = torch.zeros(
            *matrix.shape[:-2], n, dtype=matrix.dtype, device=matrix.device
        )
        converged = torch.zeros(
            matrix.shape[:-2], dtype=torch.bool, device=matrix.device
        )
        iterations = 0
        corr = torch.empty_like(matrix)

        for iteration in range(max_iterations):
            working = matrix + torch.diag_embed(diagonal)
            corr = _sym_function_torch(working, torch.exp)
            delta = torch.log(torch.diagonal(corr, dim1=-2, dim2=-1))
            residual = torch.linalg.vector_norm(delta, dim=-1)
            converged = residual <= math.sqrt(n) * tolerance
            iterations = iteration
            if bool(torch.all(converged)):
                break
            diagonal = diagonal - delta

        corr = 0.5 * (corr + corr.transpose(-1, -2))
        corr = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
        corr = corr + torch.eye(n, dtype=corr.dtype, device=corr.device)
        info = {
            "iterations": iterations,
            "converged": converged,
            "max_diagonal_error": float(
                torch.max(torch.abs(torch.diagonal(corr, dim1=-2, dim2=-1) - 1)).item()
            ),
        }
    else:
        off_log = np.asarray(off_log)
        original_shape = off_log.shape[:-2]
        flat = off_log.reshape(-1, off_log.shape[-2], off_log.shape[-1])
        results = [
            _gft_to_corr_numpy_single(
                matrix, tolerance=tolerance, max_iterations=max_iterations
            )
            for matrix in flat
        ]
        corr = np.stack([result[0] for result in results]).reshape(
            *original_shape, off_log.shape[-2], off_log.shape[-1]
        )
        info = {
            "iterations": np.array([result[1] for result in results]).reshape(
                original_shape
            ),
            "converged": np.array([result[2] for result in results]).reshape(
                original_shape
            ),
            "max_diagonal_error": float(
                np.max(np.abs(np.diagonal(corr, axis1=-2, axis2=-1) - 1))
            ),
        }

    return (corr, info) if return_info else corr


def is_correlation_matrix(
    matrix: Array | torch.Tensor,
    *,
    atol: float = 1e-6,
    positive_definite: bool = True,
) -> bool:
    """Return whether every matrix in a batch satisfies correlation constraints."""
    if matrix.shape[-1] != matrix.shape[-2]:
        return False
    if isinstance(matrix, torch.Tensor):
        symmetric = torch.allclose(matrix, matrix.transpose(-1, -2), atol=atol, rtol=0)
        diagonal = torch.allclose(
            torch.diagonal(matrix, dim1=-2, dim2=-1),
            torch.ones_like(torch.diagonal(matrix, dim1=-2, dim2=-1)),
            atol=atol,
            rtol=0,
        )
        minimum = torch.linalg.eigvalsh(matrix).amin().item()
    else:
        matrix = np.asarray(matrix)
        symmetric = np.allclose(matrix, np.swapaxes(matrix, -1, -2), atol=atol, rtol=0)
        diagonal = np.allclose(
            np.diagonal(matrix, axis1=-2, axis2=-1), 1, atol=atol, rtol=0
        )
        minimum = float(np.linalg.eigvalsh(matrix).min())
    eigen_ok = minimum > -atol if not positive_definite else minimum > 0
    return bool(symmetric and diagonal and eigen_ok)


def nearest_correlation_torch(
    matrix: torch.Tensor,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    minimum_eigenvalue: float = 1e-6,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Higham alternating projections followed by strict-PD normalization."""
    single = matrix.ndim == 2
    if single:
        matrix = matrix.unsqueeze(0)
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    n = matrix.shape[-1]
    identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    estimate = (
        matrix - torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1)) + identity
    )
    correction = torch.zeros_like(estimate)
    converged = torch.zeros(matrix.shape[0], dtype=torch.bool, device=matrix.device)

    for _iteration in range(max_iterations):
        residual = estimate - correction
        eigenvalues, eigenvectors = torch.linalg.eigh(residual)
        projected = (
            eigenvectors * eigenvalues.clamp_min(0).unsqueeze(-2)
        ) @ eigenvectors.transpose(-1, -2)
        correction = projected - residual
        previous = estimate
        estimate = (
            projected
            - torch.diag_embed(torch.diagonal(projected, dim1=-2, dim2=-1))
            + identity
        )
        relative_change = torch.linalg.matrix_norm(
            estimate - previous, dim=(-2, -1)
        ) / torch.linalg.matrix_norm(previous, dim=(-2, -1)).clamp_min(1e-12)
        converged = relative_change <= tolerance
        if bool(torch.all(converged)):
            break

    eigenvalues, eigenvectors = torch.linalg.eigh(estimate)
    estimate = (
        eigenvectors * eigenvalues.clamp_min(minimum_eigenvalue).unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)
    scale = torch.sqrt(torch.diagonal(estimate, dim1=-2, dim2=-1))
    estimate = estimate / scale.unsqueeze(-1) / scale.unsqueeze(-2)
    estimate = 0.5 * (estimate + estimate.transpose(-1, -2))
    estimate = (
        estimate
        - torch.diag_embed(torch.diagonal(estimate, dim1=-2, dim2=-1))
        + identity
    )
    output = estimate[0] if single else estimate
    info = {
        "iterations": _iteration + 1,
        "converged": converged[0] if single else converged,
        "mean_frobenius_repair": float(
            torch.linalg.matrix_norm(estimate - matrix, dim=(-2, -1)).mean()
        ),
    }
    return (output, info) if return_info else output


def corr_to_normalized_cholesky(
    corr: Array | torch.Tensor,
) -> Array | torch.Tensor:
    """Map correlations to the unrestricted strict lower of row-normalized L."""
    if isinstance(corr, torch.Tensor):
        factor = torch.linalg.cholesky(corr)
        diagonal = torch.diagonal(factor, dim1=-2, dim2=-1)
        normalized = factor / diagonal.unsqueeze(-1)
    else:
        factor = np.linalg.cholesky(corr)
        diagonal = np.diagonal(factor, axis1=-2, axis2=-1)
        normalized = factor / diagonal[..., :, None]
    return symmetric_to_lower(normalized)


def normalized_cholesky_to_corr(
    coordinates: Array | torch.Tensor,
) -> Array | torch.Tensor:
    """Invert unrestricted normalized-Cholesky coordinates exactly."""
    lower = lower_to_symmetric(coordinates)
    if isinstance(lower, torch.Tensor):
        lower = torch.tril(lower, diagonal=-1)
        identity = torch.eye(lower.shape[-1], dtype=lower.dtype, device=lower.device)
        factor = lower + identity
        covariance = factor @ factor.transpose(-1, -2)
        scale = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1))
        return covariance / scale.unsqueeze(-1) / scale.unsqueeze(-2)
    lower = np.tril(lower, k=-1)
    factor = lower + np.eye(lower.shape[-1])
    covariance = factor @ np.swapaxes(factor, -1, -2)
    scale = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return covariance / scale[..., :, None] / scale[..., None, :]
