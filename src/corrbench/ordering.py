"""Canonical orderings used by order-dependent image baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster import hierarchy

FloatArray = NDArray[np.floating]


def ward_permutation(correlation: FloatArray) -> NDArray[np.int64]:
    """Return the CorrGAN-style Ward optimal-leaf permutation.

    CorrGAN represents each permutation orbit by hierarchically ordering the
    variables.  Its public example uses the condensed dissimilarities ``1-C``,
    Ward linkage, and optimal leaf ordering.  The input is not modified.
    """
    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation contains non-finite values")
    upper = np.triu_indices(matrix.shape[0], k=1)
    condensed = np.clip(1.0 - matrix[upper], 0.0, 2.0)
    linkage = hierarchy.linkage(condensed, method="ward")
    ordered = hierarchy.optimal_leaf_ordering(linkage, condensed)
    return hierarchy.leaves_list(ordered).astype(np.int64, copy=False)


def ward_order(correlation: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
    """Return a permutation-similar Ward-ordered matrix and its permutation."""
    permutation = ward_permutation(correlation)
    matrix = np.asarray(correlation)
    return matrix[np.ix_(permutation, permutation)], permutation
