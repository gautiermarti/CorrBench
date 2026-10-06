"""Standalone, experimental Kaggle metric; embed a train-fitted CONFIG before use.

This prototype requires only NumPy and pandas. It is a new metric version,
not a numerical replacement for the frozen paper's Torch evaluation.
"""

import numpy as np
import pandas as pd

CONFIG = None


class ParticipantVisibleError(Exception):
    """A safe error that can be shown to a competition participant."""


def columns(n):
    return [f"c_{i:04d}" for i in range(n * (n - 1) // 2)]


def unpack(values, n):
    matrices = np.broadcast_to(np.eye(n), (len(values), n, n)).copy()
    lower = np.tril_indices(n, -1)
    matrices[:, lower[0], lower[1]] = values
    matrices[:, lower[1], lower[0]] = values
    return matrices


def features(matrices):
    n = matrices.shape[-1]
    lower = matrices[:, *np.tril_indices(n, -1)]
    eig = np.linalg.eigvalsh(matrices)
    clipped = np.maximum(eig, 1e-12)
    probability = clipped / clipped.sum(axis=1, keepdims=True)
    values = [
        lower.mean(axis=1),
        lower.std(axis=1),
        *[
            np.quantile(lower, q, axis=1)
            for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        ],
        *[np.quantile(eig, q, axis=1) for q in [0, 0.05, 0.25, 0.5, 0.75, 0.95]],
        eig[:, -1],
        eig[:, -5:].sum(axis=1),
        np.exp(-(probability * np.log(probability)).sum(axis=1)),
        clipped.sum(axis=1) ** 2 / (clipped**2).sum(axis=1),
        np.log(clipped).sum(axis=1),
        np.log(clipped[:, -1] / clipped[:, 0]),
    ]
    return np.column_stack(values)


def fit_configuration(training_matrices):
    """Organizer preparation only. Never fit normalizers to evaluation targets."""
    matrix = np.asarray(training_matrices, dtype=np.float64)
    if matrix.ndim != 3 or len(matrix) < 2 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("Expected at least two square training matrices")
    if not np.isfinite(matrix).all():
        raise ValueError("Training matrices must be finite")
    return {
        "metric_version": "conditional-structural-swd-numpy64-prototype-v1",
        "n": matrix.shape[-1],
        "feature_scale": np.maximum(
            features(matrix).std(axis=0, ddof=1), 1e-6
        ).tolist(),
        "classes": [0, 1, 2],
        "minimum_per_class": 20,
        "projections": 128,
        "projection_seed": 8137,
        "psd_tolerance": 1e-6,
    }


def _matrices(frame, names, participant):
    error = ParticipantVisibleError if participant else ValueError
    try:
        values = frame[names].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise error("Matrix entries must be numeric.") from exc
    if not np.isfinite(values).all():
        raise error("Matrix entries must be finite.")
    tolerance = CONFIG["psd_tolerance"]
    if (np.abs(values) > 1 + tolerance).any():
        raise error("A correlation entry lies outside [-1, 1].")
    matrices = unpack(values, CONFIG["n"])
    if (np.linalg.eigvalsh(matrices)[:, 0] < -tolerance).any():
        raise error("Each submitted matrix must be positive semidefinite.")
    return matrices


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """Return macro conditional structural sliced W2; lower is better.

    Rows within each condition are compared as distributions. The platform calls
    this separately on its Public and Private subsets; each must contain all
    conditions and at least CONFIG['minimum_per_class'] rows per condition.
    """
    if CONFIG is None:
        raise ValueError("Organizer must embed the training-fitted CONFIG.")
    names = columns(CONFIG["n"])
    expected = {row_id_column_name, *names}
    if not submission.columns.is_unique or set(submission.columns) != expected:
        raise ParticipantVisibleError("Submission columns do not match the template.")
    if not solution.columns.is_unique:
        raise ValueError("Duplicate solution columns")
    allowed_solution = expected | {"condition", "Usage"}
    if not expected | {"condition"} <= set(solution.columns):
        raise ValueError("Solution fields are incomplete")
    if not set(solution.columns) <= allowed_solution:
        raise ValueError("Unexpected solution fields")
    if solution[row_id_column_name].isna().any():
        raise ValueError("Null solution ID")
    if not solution[row_id_column_name].is_unique:
        raise ValueError("Duplicate solution ID")
    if submission[row_id_column_name].isna().any():
        raise ParticipantVisibleError("Submission IDs must be non-null.")
    if not submission[row_id_column_name].is_unique:
        raise ParticipantVisibleError("Submission IDs must be unique.")
    if set(solution[row_id_column_name]) != set(submission[row_id_column_name]):
        raise ParticipantVisibleError(
            "Submission IDs do not match the requested cases."
        )
    aligned = submission.set_index(row_id_column_name).loc[solution[row_id_column_name]]
    labels = solution["condition"].to_numpy()
    if set(labels) != set(CONFIG["classes"]):
        raise ValueError("Every scoring subset must contain all declared classes")
    if any(
        (labels == c).sum() < CONFIG["minimum_per_class"] for c in CONFIG["classes"]
    ):
        raise ValueError("Scoring subset has too few cases per class")
    real = features(_matrices(solution, names, False))
    generated = features(_matrices(aligned, names, True))
    scale = np.asarray(CONFIG["feature_scale"], dtype=np.float64)
    if scale.shape != (21,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid organizer feature normalization")
    rng = np.random.default_rng(CONFIG["projection_seed"])
    direction = rng.standard_normal((21, CONFIG["projections"]))
    direction /= np.linalg.norm(direction, axis=0, keepdims=True)
    real = real / scale @ direction
    generated = generated / scale @ direction
    scores = []
    for label in CONFIG["classes"]:
        mask = labels == label
        delta = np.sort(real[mask], axis=0) - np.sort(generated[mask], axis=0)
        scores.append(np.sqrt(np.mean(delta**2, axis=0)).mean())
    result = float(np.mean(scores))
    if not np.isfinite(result):
        raise ParticipantVisibleError("Submission produced a non-finite score.")
    return result
