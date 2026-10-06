"""Subject/site-aware construction of the declared ABIDE PCP correlation pilot."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import OAS

ABIDE_SOURCE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
ABIDE_DERIVATIVE = "Outputs/cpac/filt_noglobal/rois_ho"
ABIDE_PHENOTYPE_SHA256 = (
    "009f01a89697113e5375ded4fbcff4c85e89a4daa6d13e63c21ae5d57dda0865"
)


def eligible_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    """Prespecified PCP/Nilearn quality criteria plus mean displacement < .2."""
    if frame.SUB_ID.isna().any() or frame.SUB_ID.duplicated().any():
        raise ValueError("Require unique, nonmissing public subject IDs")
    mask = (
        frame.FILE_ID.notna()
        & frame.FILE_ID.ne("no_filename")
        & frame.qc_rater_1.eq("OK")
        & frame.qc_anat_rater_2.isin(["OK", "maybe"])
        & frame.qc_func_rater_2.isin(["OK", "maybe"])
        & frame.qc_anat_rater_3.eq("OK")
        & frame.qc_func_rater_3.eq("OK")
        & frame.func_mean_fd.ge(0)
        & frame.func_mean_fd.lt(0.2)
        & frame.DX_GROUP.isin([1, 2])
    )
    result = frame.loc[mask].sort_values("SUB_ID").copy()
    if not result.FILE_ID.map(
        lambda x: bool(re.fullmatch(r"[A-Za-z0-9_]+", str(x)))
    ).all():
        raise ValueError("Unexpected source filename")
    result["split"] = result.SITE_ID.map(subject_split)
    result["label"] = result.DX_GROUP.eq(1).astype(np.int64)
    return result


def subject_split(site: str) -> str:
    if site == "NYU":
        return "test"
    if site in {"UM_1", "UM_2"}:
        return "validation"
    if not isinstance(site, str) or not site:
        raise ValueError("Nonempty site ID required")
    return "train"


def read_roi_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text()
    first = text.splitlines()[0].split()
    if not first or not all(re.fullmatch(r"#\d+", x) for x in first):
        raise ValueError("Missing or malformed atlas ROI header")
    node_ids = np.asarray([int(x[1:]) for x in first], dtype=np.int64)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Duplicate atlas ROI IDs")
    values = np.loadtxt(io.StringIO(text), comments="#", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(node_ids):
        raise ValueError("ROI header and series dimensions differ")
    if values.shape[0] < 50:
        raise ValueError("Fewer than 50 time points")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ROI observations")
    if np.any(values.std(axis=0, ddof=1) <= 1e-12):
        raise ValueError("Constant ROI series")
    return values, node_ids


def scan_correlations(values: np.ndarray) -> dict:
    """Pearson diagnostic and per-scan OAS-regularized PD correlation target."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite observations-by-nodes array required")
    scale = x.std(axis=0, ddof=1)
    if np.any(scale <= 1e-12):
        raise ValueError("Constant ROI series")
    z = (x - x.mean(axis=0)) / scale
    empirical = z.T @ z / (len(z) - 1)
    np.fill_diagonal(empirical, 1.0)
    estimate = OAS(assume_centered=True, store_precision=False).fit(z)
    covariance = estimate.covariance_
    diagonal = np.sqrt(np.diag(covariance))
    correlation = covariance / diagonal[:, None] / diagonal[None, :]
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    minimum = float(np.linalg.eigvalsh(correlation)[0])
    if minimum <= 0:
        raise ValueError("OAS target is not positive definite")
    return {
        "matrices": correlation,
        "empirical_matrices": empirical,
        "shrinkage": float(estimate.shrinkage_),
        "minimum_empirical_eigenvalue": float(np.linalg.eigvalsh(empirical)[0]),
        "minimum_target_eigenvalue": minimum,
        "empirical_rank": int(np.linalg.matrix_rank(empirical)),
        "n_timepoints": len(x),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
