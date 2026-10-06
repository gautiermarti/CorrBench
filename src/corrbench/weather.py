"""Explicit seasonal preprocessing and spatial-correlation construction helpers."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import OAS

REGIONS = (
    ("europe", 45.0, 0.0),
    ("north_america", 40.0, -105.0),
    ("east_asia", 35.0, 120.0),
    ("south_america_transfer", -25.0, -60.0),
)


def annual_design(dates, harmonics=4):
    dates = np.asarray(dates, dtype="datetime64[D]")
    years = dates.astype("datetime64[Y]")
    start = years.astype("datetime64[D]")
    end = (years + np.timedelta64(1, "Y")).astype("datetime64[D]")
    phase = np.asarray((dates - start) / (end - start), dtype=float)
    columns = [np.ones(len(dates))]
    for k in range(1, harmonics + 1):
        columns += [np.sin(2 * np.pi * k * phase), np.cos(2 * np.pi * k * phase)]
    return np.column_stack(columns)


def remove_training_seasonality(values, dates, training_end="2010-12-31"):
    values = np.asarray(values, dtype=np.float64)
    dates = np.asarray(dates, dtype="datetime64[D]")
    if values.ndim != 2 or len(values) != len(dates) or not np.isfinite(values).all():
        raise ValueError("Require finite day-by-location observations")
    design = annual_design(dates)
    fitting = dates <= np.datetime64(training_end)
    if fitting.sum() < 365:
        raise ValueError("At least one year of training climatology is required")
    coefficients = np.linalg.lstsq(design[fitting], values[fitting], rcond=None)[0]
    return values - design @ coefficients, coefficients


def patch_indices(latitudes, longitudes, center_lat, center_lon, width=8):
    if width < 2 or width % 2:
        raise ValueError("Require even patch width >= 2")
    latitudes, longitudes = np.asarray(latitudes), np.asarray(longitudes)
    latitude_index = int(np.argmin(np.abs(latitudes - center_lat)))
    longitude_index = int(
        np.argmin(np.abs((longitudes - center_lon + 180) % 360 - 180))
    )
    offsets = np.arange(-(width // 2) + 1, width // 2 + 1)
    rows = latitude_index + offsets
    columns = (longitude_index + offsets) % len(longitudes)
    if rows.min() < 0 or rows.max() >= len(latitudes):
        raise ValueError("Patch crosses a pole")
    if len(np.unique(columns)) != width:
        raise ValueError("Repeated longitude cell")
    return rows, columns


def correlation_targets(anomalies):
    anomalies = np.asarray(anomalies, dtype=np.float64)
    if anomalies.ndim != 2 or not np.isfinite(anomalies).all():
        raise ValueError("Require finite day-by-location anomalies")
    standard_deviation = anomalies.std(axis=0, ddof=1)
    if len(anomalies) < 50 or np.any(standard_deviation <= 0):
        raise ValueError("Insufficient observations or constant cells")
    z = (anomalies - anomalies.mean(axis=0)) / standard_deviation
    empirical = z.T @ z / (len(z) - 1)
    estimated = OAS(store_precision=False).fit(z)
    covariance = estimated.covariance_
    scale = np.sqrt(np.diag(covariance))
    regularized = covariance / scale[:, None] / scale[None, :]
    np.fill_diagonal(empirical, 1)
    np.fill_diagonal(regularized, 1)
    if np.linalg.eigvalsh(regularized).min() <= 0:
        raise ValueError("Non-PD regularized matrix")
    return empirical, regularized, standard_deviation, float(estimated.shrinkage_)


def chord_distances(coordinates):
    lat, lon = np.deg2rad(np.asarray(coordinates, dtype=float)).T
    xyz = 6371 * np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)


def spatial_kernel(coordinates, length_km, nugget):
    if length_km <= 0 or not 0 < nugget <= 1:
        raise ValueError("Require positive length and nugget in (0,1]")
    matrix = (1 - nugget) * np.exp(-chord_distances(coordinates) / length_km)
    np.fill_diagonal(matrix, 1)
    return matrix


def aggregate_weights(coordinates):
    """Area-weighted regional mean and four quadrant means on the 8x8 patch."""
    coordinates = np.asarray(coordinates)
    if coordinates.shape != (64, 2):
        raise ValueError("Expected the declared 8x8 patch")
    area = np.cos(np.deg2rad(coordinates[:, 0]))
    weights = [area / area.sum()]
    grid = np.arange(64).reshape(8, 8)
    for rows, cols in [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 8)),
        (slice(4, 8), slice(0, 4)),
        (slice(4, 8), slice(4, 8)),
    ]:
        chosen = grid[rows, cols].ravel()
        w = np.zeros(64)
        w[chosen] = area[chosen] / area[chosen].sum()
        weights.append(w)
    return np.stack(weights)
