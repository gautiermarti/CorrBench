import numpy as np
import pytest

from corrbench.weather import (
    annual_design,
    correlation_targets,
    patch_indices,
    remove_training_seasonality,
    spatial_kernel,
)


def test_later_year_values_cannot_change_fitted_climatology():
    dates = np.arange(np.datetime64("2009-01-01"), np.datetime64("2012-01-01"))
    rng = np.random.default_rng(4)
    design = annual_design(dates)
    values = design @ rng.normal(size=(9, 5)) + rng.normal(size=(len(dates), 5))
    _, original = remove_training_seasonality(values, dates)
    values[dates > np.datetime64("2010-12-31")] += 10000
    _, perturbed = remove_training_seasonality(values, dates)
    np.testing.assert_array_equal(original, perturbed)


def test_rank_deficient_weather_target_records_pd_regularization():
    rng = np.random.default_rng(7)
    observations = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 64))
    empirical, regularized, _, alpha = correlation_targets(observations)
    assert np.linalg.matrix_rank(empirical, tol=1e-8) == 5
    assert 0 < alpha < 1
    assert np.linalg.eigvalsh(regularized)[0] > 0
    np.testing.assert_allclose(np.diag(regularized), 1)


def test_dateline_wrap_and_spatial_kernel_relabeling():
    lat, lon = np.arange(90, -91, -2.5), np.arange(0, 360, 2.5)
    rows, cols = patch_indices(lat, lon, 45, 0)
    assert len(rows) == len(cols) == 8
    assert cols.min() == 0 and cols.max() == len(lon) - 1
    coordinates = np.array([[45, -3], [45, 0], [46, 0], [43, 1]])
    permutation = np.array([3, 1, 0, 2])
    a = spatial_kernel(coordinates, 500, 0.05)
    b = spatial_kernel(coordinates[permutation], 500, 0.05)
    np.testing.assert_allclose(b, a[permutation][:, permutation])
    assert np.linalg.eigvalsh(a)[0] > 0
    with pytest.raises(ValueError):
        spatial_kernel(coordinates, -1, 0.05)
