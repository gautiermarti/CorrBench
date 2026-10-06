import numpy as np
import pandas as pd

from corrbench.datasets import (
    CorrelationWindowSampler,
    WindowSamplingConfig,
    make_wide_panel,
    sharpe_regime,
)


def test_sharpe_regime_uses_original_thresholds() -> None:
    assert sharpe_regime(-0.51) == 0
    assert sharpe_regime(-0.5) == 1
    assert sharpe_regime(2.0) == 1
    assert sharpe_regime(2.01) == 2


def test_sampler_is_deterministic_and_produces_valid_matrix() -> None:
    rng = np.random.default_rng(4)
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    common = rng.normal(size=(40, 1))
    values = 0.4 * common + rng.normal(size=(40, 8))
    long = pd.DataFrame(
        {
            "date": np.repeat(dates, 8),
            "asset": np.tile([f"A{i}" for i in range(8)], 40),
            "return": values.ravel(),
            "liquidity": np.tile(np.arange(8), 40),
        }
    )
    returns, liquidity = make_wide_panel(
        long,
        date_col="date",
        asset_col="asset",
        return_col="return",
        liquidity_col="liquidity",
    )
    sampler = CorrelationWindowSampler(
        returns,
        liquidity,
        WindowSamplingConfig(matrix_dim=4, lookback=20, liquid_universe_size=6),
    )
    first = sampler.sample_one(25, np.random.default_rng(10))
    second = sampler.sample_one(25, np.random.default_rng(10))

    assert first is not None and second is not None
    np.testing.assert_array_equal(first["assets"], second["assets"])
    np.testing.assert_allclose(first["matrix"], second["matrix"])
    np.testing.assert_allclose(np.diag(first["matrix"]), 1)
    assert first["minimum_eigenvalue"] > 0


def test_liquidity_filter_is_point_in_time() -> None:
    dates = pd.date_range("2021-01-01", periods=8, freq="B")
    assets = ["A", "B", "C", "D"]
    rows = []
    for date_index, date in enumerate(dates):
        for asset_index, asset in enumerate(assets):
            liquidity = asset_index
            if date_index == 7 and asset == "A":
                liquidity = 100
            rows.append((date, asset, date_index + asset_index / 10, liquidity))
    panel = pd.DataFrame(rows, columns=["date", "asset", "return", "liquidity"])
    returns, liquidity = make_wide_panel(
        panel,
        date_col="date",
        asset_col="asset",
        return_col="return",
        liquidity_col="liquidity",
    )
    sampler = CorrelationWindowSampler(
        returns,
        liquidity,
        WindowSamplingConfig(matrix_dim=2, lookback=4, liquid_universe_size=2),
    )
    assert set(sampler.assets[sampler._eligible_assets(6)]) == {"C", "D"}
    assert set(sampler.assets[sampler._eligible_assets(7)]) == {"A", "D"}


def test_nested_sampler_produces_exact_principal_submatrices() -> None:
    rng = np.random.default_rng(8)
    dates = pd.date_range("2020-01-01", periods=35, freq="B")
    values = rng.normal(size=(35, 12))
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 12),
            "asset": np.tile([f"A{i}" for i in range(12)], 35),
            "return": values.ravel(),
            "liquidity": 1.0,
        }
    )
    returns, liquidity = make_wide_panel(
        panel,
        date_col="date",
        asset_col="asset",
        return_col="return",
        liquidity_col="liquidity",
    )
    sampler = CorrelationWindowSampler(
        returns,
        liquidity,
        WindowSamplingConfig(matrix_dim=8, lookback=20, liquid_universe_size=12),
    )
    sample = sampler.sample_nested_one(25, np.random.default_rng(3), (2, 4, 8))
    assert sample is not None
    np.testing.assert_array_equal(sample[2]["assets"], sample[8]["assets"][:2])
    np.testing.assert_allclose(sample[2]["matrix"], sample[8]["matrix"][:2, :2])
    np.testing.assert_allclose(sample[4]["matrix"], sample[8]["matrix"][:4, :4])
    assert sample[2]["sharpe"] == sample[8]["sharpe"]
    assert sample[8]["parent_minimum_eigenvalue"] > 0
    assert sample[8]["parent_condition_number"] >= 1
