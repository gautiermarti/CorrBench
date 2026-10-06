import numpy as np
import pandas as pd

from corrbench.jpx import curate_jpx_frames


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SecuritiesCode": ["1000", "2000"],
            "Name": ["Domestic", "ETF"],
            "Section/Products": ["First Section (Domestic)", "ETFs/ ETNs"],
            "33SectorCode": [10, "-"],
            "33SectorName": ["Sector", "-"],
            "17SectorCode": [1, "-"],
            "17SectorName": ["Group", "-"],
            "Universe0": [True, False],
        }
    )


def test_jpx_curation_adjusts_splits_and_filters_non_stocks() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "SecuritiesCode": ["1000"] * 4 + ["2000"] * 4,
            "Close": [100.0, 51.0, 52.0, 53.0, 10.0, 10.0, 10.0, 10.0],
            "Volume": [10.0] * 8,
            "AdjustmentFactor": [0.5, 1.0, 1.0, 1.0] + [1.0] * 4,
            "SupervisionFlag": [False] * 8,
            "source_file": ["synthetic.csv"] * 8,
        }
    )
    panel, metadata, report = curate_jpx_frames(
        prices,
        _master(),
        liquidity_window=2,
        liquidity_min_periods=1,
    )

    assert set(panel["instrument_id"]) == {"1000"}
    assert set(metadata["instrument_id"]) == {"1000"}
    np.testing.assert_allclose(panel["split_adjusted_close"], [50, 51, 52, 53])
    np.testing.assert_allclose(
        panel["daily_split_adjusted_return"].iloc[1:].to_numpy(),
        [0.02, 52 / 51 - 1, 53 / 52 - 1],
    )
    assert report["adjustment_events"] == 1
    assert report["return_definition"].startswith("split-adjusted")


def test_jpx_curation_does_not_bridge_missing_market_sessions() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame(
        {
            "Date": [dates[0], dates[2], *dates],
            "SecuritiesCode": ["1000", "1000", "2000", "2000", "2000"],
            "Close": [100.0, 110.0, 10.0, 10.0, 10.0],
            "Volume": [10.0] * 5,
            "AdjustmentFactor": [1.0] * 5,
            "SupervisionFlag": [False] * 5,
            "source_file": ["synthetic.csv"] * 5,
        }
    )
    master = _master()
    master.loc[master["SecuritiesCode"].eq("2000"), "Section/Products"] = (
        "Second Section (Domestic)"
    )
    panel, _, _ = curate_jpx_frames(
        prices,
        master,
        liquidity_window=2,
        liquidity_min_periods=1,
    )
    domestic = panel[panel["instrument_id"].eq("1000")]
    assert domestic["daily_split_adjusted_return"].isna().all()


def test_jpx_curation_excludes_marketwide_zero_trade_date() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame(
        {
            "Date": dates,
            "SecuritiesCode": ["1000"] * 3,
            "Close": [100.0, np.nan, 101.0],
            "Volume": [10.0, np.nan, 10.0],
            "AdjustmentFactor": [1.0] * 3,
            "SupervisionFlag": [False] * 3,
            "source_file": ["synthetic.csv"] * 3,
        }
    )
    panel, _, report = curate_jpx_frames(
        prices,
        _master().iloc[[0]],
        liquidity_window=2,
        liquidity_min_periods=1,
    )
    assert panel["date"].tolist() == [dates[0], dates[2]]
    np.testing.assert_allclose(panel["daily_split_adjusted_return"].iloc[1], 0.01)
    assert report["excluded_marketwide_no_trade_dates"] == ["2020-01-02"]
