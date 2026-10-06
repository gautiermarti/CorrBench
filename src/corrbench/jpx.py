"""Curation utilities for the official JPX Kaggle equity archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

PRICE_FILES = (
    "train_files/stock_prices.csv",
    "train_files/secondary_stock_prices.csv",
    "supplemental_files/stock_prices.csv",
    "supplemental_files/secondary_stock_prices.csv",
)
PRICE_COLUMNS = (
    "Date",
    "SecuritiesCode",
    "Close",
    "Volume",
    "AdjustmentFactor",
    "SupervisionFlag",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jpx_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the non-overlapping equity files and security master from a JPX ZIP."""
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {*PRICE_FILES, "stock_list.csv"}
        missing = required - names
        if missing:
            raise ValueError(f"JPX archive is missing files: {sorted(missing)}")

        frames = []
        for source_file in PRICE_FILES:
            frame = pd.read_csv(
                archive.open(source_file),
                usecols=list(PRICE_COLUMNS),
                parse_dates=["Date"],
                dtype={
                    "SecuritiesCode": "string",
                    "Close": "float64",
                    "Volume": "float64",
                    "AdjustmentFactor": "float64",
                    "SupervisionFlag": "boolean",
                },
            )
            frame["source_file"] = source_file
            frames.append(frame)
        prices = pd.concat(frames, ignore_index=True)
        master = pd.read_csv(
            archive.open("stock_list.csv"),
            dtype={"SecuritiesCode": "string"},
        )
    return prices, master


def curate_jpx_frames(
    prices: pd.DataFrame,
    master: pd.DataFrame,
    *,
    liquidity_window: int = 60,
    liquidity_min_periods: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build split-adjusted returns and point-in-time trailing liquidity.

    The competition adjustment factor is attached to the last pre-action close.
    Reverse cumulative products therefore express every close on the final
    split-adjustment basis. Dividends are not included in the JPX factor.
    """
    if liquidity_window < 1:
        raise ValueError("liquidity_window must be positive.")
    if not 1 <= liquidity_min_periods <= liquidity_window:
        raise ValueError("liquidity_min_periods must be in [1, liquidity_window].")
    missing_price = set(PRICE_COLUMNS) - set(prices.columns)
    if missing_price:
        raise ValueError(f"Price data is missing columns: {sorted(missing_price)}")
    required_master = {
        "SecuritiesCode",
        "Name",
        "Section/Products",
        "33SectorCode",
        "33SectorName",
        "17SectorCode",
        "17SectorName",
        "Universe0",
    }
    missing_master = required_master - set(master.columns)
    if missing_master:
        raise ValueError(
            f"Security master is missing columns: {sorted(missing_master)}"
        )

    metadata = master.copy()
    metadata["SecuritiesCode"] = metadata["SecuritiesCode"].astype("string")
    if metadata["SecuritiesCode"].duplicated().any():
        raise ValueError("Security master contains duplicate securities codes.")
    domestic = metadata["Section/Products"].str.contains(
        "Domestic", case=False, na=False
    )
    metadata = metadata.loc[domestic].copy()
    metadata = metadata.rename(columns={"SecuritiesCode": "instrument_id"})
    metadata["instrument_id"] = metadata["instrument_id"].astype(str)

    frame = prices.copy()
    frame["SecuritiesCode"] = frame["SecuritiesCode"].astype("string")
    frame = frame[frame["SecuritiesCode"].isin(metadata["instrument_id"])].copy()
    if frame.duplicated(["Date", "SecuritiesCode"]).any():
        count = int(frame.duplicated(["Date", "SecuritiesCode"]).sum())
        raise ValueError(f"Price data contains {count} duplicate date/security rows.")
    if frame["Date"].isna().any():
        raise ValueError("Price data contains invalid dates.")
    if frame["AdjustmentFactor"].isna().any() or (frame["AdjustmentFactor"] <= 0).any():
        raise ValueError("Adjustment factors must be finite and positive.")

    close_counts = frame.groupby("Date", sort=True)["Close"].count()
    marketwide_no_trade_dates = close_counts.index[close_counts.eq(0)]
    frame = frame[~frame["Date"].isin(marketwide_no_trade_dates)].copy()

    frame = frame.rename(
        columns={
            "Date": "date",
            "SecuritiesCode": "instrument_id",
            "Close": "close",
            "Volume": "volume",
            "AdjustmentFactor": "adjustment_factor",
            "SupervisionFlag": "supervision_flag",
        }
    )
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame = frame.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    sessions = pd.Index(frame["date"].drop_duplicates().sort_values())
    session_lookup = pd.Series(np.arange(len(sessions), dtype=np.int32), index=sessions)
    frame["_session"] = frame["date"].map(session_lookup).astype(np.int32)

    reverse_adjustment = frame.groupby("instrument_id", sort=False)[
        "adjustment_factor"
    ].transform(lambda values: values.iloc[::-1].cumprod().iloc[::-1])
    frame["split_adjusted_close"] = frame["close"] * reverse_adjustment
    groups = frame.groupby("instrument_id", sort=False)
    previous_close = groups["split_adjusted_close"].shift()
    session_gap = groups["_session"].diff()
    returns = frame["split_adjusted_close"] / previous_close - 1.0
    frame["daily_split_adjusted_return"] = returns.where(session_gap.eq(1))

    frame["traded_value"] = frame["close"] * frame["volume"]
    frame["median_value_traded_60d"] = (
        frame.groupby("instrument_id", sort=False)["traded_value"]
        .rolling(window=liquidity_window, min_periods=liquidity_min_periods)
        .median()
        .reset_index(level=0, drop=True)
    )

    output_columns = [
        "date",
        "instrument_id",
        "daily_split_adjusted_return",
        "median_value_traded_60d",
        "close",
        "split_adjusted_close",
        "volume",
        "traded_value",
        "adjustment_factor",
        "supervision_flag",
        "source_file",
    ]
    panel = frame.loc[:, output_columns].copy()
    numeric_float32 = [
        "daily_split_adjusted_return",
        "median_value_traded_60d",
        "close",
        "split_adjusted_close",
        "volume",
        "traded_value",
        "adjustment_factor",
    ]
    panel[numeric_float32] = panel[numeric_float32].astype(np.float32)

    end_date = panel["date"].max()
    first_dates = panel.groupby("instrument_id", sort=False)["date"].min()
    last_dates = panel.groupby("instrument_id", sort=False)["date"].max()
    finite_returns = panel["daily_split_adjusted_return"].dropna()
    report: dict[str, Any] = {
        "rows": int(len(panel)),
        "assets": int(panel["instrument_id"].nunique()),
        "sessions": int(panel["date"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(end_date.date()),
        "domestic_filter": "Section/Products contains 'Domestic'",
        "assets_starting_after_first_session": int(
            (first_dates > panel["date"].min()).sum()
        ),
        "assets_ending_before_last_session": int((last_dates < end_date).sum()),
        "adjustment_events": int(panel["adjustment_factor"].ne(1).sum()),
        "missing_close_rows": int(panel["close"].isna().sum()),
        "excluded_marketwide_no_trade_dates": [
            str(date.date()) for date in marketwide_no_trade_dates
        ],
        "finite_return_rows": int(finite_returns.size),
        "absolute_return_over_50pct": int(finite_returns.abs().gt(0.5).sum()),
        "absolute_return_over_100pct": int(finite_returns.abs().gt(1.0).sum()),
        "return_min": float(finite_returns.min()),
        "return_max": float(finite_returns.max()),
        "liquidity_window": liquidity_window,
        "liquidity_min_periods": liquidity_min_periods,
        "return_definition": "split-adjusted close-to-close; dividends excluded",
    }
    return panel, metadata, report


def build_jpx_panel(
    archive_path: Path,
    panel_path: Path,
    metadata_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Curate the JPX archive and persist Parquet outputs plus provenance."""
    prices, master = read_jpx_archive(archive_path)
    raw_rows = int(len(prices))
    raw_assets = int(prices["SecuritiesCode"].nunique())
    panel, metadata, report = curate_jpx_frames(prices, master)

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    metadata.to_parquet(metadata_path, index=False)
    manifest = {
        "source": "JPX Tokyo Stock Exchange Prediction competition archive",
        "source_url": (
            "https://www.kaggle.com/competitions/"
            "jpx-tokyo-stock-exchange-prediction/data"
        ),
        "license": "Subject to competition rules; raw archive is not redistributed",
        "archive_path": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "included_price_files": list(PRICE_FILES),
        "excluded_overlapping_files": [
            "example_test_files/stock_prices.csv",
            "example_test_files/secondary_stock_prices.csv",
        ],
        "raw_rows": raw_rows,
        "raw_assets": raw_assets,
        "panel_path": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        **report,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
