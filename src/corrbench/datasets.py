"""Reproducible construction of correlation-matrix datasets from price panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class WindowSamplingConfig:
    """Parameters matching the original CorrGAN sampling design by default."""

    matrix_dim: int = 80
    lookback: int = 252
    liquid_universe_size: int | None = 500
    annualization: int = 252
    stress_threshold: float = -0.5
    rally_threshold: float = 2.0


def sharpe_regime(
    sharpe: float,
    *,
    stress_threshold: float = -0.5,
    rally_threshold: float = 2.0,
) -> int:
    """Map equal-weight annualized Sharpe to the original three regimes."""
    if sharpe < stress_threshold:
        return 0
    if sharpe > rally_threshold:
        return 2
    return 1


def make_wide_panel(
    panel: pd.DataFrame,
    *,
    date_col: str,
    asset_col: str,
    return_col: str,
    liquidity_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convert a unique long return panel to aligned wide return/liquidity panels."""
    columns = [date_col, asset_col, return_col]
    if liquidity_col is not None:
        columns.append(liquidity_col)
    frame = panel.loc[:, columns].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])

    duplicates = frame.duplicated([date_col, asset_col])
    if duplicates.any():
        raise ValueError(
            f"Panel contains {int(duplicates.sum())} duplicate date/asset rows."
        )

    returns = frame.pivot(index=date_col, columns=asset_col, values=return_col)
    returns = returns.sort_index().sort_index(axis=1)
    liquidity = None
    if liquidity_col is not None:
        liquidity = frame.pivot(
            index=date_col, columns=asset_col, values=liquidity_col
        ).reindex(index=returns.index, columns=returns.columns)
    return returns, liquidity


class CorrelationWindowSampler:
    """Sample complete asset windows without using future universe information."""

    def __init__(
        self,
        returns: pd.DataFrame,
        liquidity: pd.DataFrame | None,
        config: WindowSamplingConfig,
    ) -> None:
        if not returns.index.is_monotonic_increasing:
            raise ValueError("Return dates must be sorted.")
        if returns.columns.has_duplicates or returns.index.has_duplicates:
            raise ValueError("Return panel axes must be unique.")
        if liquidity is not None and not (
            returns.index.equals(liquidity.index)
            and returns.columns.equals(liquidity.columns)
        ):
            raise ValueError("Liquidity and return panels must have identical axes.")
        if config.lookback <= config.matrix_dim:
            raise ValueError("lookback must exceed matrix_dim for full-rank estimates.")

        self.returns = returns.to_numpy(dtype=np.float64, na_value=np.nan)
        self.liquidity = (
            None
            if liquidity is None
            else liquidity.to_numpy(dtype=np.float64, na_value=np.nan)
        )
        self.dates = returns.index.to_numpy(dtype="datetime64[ns]")
        self.assets = returns.columns.to_numpy(dtype=str)
        self.config = config

        observed = np.isfinite(self.returns).astype(np.int32)
        self._observed_cumsum = np.vstack(
            [np.zeros((1, observed.shape[1]), dtype=np.int32), observed.cumsum(axis=0)]
        )

    def eligible_end_indices(
        self, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp
    ) -> NDArray[np.int64]:
        """Return session indices whose lookback ends inside the requested period."""
        start = np.datetime64(pd.Timestamp(start_date), "ns")
        end = np.datetime64(pd.Timestamp(end_date), "ns")
        indices = np.flatnonzero((self.dates >= start) & (self.dates <= end))
        return indices[indices >= self.config.lookback - 1].astype(np.int64)

    def _eligible_assets(self, end_index: int) -> NDArray[np.int64]:
        start = end_index - self.config.lookback + 1
        counts = self._observed_cumsum[end_index + 1] - self._observed_cumsum[start]
        eligible = np.flatnonzero(counts == self.config.lookback)

        universe_size = self.config.liquid_universe_size
        if self.liquidity is not None and universe_size is not None:
            end_liquidity = self.liquidity[end_index, eligible]
            finite = np.isfinite(end_liquidity)
            eligible = eligible[finite]
            end_liquidity = end_liquidity[finite]
            if eligible.size > universe_size:
                keep = np.argpartition(end_liquidity, -universe_size)[-universe_size:]
                eligible = eligible[keep]
        return eligible.astype(np.int64)

    def sample_one(
        self, end_index: int, rng: np.random.Generator
    ) -> dict[str, Any] | None:
        """Sample one matrix and its auditable provenance at a fixed window end."""
        eligible = self._eligible_assets(end_index)
        if eligible.size < self.config.matrix_dim:
            return None
        # Keep the RNG order: identifier sorting can encode exchange or sector
        # conventions. Image baselines receive explicit Ward ordering separately.
        selected = rng.choice(eligible, size=self.config.matrix_dim, replace=False)
        start = end_index - self.config.lookback + 1
        window = self.returns[start : end_index + 1, selected]
        corr = np.corrcoef(window, rowvar=False)
        if not np.isfinite(corr).all():
            return None

        portfolio_returns = window.mean(axis=1)
        volatility = portfolio_returns.std(ddof=1)
        if not np.isfinite(volatility) or volatility <= 0:
            return None
        sharpe = (
            portfolio_returns.mean()
            * self.config.annualization
            / (volatility * np.sqrt(self.config.annualization))
        )
        label = sharpe_regime(
            float(sharpe),
            stress_threshold=self.config.stress_threshold,
            rally_threshold=self.config.rally_threshold,
        )
        minimum_eigenvalue = float(np.linalg.eigvalsh(corr)[0])
        return {
            "matrix": corr.astype(np.float32),
            "label": label,
            "sharpe": float(sharpe),
            "end_date": self.dates[end_index],
            "assets": self.assets[selected],
            "minimum_eigenvalue": minimum_eigenvalue,
        }

    def sample_nested_one(
        self,
        end_index: int,
        rng: np.random.Generator,
        matrix_dims: tuple[int, ...],
    ) -> dict[int, dict[str, Any]] | None:
        """Draw one maximum universe and return its nested principal matrices."""
        dims = tuple(sorted(set(matrix_dims)))
        if not dims or dims[0] < 2:
            raise ValueError("matrix_dims must contain dimensions of at least two.")
        if dims[-1] >= self.config.lookback:
            raise ValueError("Every matrix dimension must be below the lookback.")

        eligible = self._eligible_assets(end_index)
        if eligible.size < dims[-1]:
            return None
        selected = rng.choice(eligible, size=dims[-1], replace=False)
        start = end_index - self.config.lookback + 1
        window = self.returns[start : end_index + 1, selected]
        corr = np.corrcoef(window, rowvar=False)
        if not np.isfinite(corr).all():
            return None

        shared_returns = window.mean(axis=1)
        shared_volatility = shared_returns.std(ddof=1)
        if not np.isfinite(shared_volatility) or shared_volatility <= 0:
            return None
        shared_sharpe = (
            shared_returns.mean()
            * self.config.annualization
            / (shared_volatility * np.sqrt(self.config.annualization))
        )
        label = sharpe_regime(
            float(shared_sharpe),
            stress_threshold=self.config.stress_threshold,
            rally_threshold=self.config.rally_threshold,
        )
        parent_eigenvalues = np.linalg.eigvalsh(corr)
        parent_minimum_eigenvalue = float(parent_eigenvalues[0])
        parent_maximum_eigenvalue = float(parent_eigenvalues[-1])
        parent_condition_number = parent_maximum_eigenvalue / parent_minimum_eigenvalue

        nested = {}
        for matrix_dim in dims:
            local_returns = window[:, :matrix_dim].mean(axis=1)
            local_volatility = local_returns.std(ddof=1)
            local_sharpe = (
                local_returns.mean()
                * self.config.annualization
                / (local_volatility * np.sqrt(self.config.annualization))
            )
            nested[matrix_dim] = {
                "matrix": corr[:matrix_dim, :matrix_dim].astype(np.float32),
                "label": label,
                "sharpe": float(shared_sharpe),
                "local_sharpe": float(local_sharpe),
                "end_date": self.dates[end_index],
                "assets": self.assets[selected[:matrix_dim]],
                "parent_minimum_eigenvalue": parent_minimum_eigenvalue,
                "parent_maximum_eigenvalue": parent_maximum_eigenvalue,
                "parent_condition_number": parent_condition_number,
            }
        return nested

    def sample_nested_by_date(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        samples_per_date: int,
        seed: int,
        matrix_dims: tuple[int, ...],
    ) -> tuple[dict[int, dict[str, NDArray[Any]]], dict[str, Any]]:
        """Build paired, nested matrices at several dimensions for every date."""
        if samples_per_date < 1:
            raise ValueError("samples_per_date must be positive.")
        dims = tuple(sorted(set(matrix_dims)))
        end_indices = self.eligible_end_indices(start_date, end_date)
        if end_indices.size == 0:
            raise ValueError("No eligible window ends in the requested date range.")

        rng = np.random.default_rng(seed)
        buckets: dict[int, list[dict[str, Any]]] = {dim: [] for dim in dims}
        skipped_dates = []
        for end_index in end_indices:
            date_samples = []
            for _ in range(samples_per_date):
                sample = self.sample_nested_one(int(end_index), rng, dims)
                if sample is not None:
                    date_samples.append(sample)
            if len(date_samples) != samples_per_date:
                skipped_dates.append(self.dates[end_index])
                continue
            for sample in date_samples:
                for dim in dims:
                    buckets[dim].append(sample[dim])

        if not buckets[dims[-1]]:
            raise RuntimeError("No complete nested windows could be sampled.")

        arrays_by_dim = {}
        for dim, samples in buckets.items():
            arrays_by_dim[dim] = {
                "matrices": np.stack([item["matrix"] for item in samples]),
                "original_labels": np.asarray(
                    [item["label"] for item in samples], dtype=np.uint8
                ),
                "sharpes": np.asarray(
                    [item["sharpe"] for item in samples], dtype=np.float32
                ),
                "local_sharpes": np.asarray(
                    [item["local_sharpe"] for item in samples], dtype=np.float32
                ),
                "end_dates": np.asarray(
                    [item["end_date"] for item in samples], dtype="datetime64[ns]"
                ),
                "assets": np.stack([item["assets"] for item in samples]),
                "parent_minimum_eigenvalues": np.asarray(
                    [item["parent_minimum_eigenvalue"] for item in samples],
                    dtype=np.float32,
                ),
                "parent_maximum_eigenvalues": np.asarray(
                    [item["parent_maximum_eigenvalue"] for item in samples],
                    dtype=np.float32,
                ),
                "parent_condition_numbers": np.asarray(
                    [item["parent_condition_number"] for item in samples],
                    dtype=np.float32,
                ),
            }
        largest = arrays_by_dim[dims[-1]]
        report = {
            "samples": int(largest["matrices"].shape[0]),
            "samples_per_date": samples_per_date,
            "matrix_dims": list(dims),
            "eligible_end_dates": int(end_indices.size),
            "unique_end_dates": int(np.unique(largest["end_dates"]).size),
            "skipped_end_dates": len(skipped_dates),
            "skipped_end_date_values": [
                str(date.astype("datetime64[D]")) for date in skipped_dates
            ],
            "date_min": str(largest["end_dates"].min().astype("datetime64[D]")),
            "date_max": str(largest["end_dates"].max().astype("datetime64[D]")),
            "parent_minimum_eigenvalue": float(
                largest["parent_minimum_eigenvalues"].min()
            ),
            "parent_condition_number_median": float(
                np.median(largest["parent_condition_numbers"])
            ),
            "parent_condition_number_maximum": float(
                largest["parent_condition_numbers"].max()
            ),
        }
        return arrays_by_dim, report

    def sample_balanced(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        samples_per_class: int,
        seed: int,
        max_attempt_multiplier: int = 200,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Rejection-sample equal regime counts and return coverage diagnostics."""
        end_indices = self.eligible_end_indices(start_date, end_date)
        if end_indices.size == 0:
            raise ValueError("No eligible window ends in the requested date range.")

        rng = np.random.default_rng(seed)
        buckets: list[list[dict[str, Any]]] = [[], [], []]
        attempts = 0
        max_attempts = samples_per_class * 3 * max_attempt_multiplier
        while min(map(len, buckets)) < samples_per_class and attempts < max_attempts:
            end_index = int(rng.choice(end_indices))
            sample = self.sample_one(end_index, rng)
            attempts += 1
            if sample is None:
                continue
            label = sample["label"]
            if len(buckets[label]) < samples_per_class:
                buckets[label].append(sample)

        counts = [len(bucket) for bucket in buckets]
        if min(counts) < samples_per_class:
            raise RuntimeError(
                "Unable to fill all regimes after "
                f"{attempts} attempts; collected class counts {counts}."
            )

        samples = [item for bucket in buckets for item in bucket]
        order = rng.permutation(len(samples))
        samples = [samples[index] for index in order]
        arrays = {
            "matrices": np.stack([item["matrix"] for item in samples]),
            "labels": np.asarray([item["label"] for item in samples], dtype=np.uint8),
            "sharpes": np.asarray(
                [item["sharpe"] for item in samples], dtype=np.float32
            ),
            "end_dates": np.asarray(
                [item["end_date"] for item in samples], dtype="datetime64[ns]"
            ),
            "assets": np.stack([item["assets"] for item in samples]),
            "minimum_eigenvalues": np.asarray(
                [item["minimum_eigenvalue"] for item in samples], dtype=np.float32
            ),
        }
        coverage = {
            "attempts": attempts,
            "accepted": len(samples),
            "acceptance_rate": len(samples) / attempts,
            "unique_end_dates": int(np.unique(arrays["end_dates"]).size),
            "unique_end_dates_by_class": {
                str(label): int(
                    np.unique(arrays["end_dates"][arrays["labels"] == label]).size
                )
                for label in range(3)
            },
            "date_min": str(arrays["end_dates"].min().astype("datetime64[D]")),
            "date_max": str(arrays["end_dates"].max().astype("datetime64[D]")),
            "minimum_eigenvalue": float(arrays["minimum_eigenvalues"].min()),
        }
        return arrays, coverage

    def sample_by_date(
        self,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        samples_per_date: int,
        seed: int,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Sample a fixed number per session, preserving natural time coverage."""
        if samples_per_date < 1:
            raise ValueError("samples_per_date must be positive.")
        end_indices = self.eligible_end_indices(start_date, end_date)
        if end_indices.size == 0:
            raise ValueError("No eligible window ends in the requested date range.")

        rng = np.random.default_rng(seed)
        samples = []
        skipped_dates = []
        for end_index in end_indices:
            date_samples = []
            for _ in range(samples_per_date):
                sample = self.sample_one(int(end_index), rng)
                if sample is not None:
                    date_samples.append(sample)
            if len(date_samples) != samples_per_date:
                skipped_dates.append(self.dates[end_index])
                continue
            samples.extend(date_samples)

        if not samples:
            raise RuntimeError("No complete windows could be sampled.")
        arrays = {
            "matrices": np.stack([item["matrix"] for item in samples]),
            "original_labels": np.asarray(
                [item["label"] for item in samples], dtype=np.uint8
            ),
            "sharpes": np.asarray(
                [item["sharpe"] for item in samples], dtype=np.float32
            ),
            "end_dates": np.asarray(
                [item["end_date"] for item in samples], dtype="datetime64[ns]"
            ),
            "assets": np.stack([item["assets"] for item in samples]),
            "minimum_eigenvalues": np.asarray(
                [item["minimum_eigenvalue"] for item in samples], dtype=np.float32
            ),
        }
        report = {
            "samples": len(samples),
            "samples_per_date": samples_per_date,
            "eligible_end_dates": int(end_indices.size),
            "unique_end_dates": int(np.unique(arrays["end_dates"]).size),
            "skipped_end_dates": len(skipped_dates),
            "skipped_end_date_values": [
                str(date.astype("datetime64[D]")) for date in skipped_dates
            ],
            "original_label_counts": {
                str(label): int(np.sum(arrays["original_labels"] == label))
                for label in range(3)
            },
            "date_min": str(arrays["end_dates"].min().astype("datetime64[D]")),
            "date_max": str(arrays["end_dates"].max().astype("datetime64[D]")),
            "minimum_eigenvalue": float(arrays["minimum_eigenvalues"].min()),
        }
        return arrays, report
