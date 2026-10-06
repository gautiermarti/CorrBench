#!/usr/bin/env python3
"""Build chronological correlation-matrix splits from a configured price panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from corrbench.datasets import (
    CorrelationWindowSampler,
    WindowSamplingConfig,
    make_wide_panel,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="Override the dated return-panel path."
    )
    parser.add_argument(
        "--samples-per-date",
        type=int,
        help="Override every split size for a quick smoke run.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new dataset directory: {args.output}")

    raw_config = load_config(args.config)
    source = args.source if args.source is not None else Path(raw_config["source"])
    columns = raw_config["columns"]
    selected_columns = list(dict.fromkeys(columns.values()))
    panel = pd.read_parquet(source, columns=selected_columns)
    returns, liquidity = make_wide_panel(
        panel,
        date_col=columns["date"],
        asset_col=columns["asset"],
        return_col=columns["return"],
        liquidity_col=columns.get("liquidity"),
    )
    sampling_config = WindowSamplingConfig(**raw_config["sampling"])
    sampler = CorrelationWindowSampler(returns, liquidity, sampling_config)

    args.output.mkdir(parents=True, exist_ok=False)
    split_reports = {}
    split_arrays = {}
    for split_name, split_config in raw_config["splits"].items():
        count = args.samples_per_date or int(split_config["samples_per_date"])
        arrays, report = sampler.sample_by_date(
            start_date=split_config["start"],
            end_date=split_config["end"],
            samples_per_date=count,
            seed=int(split_config["seed"]),
        )
        split_arrays[split_name] = arrays
        split_reports[split_name] = report

    lower_quantile, upper_quantile = raw_config["regime_quantiles"]
    lower_threshold, upper_threshold = np.quantile(
        split_arrays["train"]["sharpes"], [lower_quantile, upper_quantile]
    )
    for split_name, arrays in split_arrays.items():
        arrays["labels"] = np.where(
            arrays["sharpes"] < lower_threshold,
            0,
            np.where(arrays["sharpes"] > upper_threshold, 2, 1),
        ).astype(np.uint8)
        output_path = args.output / f"{split_name}.npz"
        np.savez_compressed(output_path, **arrays)
        report = split_reports[split_name]
        report["label_counts"] = {
            str(label): int(np.sum(arrays["labels"] == label)) for label in range(3)
        }
        report["sha256"] = sha256(output_path)
        print(f"{split_name}: {json.dumps(report, sort_keys=True)}", flush=True)

    manifest = {
        "market": raw_config["market"],
        "source": str(source),
        "source_sha256": sha256(source),
        "source_rows": int(panel.shape[0]),
        "source_assets": int(returns.shape[1]),
        "source_sessions": int(returns.shape[0]),
        "sampling": asdict(sampling_config),
        "regime_definition": {
            "conditioning": "equal-weight trailing-window annualized Sharpe",
            "training_quantiles": [lower_quantile, upper_quantile],
            "frozen_thresholds": [float(lower_threshold), float(upper_threshold)],
            "original_corrgan_thresholds": [
                sampling_config.stress_threshold,
                sampling_config.rally_threshold,
            ],
        },
        "splits": split_reports,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
