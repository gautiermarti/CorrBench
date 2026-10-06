#!/usr/bin/env python3
"""Build paired nested correlation matrices at multiple dimensions."""

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
    parser.add_argument("--samples-per-date", type=int)
    args = parser.parse_args()

    raw_config = load_config(args.config)
    source = Path(raw_config["source"])
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

    matrix_dims = tuple(int(dim) for dim in raw_config["matrix_dims"])
    sampling_values = dict(raw_config["sampling"])
    sampling_values["matrix_dim"] = max(matrix_dims)
    sampling_config = WindowSamplingConfig(**sampling_values)
    sampler = CorrelationWindowSampler(returns, liquidity, sampling_config)

    split_arrays = {}
    split_reports = {}
    for split_name, split_config in raw_config["splits"].items():
        count = args.samples_per_date or int(split_config["samples_per_date"])
        arrays, report = sampler.sample_nested_by_date(
            start_date=split_config["start"],
            end_date=split_config["end"],
            samples_per_date=count,
            seed=int(split_config["seed"]),
            matrix_dims=matrix_dims,
        )
        split_arrays[split_name] = arrays
        split_reports[split_name] = report

    lower_quantile, upper_quantile = raw_config["regime_quantiles"]
    reference_dim = max(matrix_dims)
    lower_threshold, upper_threshold = np.quantile(
        split_arrays["train"][reference_dim]["sharpes"],
        [lower_quantile, upper_quantile],
    )

    output_hashes = {}
    for split_name, arrays_by_dim in split_arrays.items():
        output_hashes[split_name] = {}
        for dim, arrays in arrays_by_dim.items():
            arrays["labels"] = np.where(
                arrays["sharpes"] < lower_threshold,
                0,
                np.where(arrays["sharpes"] > upper_threshold, 2, 1),
            ).astype(np.uint8)
            output_path = args.output / f"n{dim}" / f"{split_name}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **arrays)
            output_hashes[split_name][str(dim)] = sha256(output_path)
        labels = arrays_by_dim[reference_dim]["labels"]
        split_reports[split_name]["label_counts"] = {
            str(label): int(np.sum(labels == label)) for label in range(3)
        }
        print(f"{split_name}: {json.dumps(split_reports[split_name])}")

    manifest = {
        "market": raw_config["market"],
        "source": str(source),
        "source_sha256": sha256(source),
        "source_rows": int(panel.shape[0]),
        "source_assets": int(returns.shape[1]),
        "source_sessions": int(returns.shape[0]),
        "matrix_dims": list(matrix_dims),
        "nesting": "Each n_k matrix is the leading principal submatrix of n_max.",
        "conditioning": "Shared n_max equal-weight trailing-window Sharpe.",
        "sampling": asdict(sampling_config),
        "regime_definition": {
            "training_quantiles": [lower_quantile, upper_quantile],
            "frozen_thresholds": [float(lower_threshold), float(upper_threshold)],
        },
        "splits": split_reports,
        "output_sha256": output_hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
