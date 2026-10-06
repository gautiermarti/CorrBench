#!/usr/bin/env python3
"""Aggregate comparable CorrFlow evaluation reports across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = {
    "gft_swd": ("gft_sliced_wasserstein",),
    "orbit_gft_swd": ("orbit_gft_sliced_wasserstein",),
    "feature_swd": ("feature_sliced_wasserstein",),
    "feature_energy": ("feature_energy_distance",),
    "feature_precision": ("feature_manifold", "precision"),
    "feature_recall": ("feature_manifold", "recall"),
    "classifier_auc": ("feature_classifier_two_sample", "auc"),
    "nearest_train_ratio": ("nearest_train", "generated_to_real_ratio"),
    "portfolio_swd": ("portfolio_sliced_wasserstein",),
    "portfolio_energy": ("portfolio_energy_distance",),
    "valid_fraction": ("validity", "valid_fraction"),
}
ROOT_METRICS = {
    "sampling_seconds_per_1000": ("sampling_seconds_per_1000",),
    "inverse_seconds": ("inverse_seconds",),
    "inverse_failed_samples": ("inverse_failed_samples",),
    "raw_valid_fraction": (
        "raw_validity_after_symmetry_and_diagonal",
        "valid_fraction",
    ),
    "mean_frobenius_repair": ("repair", "mean_frobenius_repair"),
    "mean_relative_frobenius_repair": (
        "repair",
        "mean_relative_frobenius_repair",
    ),
    "mean_offdiagonal_rms_repair": (
        "repair",
        "mean_offdiagonal_rms_repair",
    ),
}


def nested(source: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = source
    for key in path:
        value = value[key]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.reports]
    rows: dict[str, list[float]] = {name: [] for name in METRICS}
    root_rows: dict[str, list[float]] = {name: [] for name in ROOT_METRICS}
    for report in reports:
        pooled = report["pooled"]
        for name, path in METRICS.items():
            rows[name].append(nested(pooled, path))
        for name, path in ROOT_METRICS.items():
            try:
                root_rows[name].append(nested(report, path))
            except (KeyError, TypeError):
                continue
    rows.update({name: values for name, values in root_rows.items() if values})
    summary = {
        "reports": [str(path) for path in args.reports],
        "seeds": len(reports),
        "metrics": {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }
            for name, values in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
