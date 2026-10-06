#!/usr/bin/env python3
"""Aggregate statistical-control evaluations over generation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = {
    "feature_swd": ("feature_sliced_wasserstein",),
    "feature_energy": ("feature_energy_distance",),
    "feature_precision": ("feature_manifold", "precision"),
    "feature_recall": ("feature_manifold", "recall"),
    "valid_fraction": ("validity", "valid_fraction"),
    "nearest_train_ratio": ("nearest_train", "generated_to_real_ratio"),
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
    model_names = sorted(
        set.intersection(*(set(report["models"]) for report in reports))
    )
    models = {}
    for model_name in model_names:
        metrics = {}
        for metric_name, path in METRICS.items():
            values = [nested(report["models"][model_name], path) for report in reports]
            metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }
        models[model_name] = metrics
    summary = {
        "reports": [str(path) for path in args.reports],
        "generation_seeds": [report["generation_seed"] for report in reports],
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
