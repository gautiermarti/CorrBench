#!/usr/bin/env python3
"""Measure permutation-invariant structural shift between dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrbench.metrics import energy_distance, sliced_wasserstein, structural_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    split_names = ["train", "validation", "test"]
    labels = {}
    features = {}
    for split in split_names:
        with np.load(args.data / f"{split}.npz") as source:
            matrices = source["matrices"]
            labels[split] = source["labels"]
        features[split], _ = structural_features(matrices)
    scale = np.maximum(features["train"].std(axis=0, ddof=1), 1e-6)

    report = {}
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        repetitions = []
        for seed in range(args.repeats):
            rng = np.random.default_rng(seed)
            first_indices = []
            second_indices = []
            for label in range(3):
                first_candidates = np.flatnonzero(labels[first] == label)
                second_candidates = np.flatnonzero(labels[second] == label)
                count = min(
                    args.per_class,
                    first_candidates.size,
                    second_candidates.size,
                )
                first_indices.extend(rng.choice(first_candidates, count, replace=False))
                second_indices.extend(
                    rng.choice(second_candidates, count, replace=False)
                )
            first_sample = features[first][first_indices] / scale
            second_sample = features[second][second_indices] / scale
            repetitions.append(
                {
                    "sliced_wasserstein": sliced_wasserstein(
                        first_sample,
                        second_sample,
                        seed=seed,
                        device="cpu",
                    ),
                    "energy_distance": energy_distance(
                        first_sample, second_sample, seed=seed
                    ),
                }
            )
        report[f"{first}_vs_{second}"] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in repetitions])),
                "std": float(np.std([row[metric] for row in repetitions], ddof=1)),
            }
            for metric in repetitions[0]
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
