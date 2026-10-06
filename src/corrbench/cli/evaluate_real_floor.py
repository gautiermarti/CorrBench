#!/usr/bin/env python3
"""Estimate finite-sample real-versus-real distances on a frozen test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrbench.metrics import (
    classifier_two_sample_auc,
    energy_distance,
    manifold_precision_recall,
    orbit_sliced_wasserstein,
    sliced_wasserstein,
    structural_features,
)


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1)),
        }
        for metric in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    with np.load(args.data / "test.npz") as source:
        correlations = source["matrices"]
        labels = source["labels"]
    with np.load(args.data / "gft" / "test.npz") as source:
        gft = source["gft"]
    with np.load(args.data / "train.npz") as source:
        train_correlations = source["matrices"]
        train_labels = source["labels"]
    with np.load(args.data / "gft" / "train.npz") as source:
        gft_scale = float(source["gft"].std())
    features, _ = structural_features(correlations)
    train_features, _ = structural_features(train_correlations)
    feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)

    rows = []
    manifold_rows = []
    classifier_rows = []
    temporal_classifier_rows = []
    for seed in range(args.repeats):
        rng = np.random.default_rng(seed)
        historical_rng = np.random.default_rng(seed + 10_000)
        first = []
        second = []
        historical = []
        for label in range(3):
            candidates = rng.permutation(np.flatnonzero(labels == label))
            count = min(args.per_class, candidates.size // 2)
            first.extend(candidates[:count])
            second.extend(candidates[count : 2 * count])
            train_candidates = np.flatnonzero(train_labels == label)
            historical.extend(
                historical_rng.choice(train_candidates, count, replace=False)
            )
        first = np.asarray(first)
        second = np.asarray(second)
        historical = np.asarray(historical)
        first_features = features[first] / feature_scale
        second_features = features[second] / feature_scale
        historical_features = train_features[historical] / feature_scale
        rows.append(
            {
                "gft_swd": sliced_wasserstein(
                    gft[first], gft[second], scale=gft_scale, seed=seed
                ),
                "orbit_gft_swd": orbit_sliced_wasserstein(
                    gft[first], gft[second], scale=gft_scale, seed=seed
                ),
                "feature_swd": sliced_wasserstein(
                    first_features, second_features, seed=seed, device="cpu"
                ),
                "feature_energy": energy_distance(
                    first_features, second_features, seed=seed
                ),
            }
        )
        manifold_rows.append(
            manifold_precision_recall(first_features, second_features, seed=seed)
        )
        classifier_rows.append(
            classifier_two_sample_auc(first_features, second_features, seed=seed)
        )
        temporal_classifier_rows.append(
            classifier_two_sample_auc(first_features, historical_features, seed=seed)
        )
    report = summarize(rows)
    report["manifold_precision"] = summarize(manifold_rows)["precision"]
    report["manifold_recall"] = summarize(manifold_rows)["recall"]
    report["classifier_auc"] = summarize(classifier_rows)["auc"]
    report["train_test_classifier_auc"] = summarize(temporal_classifier_rows)["auc"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
