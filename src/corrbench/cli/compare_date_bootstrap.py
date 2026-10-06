#!/usr/bin/env python3
"""Paired date-block bootstrap for two families of generator evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from corrbench.metrics import structural_features


def load_generated(
    paths: list[Path], selected: np.ndarray | None
) -> tuple[list[np.ndarray], np.ndarray]:
    features = []
    for path in paths:
        with np.load(path) as source:
            current = source["selected_test_indices"]
            if selected is None:
                selected = current
            elif not np.array_equal(current, selected):
                raise ValueError(f"Selected test indices differ in {path}")
            current_features, _ = structural_features(source["matrices"])
            features.append(current_features)
    if selected is None:
        raise ValueError("At least one evaluation artifact is required")
    return features, selected


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "ci_025": float(np.quantile(values, 0.025)),
        "ci_975": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--first", type=Path, nargs="+", required=True)
    parser.add_argument("--second", type=Path, nargs="+", required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if len(args.first) != len(args.second):
        raise ValueError("Generator families must contain paired seed counts")
    if args.repeats < 2:
        raise ValueError("At least two bootstrap repeats are required")

    first_features, selected = load_generated(args.first, None)
    second_features, selected = load_generated(args.second, selected)
    with np.load(args.data / "test.npz") as source:
        real_corr = source["matrices"][selected]
        selected_dates = source["end_dates"][selected].astype("datetime64[D]")
    with np.load(args.data / "train.npz") as source:
        train_features, _ = structural_features(source["matrices"])
    real_features, feature_names = structural_features(real_corr)
    scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    real = real_features / scale
    families = {
        args.first_name: [features / scale for features in first_features],
        args.second_name: [features / scale for features in second_features],
    }

    direction_generator = torch.Generator("cpu").manual_seed(args.seed)
    directions = torch.randn(
        real.shape[1],
        args.projections,
        generator=direction_generator,
        dtype=torch.float32,
    )
    directions /= torch.linalg.vector_norm(directions, dim=0, keepdim=True)
    directions = directions.numpy()
    real_projected = real.astype(np.float32) @ directions
    projected = {
        name: [features.astype(np.float32) @ directions for features in seed_features]
        for name, seed_features in families.items()
    }
    distance_matrices = {
        name: [
            {
                "cross": cdist(real, features),
                "generated": cdist(features, features),
            }
            for features in seed_features
        ]
        for name, seed_features in families.items()
    }
    real_distances = cdist(real, real)

    unique_dates = np.unique(selected_dates)
    date_groups = [np.flatnonzero(selected_dates == date) for date in unique_dates]
    full_indices = np.arange(selected.size)

    def metric_for_indices(
        indices: np.ndarray, name: str, seed_index: int
    ) -> tuple[float, float]:
        real_sorted = np.sort(real_projected[indices], axis=0)
        generated_sorted = np.sort(projected[name][seed_index][indices], axis=0)
        swd = np.sqrt(np.mean((real_sorted - generated_sorted) ** 2, axis=0)).mean()
        matrices = distance_matrices[name][seed_index]
        energy = (
            2 * matrices["cross"][np.ix_(indices, indices)].mean()
            - real_distances[np.ix_(indices, indices)].mean()
            - matrices["generated"][np.ix_(indices, indices)].mean()
        )
        return float(swd), float(energy)

    point = {}
    point_by_seed = {}
    for name in families:
        rows = [
            metric_for_indices(full_indices, name, seed_index)
            for seed_index in range(len(families[name]))
        ]
        point_by_seed[name] = {
            "feature_swd": [row[0] for row in rows],
            "feature_energy": [row[1] for row in rows],
        }
        point[name] = {
            "feature_swd": float(np.mean([row[0] for row in rows])),
            "feature_energy": float(np.mean([row[1] for row in rows])),
        }

    rng = np.random.default_rng(args.seed)
    date_only = {"feature_swd": [], "feature_energy": []}
    joint = {"feature_swd": [], "feature_energy": []}
    for _ in range(args.repeats):
        sampled_date_groups = rng.choice(len(date_groups), len(date_groups))
        indices = np.concatenate([date_groups[index] for index in sampled_date_groups])
        seed_rows = {}
        for name in families:
            seed_rows[name] = [
                metric_for_indices(indices, name, seed_index)
                for seed_index in range(len(families[name]))
            ]
        for metric_index, metric_name in enumerate(("feature_swd", "feature_energy")):
            first_all = np.asarray(
                [row[metric_index] for row in seed_rows[args.first_name]]
            )
            second_all = np.asarray(
                [row[metric_index] for row in seed_rows[args.second_name]]
            )
            date_only[metric_name].append(float(second_all.mean() - first_all.mean()))
            sampled_seeds = rng.choice(len(first_all), len(first_all))
            joint[metric_name].append(
                float(
                    second_all[sampled_seeds].mean() - first_all[sampled_seeds].mean()
                )
            )

    report = {
        "comparison": (
            f"{args.second_name} minus {args.first_name}; positive favors first"
        ),
        "data": str(args.data),
        "samples": int(selected.size),
        "unique_date_blocks": int(unique_dates.size),
        "paired_seeds": len(args.first),
        "bootstrap_repeats": args.repeats,
        "feature_names": feature_names,
        "point_estimates": point,
        "point_estimates_by_seed": point_by_seed,
        "point_differences": {
            metric: point[args.second_name][metric] - point[args.first_name][metric]
            for metric in ("feature_swd", "feature_energy")
        },
        "date_block_bootstrap_differences": {
            metric: summarize(np.asarray(values))
            for metric, values in date_only.items()
        },
        "date_and_paired_seed_bootstrap_differences": {
            metric: summarize(np.asarray(values)) for metric, values in joint.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
