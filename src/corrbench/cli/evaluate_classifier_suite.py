#!/usr/bin/env python3
"""Compare all classifier two-sample AUCs at one stratified sample size."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from corrbench.metrics import classifier_two_sample_auc, structural_features


def parse_input(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected NAME=PATH or NAME=PATH:KEY")
    name, source = parts
    if ":" in source:
        path_text, key = source.rsplit(":", maxsplit=1)
    else:
        path_text, key = source, "matrices"
    return name, Path(path_text), key


def stratified_indices(
    labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> np.ndarray:
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if candidates.size < per_class:
            raise ValueError(
                f"Class {label} has {candidates.size} samples, fewer than {per_class}."
            )
        selected.extend(rng.choice(candidates, per_class, replace=False))
    return np.asarray(selected, dtype=np.int64)


def repeated_auc(
    real: np.ndarray,
    generated: np.ndarray,
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    *,
    per_class: int,
    repeats: int,
    seed: int,
) -> list[float]:
    values = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        real_index = stratified_indices(real_labels, per_class, rng)
        generated_index = stratified_indices(generated_labels, per_class, rng)
        values.append(
            classifier_two_sample_auc(
                real[real_index],
                generated[generated_index],
                seed=seed + repeat,
            )["auc"]
        )
    return values


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", action="append", type=parse_input, default=[])
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.data / "train.npz") as source:
        train_corr = source["matrices"]
        train_labels = source["labels"]
    with np.load(args.data / "test.npz") as source:
        test_corr = source["matrices"]
        test_labels = source["labels"]
    train_features, _ = structural_features(train_corr)
    test_features, _ = structural_features(test_corr)

    real_real = []
    train_test = []
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        first = []
        second = []
        for label in np.unique(test_labels):
            candidates = rng.permutation(np.flatnonzero(test_labels == label))
            if candidates.size < 2 * args.per_class:
                raise ValueError(
                    f"Test class {label} cannot supply two disjoint samples of "
                    f"{args.per_class}."
                )
            first.extend(candidates[: args.per_class])
            second.extend(candidates[args.per_class : 2 * args.per_class])
        first = np.asarray(first, dtype=np.int64)
        second = np.asarray(second, dtype=np.int64)
        real_real.append(
            classifier_two_sample_auc(
                test_features[first], test_features[second], seed=args.seed + repeat
            )["auc"]
        )
        train_index = stratified_indices(train_labels, args.per_class, rng)
        test_index = stratified_indices(test_labels, args.per_class, rng)
        train_test.append(
            classifier_two_sample_auc(
                train_features[train_index],
                test_features[test_index],
                seed=args.seed + repeat,
            )["auc"]
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for name, path, key in args.input:
        with np.load(path) as source:
            generated_corr = source[key]
            generated_labels = source["labels"]
            selected = source["selected_test_indices"]
        order = np.argsort(selected)
        selected = selected[order]
        generated_corr = generated_corr[order]
        generated_labels = generated_labels[order]
        generated_features, _ = structural_features(generated_corr)
        candidate_real = test_features[selected]
        candidate_labels = test_labels[selected]
        if not np.array_equal(candidate_labels, generated_labels):
            raise ValueError(f"Condition labels do not match for {path}:{key}")
        values = repeated_auc(
            candidate_real,
            generated_features,
            candidate_labels,
            generated_labels,
            per_class=args.per_class,
            repeats=args.repeats,
            seed=args.seed,
        )
        grouped[name].append(
            {
                "source": f"{path}:{key}",
                "repeat_auc": values,
                "mean_auc": float(np.mean(values)),
            }
        )

    methods = {}
    for name, rows in grouped.items():
        seed_means = [float(row["mean_auc"]) for row in rows]
        methods[name] = {
            "mean": float(np.mean(seed_means)),
            "std_across_model_or_generation_seeds": (
                float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            ),
            "seed_means": seed_means,
            "runs": rows,
        }

    report = {
        "per_class_per_side": args.per_class,
        "samples_per_side": 3 * args.per_class,
        "repeats_per_run": args.repeats,
        "real_real": summarize(real_real),
        "train_test": summarize(train_test),
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
