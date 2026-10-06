#!/usr/bin/env python3
"""Fit and evaluate empirical/Gaussian/PCA baselines on a frozen test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrbench.baselines import (
    ConditionalGaussian,
    ConditionalLinearPCAGaussian,
    ConditionalPCAGaussian,
    empirical_bootstrap,
    nearest_covariate_bootstrap,
)
from corrbench.geometry import gft_to_corr
from corrbench.metrics import (
    classifier_two_sample_auc,
    energy_distance,
    manifold_precision_recall,
    nearest_train_diagnostic,
    orbit_sliced_wasserstein,
    portfolio_functionals,
    sliced_wasserstein,
    structural_features,
    validity_summary,
)


def stratified_subset(
    labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> np.ndarray:
    indices = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        indices.extend(
            rng.choice(candidates, min(per_class, candidates.size), replace=False)
        )
    return np.asarray(indices)[rng.permutation(len(indices))]


def invert_gft(
    vectors: np.ndarray, *, batch_size: int, device: str
) -> tuple[np.ndarray, dict[str, Any]]:
    matrices = []
    all_converged = True
    maximum_iterations = 0
    for start in range(0, vectors.shape[0], batch_size):
        batch = torch.from_numpy(vectors[start : start + batch_size]).to(device)
        correlations, info = gft_to_corr(
            batch,
            tolerance=1e-6,
            max_iterations=200,
            return_info=True,
        )
        matrices.append(correlations.cpu().numpy().astype(np.float32))
        all_converged &= bool(torch.all(info["converged"]))
        maximum_iterations = max(maximum_iterations, int(info["iterations"]))
    return np.concatenate(matrices), {
        "all_inverse_iterations_converged": all_converged,
        "maximum_inverse_iterations": maximum_iterations,
    }


def distribution_report(
    real_gft: np.ndarray,
    generated_gft: np.ndarray,
    real_corr: np.ndarray,
    generated_corr: np.ndarray,
    *,
    train_scale: float,
    train_feature_scale: np.ndarray,
    train_features: np.ndarray,
    train_portfolio_scale: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    real_features, feature_names = structural_features(real_corr)
    generated_features, _ = structural_features(generated_corr)
    standardized_real = real_features / train_feature_scale
    standardized_generated = generated_features / train_feature_scale
    real_portfolio, portfolio_names = portfolio_functionals(real_corr)
    generated_portfolio, _ = portfolio_functionals(generated_corr)
    return {
        "gft_sliced_wasserstein": sliced_wasserstein(
            real_gft,
            generated_gft,
            scale=train_scale,
            seed=seed,
        ),
        "orbit_gft_sliced_wasserstein": orbit_sliced_wasserstein(
            real_gft,
            generated_gft,
            scale=train_scale,
            seed=seed,
        ),
        "feature_sliced_wasserstein": sliced_wasserstein(
            real_features,
            generated_features,
            scale=train_feature_scale,
            seed=seed,
            device="cpu",
        ),
        "feature_energy_distance": energy_distance(
            standardized_real,
            standardized_generated,
            seed=seed,
        ),
        "feature_manifold": manifold_precision_recall(
            standardized_real,
            standardized_generated,
            seed=seed,
        ),
        "feature_classifier_two_sample": classifier_two_sample_auc(
            standardized_real,
            standardized_generated,
            seed=seed,
        ),
        "feature_mean_absolute_z_error": float(
            np.mean(
                np.abs(real_features.mean(axis=0) - generated_features.mean(axis=0))
                / train_feature_scale
            )
        ),
        "feature_names": feature_names,
        "nearest_train": nearest_train_diagnostic(
            train_features,
            real_features,
            generated_features,
            scale=train_feature_scale,
            seed=seed,
        ),
        "portfolio_functional_names": portfolio_names,
        "portfolio_sliced_wasserstein": sliced_wasserstein(
            real_portfolio,
            generated_portfolio,
            scale=train_portfolio_scale,
            seed=seed,
            device="cpu",
        ),
        "portfolio_energy_distance": energy_distance(
            real_portfolio / train_portfolio_scale,
            generated_portfolio / train_portfolio_scale,
            seed=seed,
        ),
        "validity": validity_summary(generated_corr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate the complete natural test distribution instead of stratifying.",
    )
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--generation-seed",
        type=int,
        help=(
            "Sampling/PCA seed; defaults to --seed while keeping test selection fixed."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inverse-batch-size", type=int, default=256)
    args = parser.parse_args()
    generation_seed = (
        args.seed if args.generation_seed is None else args.generation_seed
    )

    with np.load(args.data / "gft" / "train.npz") as train_source:
        train_gft = train_source["gft"]
        train_labels = train_source["labels"]
        train_sharpes = (
            train_source["sharpes"] if "sharpes" in train_source.files else None
        )
    with np.load(args.data / "gft" / "test.npz") as test_source:
        test_gft_all = test_source["gft"]
        test_labels_all = test_source["labels"]
        test_sharpes_all = (
            test_source["sharpes"] if "sharpes" in test_source.files else None
        )
    with np.load(args.data / "test.npz") as test_source:
        test_corr_all = test_source["matrices"]
    with np.load(args.data / "train.npz") as train_source:
        train_corr = train_source["matrices"]

    selection_rng = np.random.default_rng(args.seed)
    selected = (
        np.arange(test_labels_all.size)
        if args.all_samples
        else stratified_subset(test_labels_all, args.per_class, selection_rng)
    )
    test_gft = test_gft_all[selected]
    test_labels = test_labels_all[selected]
    test_sharpes = None if test_sharpes_all is None else test_sharpes_all[selected]
    test_corr = test_corr_all[selected]
    train_scale = float(train_gft.std())
    train_features, _ = structural_features(train_corr)
    train_feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    train_portfolio, _ = portfolio_functionals(train_corr)
    train_portfolio_scale = np.maximum(train_portfolio.std(axis=0, ddof=1), 1e-6)

    models: dict[str, Any] = {
        "iid_gft_gaussian": ConditionalGaussian(coordinatewise=False),
        "diagonal_gft_gaussian": ConditionalGaussian(coordinatewise=True),
        "pca_gft_gaussian": ConditionalPCAGaussian(
            components=args.pca_components, random_state=generation_seed
        ),
    }
    generated = {
        "empirical_bootstrap": empirical_bootstrap(
            train_gft,
            train_labels,
            test_labels,
            np.random.default_rng(generation_seed + 100),
        )
    }
    if train_sharpes is not None and test_sharpes is not None:
        generated["nearest_sharpe_bootstrap"] = nearest_covariate_bootstrap(
            train_gft,
            train_labels,
            train_sharpes,
            test_labels,
            test_sharpes,
            np.random.default_rng(generation_seed + 101),
        )
        conditional_pca = ConditionalLinearPCAGaussian(
            components=args.pca_components,
            random_state=generation_seed,
        )
        print("fitting linear_sharpe_pca_gft_gaussian", flush=True)
        conditional_pca.fit(train_gft, train_labels, train_sharpes)
        generated["linear_sharpe_pca_gft_gaussian"] = conditional_pca.sample(
            test_labels,
            test_sharpes,
            np.random.default_rng(generation_seed + 102),
        )
    for offset, (name, model) in enumerate(models.items(), start=200):
        print(f"fitting {name}", flush=True)
        model.fit(train_gft, train_labels)
        generated[name] = model.sample(
            test_labels, np.random.default_rng(generation_seed + offset)
        )

    report: dict[str, Any] = {
        "test_samples": int(test_labels.size),
        "sample_selection": "all" if args.all_samples else "stratified",
        "test_label_counts": {
            str(label): int(np.sum(test_labels == label)) for label in range(3)
        },
        "train_gft_scale": train_scale,
        "generation_seed": generation_seed,
        "models": {},
    }
    generated_correlations: dict[str, np.ndarray] = {}
    for name, generated_gft in generated.items():
        print(f"inverting and evaluating {name}", flush=True)
        generated_corr, inverse_info = invert_gft(
            generated_gft,
            batch_size=args.inverse_batch_size,
            device=args.device,
        )
        model_report = distribution_report(
            test_gft,
            generated_gft,
            test_corr,
            generated_corr,
            train_scale=train_scale,
            train_feature_scale=train_feature_scale,
            train_features=train_features,
            train_portfolio_scale=train_portfolio_scale,
            seed=args.seed,
        )
        model_report["inverse"] = inverse_info
        generated_correlations[name] = generated_corr
        model_report["by_class"] = {}
        for label in range(3):
            mask = test_labels == label
            model_report["by_class"][str(label)] = distribution_report(
                test_gft[mask],
                generated_gft[mask],
                test_corr[mask],
                generated_corr[mask],
                train_scale=train_scale,
                train_feature_scale=train_feature_scale,
                train_features=train_features[train_labels == label],
                train_portfolio_scale=train_portfolio_scale,
                seed=args.seed + label + 1,
            )
        report["models"][name] = model_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        selected_test_indices=selected,
        labels=test_labels,
        **generated_correlations,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
