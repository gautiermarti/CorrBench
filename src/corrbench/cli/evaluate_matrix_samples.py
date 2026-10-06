#!/usr/bin/env python3
"""Evaluate externally generated matrices, with optional CorrGAN repair."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from corrbench.geometry import corr_to_gft, nearest_correlation_torch
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


def distribution_metrics(
    real_corr: np.ndarray,
    generated_corr: np.ndarray,
    real_gft: np.ndarray,
    generated_gft: np.ndarray,
    *,
    train_gft_scale: float,
    train_features: np.ndarray,
    feature_scale: np.ndarray,
    train_portfolio: np.ndarray,
    portfolio_scale: np.ndarray,
    seed: int,
    include_diagnostics: bool,
    device: str = "cuda",
) -> dict[str, object]:
    real_features, feature_names = structural_features(real_corr)
    generated_features, _ = structural_features(generated_corr)
    standardized_real = real_features / feature_scale
    standardized_generated = generated_features / feature_scale
    real_portfolio, portfolio_names = portfolio_functionals(real_corr)
    generated_portfolio, _ = portfolio_functionals(generated_corr)
    report: dict[str, object] = {
        "gft_sliced_wasserstein": sliced_wasserstein(
            real_gft,
            generated_gft,
            scale=train_gft_scale,
            seed=seed,
            device=device,
        ),
        "orbit_gft_sliced_wasserstein": orbit_sliced_wasserstein(
            real_gft,
            generated_gft,
            scale=train_gft_scale,
            seed=seed,
            device=device,
        ),
        "feature_sliced_wasserstein": sliced_wasserstein(
            real_features,
            generated_features,
            scale=feature_scale,
            seed=seed,
            device="cpu",
        ),
        "feature_energy_distance": energy_distance(
            standardized_real, standardized_generated, seed=seed
        ),
        "feature_manifold": manifold_precision_recall(
            standardized_real,
            standardized_generated,
            neighbors=min(5, real_corr.shape[0] - 1),
            seed=seed,
        ),
        "feature_mean_absolute_z_error": float(
            np.mean(
                np.abs(real_features.mean(axis=0) - generated_features.mean(axis=0))
                / feature_scale
            )
        ),
        "feature_names": feature_names,
        "portfolio_functional_names": portfolio_names,
        "portfolio_sliced_wasserstein": sliced_wasserstein(
            real_portfolio,
            generated_portfolio,
            scale=portfolio_scale,
            seed=seed,
            device="cpu",
        ),
        "portfolio_energy_distance": energy_distance(
            real_portfolio / portfolio_scale,
            generated_portfolio / portfolio_scale,
            seed=seed,
        ),
        "validity": validity_summary(generated_corr),
    }
    if include_diagnostics:
        report["feature_classifier_two_sample"] = classifier_two_sample_auc(
            standardized_real, standardized_generated, seed=seed
        )
        report["nearest_train"] = nearest_train_diagnostic(
            train_features,
            real_features,
            generated_features,
            scale=feature_scale,
            seed=seed,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--repair-max-iterations", type=int, default=100)
    parser.add_argument("--repair-float64", action="store_true")
    parser.add_argument("--require-repair-convergence", action="store_true")
    parser.add_argument("--symmetrize", choices=["average", "upper"], default="average")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json; arrays use the matching .npz path")
    for target in (args.output, args.output.with_suffix(".npz")):
        if target.exists():
            raise FileExistsError(f"Choose a new evaluation output: {target}")

    with np.load(args.generated) as source:
        raw = source["matrices"].astype(np.float32)
        generated_labels = (
            source["labels"].astype(np.uint8) if "labels" in source.files else None
        )
        stored_selection = (
            source["selected_test_indices"].astype(np.int64)
            if "selected_test_indices" in source.files
            else None
        )
        stored_split = (
            str(source["split"].item()) if "split" in source.files else "test"
        )
        recorded_sampling_seconds = (
            float(source["sampling_seconds"]) if "sampling_seconds" in source else None
        )
    if stored_split != args.split:
        raise ValueError(
            f"generated split {stored_split!r} does not match requested {args.split!r}"
        )
    if args.symmetrize == "upper":
        upper = np.triu(raw, k=1)
        prepared = upper + np.swapaxes(upper, -1, -2)
    else:
        prepared = 0.5 * (raw + np.swapaxes(raw, -1, -2))
        diagonal = np.arange(prepared.shape[-1])
        prepared[:, diagonal, diagonal] = 0
    prepared += np.eye(prepared.shape[-1], dtype=np.float32)
    raw_validity = validity_summary(prepared)

    repair_info = None
    repair_started = time.perf_counter()
    if args.repair:
        repaired = []
        repairs = []
        convergence = []
        for start in range(0, prepared.shape[0], 128):
            batch = torch.from_numpy(prepared[start : start + 128]).to(args.device)
            if args.repair_float64:
                batch = batch.double()
            output, info = nearest_correlation_torch(
                batch,
                return_info=True,
                max_iterations=args.repair_max_iterations,
            )
            repaired.append(output.cpu().numpy().astype(np.float32))
            repairs.extend(
                torch.linalg.matrix_norm(output - batch).cpu().numpy().tolist()
            )
            convergence.append(bool(torch.all(info["converged"])))
        generated_corr = np.concatenate(repaired)
        repair_distance = np.linalg.norm(generated_corr - prepared, axis=(1, 2))
        prepared_norm = np.linalg.norm(prepared, axis=(1, 2))
        matrix_dim = prepared.shape[-1]
        repair_info = {
            "all_converged": all(convergence),
            "mean_frobenius_repair": float(np.mean(repairs)),
            "mean_relative_frobenius_repair": float(
                np.mean(repair_distance / np.maximum(prepared_norm, 1e-12))
            ),
            "mean_offdiagonal_rms_repair": float(
                np.mean(repair_distance / np.sqrt(matrix_dim * (matrix_dim - 1)))
            ),
        }
        repair_info["seconds"] = time.perf_counter() - repair_started
        repair_info["max_iterations"] = args.repair_max_iterations
        repair_info["dtype"] = "float64" if args.repair_float64 else "float32"
        if args.require_repair_convergence and not all(convergence):
            raise RuntimeError("Repair failed to converge; no fidelity report written")
    else:
        generated_corr = prepared

    with np.load(args.data / f"{args.split}.npz") as source:
        all_real_corr = source["matrices"]
        labels = source["labels"]
    with np.load(args.data / "gft" / f"{args.split}.npz") as source:
        all_real_gft = source["gft"]
    with np.load(args.data / "train.npz") as source:
        train_corr = source["matrices"]
    with np.load(args.data / "gft" / "train.npz") as source:
        train_gft_scale = float(source["gft"].std())

    if stored_selection is not None:
        selected = stored_selection
    else:
        rng = np.random.default_rng(args.seed)
        per_class = generated_corr.shape[0] // 3
        selected = np.concatenate(
            [
                rng.choice(np.flatnonzero(labels == label), per_class, replace=False)
                for label in range(3)
            ]
        )
    real_corr = all_real_corr[selected]
    real_gft = all_real_gft[selected]
    generated_corr = generated_corr[: selected.size]
    selected_labels = labels[selected]
    if generated_labels is not None and not np.array_equal(
        generated_labels[: selected.size], selected_labels
    ):
        raise ValueError("generated labels do not match selected real labels")

    generated_gft = []
    for start in range(0, generated_corr.shape[0], 256):
        batch = torch.from_numpy(generated_corr[start : start + 256]).to(args.device)
        generated_gft.append(corr_to_gft(batch).cpu().numpy())
    generated_gft = np.concatenate(generated_gft)

    train_features, _ = structural_features(train_corr)
    feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    train_portfolio, _ = portfolio_functionals(train_corr)
    portfolio_scale = np.maximum(train_portfolio.std(axis=0, ddof=1), 1e-6)
    report = {
        "samples": int(selected.size),
        "sampling_seconds": recorded_sampling_seconds,
        "coordinate_metric_device": args.device,
        "raw_validity_after_symmetry_and_diagonal": raw_validity,
        "repair": repair_info,
        "pooled": distribution_metrics(
            real_corr,
            generated_corr,
            real_gft,
            generated_gft,
            train_gft_scale=train_gft_scale,
            train_features=train_features,
            feature_scale=feature_scale,
            train_portfolio=train_portfolio,
            portfolio_scale=portfolio_scale,
            seed=args.seed,
            include_diagnostics=True,
            device=args.device,
        ),
        "by_class": {},
    }
    for label in range(3):
        mask = selected_labels == label
        report["by_class"][str(label)] = distribution_metrics(
            real_corr[mask],
            generated_corr[mask],
            real_gft[mask],
            generated_gft[mask],
            train_gft_scale=train_gft_scale,
            train_features=train_features,
            feature_scale=feature_scale,
            train_portfolio=train_portfolio,
            portfolio_scale=portfolio_scale,
            seed=args.seed + label + 1,
            include_diagnostics=False,
            device=args.device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        raw=raw[: selected.size],
        prepared=prepared[: selected.size],
        matrices=generated_corr,
        gft=generated_gft,
        labels=selected_labels,
        selected_test_indices=selected,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
