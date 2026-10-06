"""Evaluate fixed spatial/replay baselines and year-block aggregate-risk diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from corrbench.neuro import sha256
from corrbench.weather import aggregate_weights, spatial_kernel

LENGTHS = [100, 250, 500, 1000, 2000, 4000]
NUGGETS = [0.01, 0.05, 0.1, 0.2, 0.4]


def read_arrays(path):
    with np.load(path) as a:
        return {k: a[k] for k in a.files}


def aggregate_variances(matrices, coordinates):
    w = aggregate_weights(coordinates)
    return np.einsum("fi,nij,fj->nf", w, matrices, w, optimize=True)


def bootstrap_wasserstein(real, generated, counts):
    """Exact 1-Wasserstein for empirical year-resampled targets and fixed draws."""
    supports = np.unique(np.concatenate([real, generated]))
    widths = np.diff(supports)
    thresholds = supports[:-1]
    observed = (real[:, None] <= thresholds).astype(float)
    predicted = (generated[:, None] <= thresholds).mean(axis=0)
    boot_cdf = counts @ observed / counts.sum(axis=1, keepdims=True)
    result = np.abs(boot_cdf - predicted) @ widths
    unweighted = np.abs(observed.mean(axis=0) - predicted) @ widths
    np.testing.assert_allclose(unweighted, wasserstein_distance(real, generated))
    return float(unweighted), result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = read_arrays(args.data / "train.npz")
    lower = np.tril_indices(64, -1)
    chosen_kernels, search = {}, {}
    for quarter in range(4):
        mask = train["quarters"] == quarter
        rows = []
        for length in LENGTHS:
            for nugget in NUGGETS:
                errors = []
                for region in range(3):
                    indices = np.flatnonzero(mask & (train["region_ids"] == region))
                    kernel = spatial_kernel(
                        train["node_coordinates"][indices[0]], length, nugget
                    )
                    error = (
                        train["matrices"][indices][:, lower[0], lower[1]]
                        - (kernel[lower])
                    )
                    errors.append(float(np.mean(error**2)))
                rows.append(
                    {
                        "length_km": length,
                        "nugget": nugget,
                        "train_mse": float(np.mean(errors)),
                    }
                )
        chosen_kernels[quarter] = min(rows, key=lambda x: x["train_mse"])
        search[quarter] = rows
    args.output.mkdir(parents=True)
    selection = {
        "training_sha256": sha256(args.data / "train.npz"),
        "selection": "minimum train mean squared off-diagonal error; first tie",
        "kernels": chosen_kernels,
        "grid": search,
        "code_sha256": sha256(Path(__file__)),
        "transforms_sha256": sha256(
            Path(__file__).parents[1] / "src/corrbench/weather.py"
        ),
        "holdout_arrays_opened": False,
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    rows, split_reports = [], {}
    for split in ["validation", "test", "spatial_transfer"]:
        data = read_arrays(args.data / f"{split}.npz")
        years = np.unique(data["years"])
        rng = np.random.default_rng(7201)
        counts = rng.multinomial(
            len(years), np.repeat(1 / len(years), len(years)), size=1000
        )
        aggregated = {}
        for region in np.unique(data["region_ids"]):
            for quarter in range(4):
                indices = np.flatnonzero(
                    (data["region_ids"] == region) & (data["quarters"] == quarter)
                )
                indices = indices[np.argsort(data["years"][indices])]
                np.testing.assert_array_equal(data["years"][indices], years)
                coords = data["node_coordinates"][indices[0]]
                empirical = aggregate_variances(
                    data["empirical_matrices"][indices], coords
                )
                regularized = aggregate_variances(data["matrices"][indices], coords)
                quarter_rows = train["quarters"] == quarter
                pooled = train["matrices"][quarter_rows].mean(axis=0)
                settings = chosen_kernels[quarter]
                kernel = spatial_kernel(
                    coords, settings["length_km"], settings["nugget"]
                )
                candidates = {
                    "identity": np.eye(64)[None],
                    "spatial_kernel": kernel[None],
                    "pooled_grid_mean": pooled[None],
                }
                training_indices = np.flatnonzero(
                    quarter_rows & (train["region_ids"] == region)
                )
                if len(training_indices):
                    real_train = train["matrices"][training_indices]
                    candidates["conditional_mean"] = real_train.mean(axis=0)[None]
                    replay_rng = np.random.default_rng(2027)
                    draw_indices = replay_rng.choice(
                        training_indices, 1000, replace=True
                    )
                    candidates["conditional_replay"] = train["matrices"][draw_indices]
                    calibration = aggregate_variances(
                        train["empirical_matrices"][training_indices], coords
                    )
                else:
                    # Calibrate diagnostic scale from training-region patches.
                    calibration = np.concatenate(
                        [
                            aggregate_variances(
                                train["empirical_matrices"][
                                    quarter_rows & (train["region_ids"] == r)
                                ],
                                train["node_coordinates"][
                                    np.flatnonzero(train["region_ids"] == r)[0]
                                ],
                            )
                            for r in range(3)
                        ]
                    )
                scale = np.maximum(calibration.mean(axis=0), 1e-6)
                for method, matrices in candidates.items():
                    predicted = aggregate_variances(matrices, coords)
                    w1, boot, regularized_w1 = [], [], []
                    for f in range(5):
                        value, draws = bootstrap_wasserstein(
                            empirical[:, f], predicted[:, f], counts
                        )
                        w1.append(value / scale[f])
                        boot.append(draws / scale[f])
                        regularized_w1.append(
                            wasserstein_distance(regularized[:, f], predicted[:, f])
                            / scale[f]
                        )
                    point, boot_mean = float(np.mean(w1)), np.mean(boot, axis=0)
                    aggregated.setdefault(method, []).append((point, boot_mean))
                    rows.append(
                        {
                            "split": split,
                            "region": int(region),
                            "quarter": quarter,
                            "method": method,
                            "year_groups": len(years),
                            "normalized_aggregate_w1": point,
                            "normalized_regularized_target_w1": float(
                                np.mean(regularized_w1)
                            ),
                            "normalized_w1_by_aggregate": w1,
                            "regional_mean_variance_ratio": float(
                                predicted[:, 0].mean() / empirical[:, 0].mean()
                            ),
                            "empirical_mean_regional_variance": float(
                                empirical[:, 0].mean()
                            ),
                            "predicted_mean_regional_variance": float(
                                predicted[:, 0].mean()
                            ),
                            "regional_90pct_interval_coverage": float(
                                np.mean(
                                    (
                                        empirical[:, 0]
                                        >= np.quantile(predicted[:, 0], 0.05)
                                    )
                                    & (
                                        empirical[:, 0]
                                        <= np.quantile(predicted[:, 0], 0.95)
                                    )
                                )
                            ),
                            "interval_scope": "Quarter estimates; not forecasts",
                        }
                    )
        split_reports[split] = {
            method: {
                "normalized_aggregate_w1": float(np.mean([x[0] for x in values])),
                "year_bootstrap_95pct": np.quantile(
                    np.mean([x[1] for x in values], axis=0), [0.025, 0.975]
                ).tolist(),
                "contexts": len(values),
                "year_groups": len(years),
            }
            for method, values in aggregated.items()
        }
        print(split, json.dumps(split_reports[split]), flush=True)
    report = {
        "protocol": "weather_pilot_protocol_20261005",
        "scope": "Dependence-only spatial aggregate variance; not a weather forecast",
        "main_metric": (
            "1-Wasserstein to empirical quarter aggregate variances, divided by "
            "train mean variance per functional; macro average over contexts and "
            "regional/four-quadrant functionals"
        ),
        "bootstrap": (
            "1000 year-group draws seed 7201; resample the same years jointly "
            "across regions and quarters; conditional on fitted baselines. "
            "Descriptive resampling percentiles, not calibrated confidence limits: "
            "small-sample Wasserstein resampling can be upward biased."
        ),
        "limits": (
            "Years are not guaranteed iid; only nine test years. These are "
            "historical retrospective reanalysis periods. No learned generator "
            "or operational forecast benefit is established."
        ),
        "source_sha256": {
            s: sha256(args.data / f"{s}.npz")
            for s in ["train", "validation", "test", "spatial_transfer"]
        },
        "selection_sha256": sha256(args.output / "selection.json"),
        "summary": split_reports,
        "contexts": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
