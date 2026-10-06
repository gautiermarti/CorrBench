"""Controlled node-label scrambling diagnostic; no model fitting or selection."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from corrbench.metrics import structural_features
from corrbench.weather import aggregate_weights


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classifier_features(matrices):
    i, j = np.triu_indices(matrices.shape[-1], 1)
    return np.arctanh(np.clip(matrices[:, i, j], -1 + 1e-7, 1 - 1e-7))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    brain = args.bench_root / "artifacts/abide_ho_20261005"
    utility = args.bench_root / "artifacts/abide_complete_utility_20261005"
    weather = args.bench_root / "artifacts/weather_ncep_20261005"
    selection = json.loads((utility / "selection.json").read_text())
    for name in ["real_only.joblib", "feature_scaler.joblib"]:
        assert sha(utility / name) == selection["model_sha256"][name]
    model, scaler = [
        joblib.load(utility / name)
        for name in ["real_only.joblib", "feature_scaler.joblib"]
    ]
    with np.load(brain / "test.npz") as a:
        matrices, labels = a["matrices"], a["labels"]
    def predict(x):
        return model.predict_proba(scaler.transform(classifier_features(x)))[:, 1]
    original_probability = predict(matrices)
    reference_auc = float(roc_auc_score(labels, original_probability))
    recorded = json.loads((utility / "test_report.json").read_text())["arms"]
    np.testing.assert_allclose(
        reference_auc,
        next(r["roc_auc"] for r in recorded if r["name"] == "real_only"),
        atol=1e-14,
    )
    brain_features, _ = structural_features(matrices)
    with np.load(brain / "train.npz") as a:
        scale = np.maximum(structural_features(a["matrices"])[0].std(0, ddof=1), 1e-6)
    brain_rows, predictions = [], {"original": original_probability, "labels": labels}
    for seed in range(2027, 2037):
        permutation = np.random.default_rng(seed).permutation(matrices.shape[-1])
        scrambled = matrices[:, permutation][:, :, permutation]
        new_features, _ = structural_features(scrambled)
        error = float(np.abs((new_features - brain_features) / scale).max())
        assert error < 1e-8
        probabilities = predict(scrambled)
        inverse = np.argsort(permutation)
        restored = scrambled[:, inverse][:, :, inverse]
        np.testing.assert_array_equal(restored, matrices)
        np.testing.assert_array_equal(predict(restored), original_probability)
        brain_rows.append(
            {
                "permutation_seed": seed,
                "max_standardized_structural_feature_error": error,
                "auc_wrong_roi_labels": float(roc_auc_score(labels, probabilities)),
                "auc_canonicalized_labels": reference_auc,
            }
        )
        predictions[f"scrambled_{seed}"] = probabilities
    with np.load(weather / "test.npz") as a:
        matrices, coordinates = a["matrices"], a["node_coordinates"]
    features, _ = structural_features(matrices)
    with np.load(weather / "train.npz") as a:
        scale = np.maximum(structural_features(a["matrices"])[0].std(0, ddof=1), 1e-6)
    weights = np.stack([aggregate_weights(c) for c in coordinates])
    original = np.einsum("nfi,nij,nfj->nf", weights, matrices, weights)
    weather_rows = []
    for seed in range(2027, 2037):
        permutation = np.random.default_rng(seed).permutation(64)
        scrambled = matrices[:, permutation][:, :, permutation]
        error = float(
            np.abs((structural_features(scrambled)[0] - features) / scale).max()
        )
        assert error < 1e-8
        incorrect = np.einsum("nfi,nij,nfj->nf", weights, scrambled, weights)
        correct = np.einsum(
            "nfi,nij,nfj->nf",
            weights[:, :, permutation],
            scrambled,
            weights[:, :, permutation],
        )
        np.testing.assert_allclose(correct, original, atol=1e-12, rtol=0)
        relative_error = np.abs(incorrect - original) / original
        weather_rows.append(
            {
                "permutation_seed": seed,
                "max_standardized_structural_feature_error": error,
                "mean_relative_error_whole_region": float(relative_error[:, 0].mean()),
                "mean_relative_error_quadrants": float(relative_error[:, 1:].mean()),
                "co_relabeling_max_variance_error": float(
                    np.abs(correct - original).max()
                ),
            }
        )
    source_paths = [
        brain / "train.npz",
        brain / "test.npz",
        weather / "train.npz",
        weather / "test.npz",
        utility / "real_only.joblib",
        utility / "feature_scaler.joblib",
        utility / "selection.json",
    ]
    aucs = [r["auc_wrong_roi_labels"] for r in brain_rows]
    report = {
        "scope": "Post-pilot controlled diagnostic on previously inspected test data",
        "design": "Ten fixed permutation seeds 2027–2036, one common permutation per "
        "dataset draw; all results retained, no fitting or selection",
        "interpretation": "Anonymous structural summaries cannot detect wrong named "
        "ROI/location assignments. Simultaneous node and attribute "
        "relabeling preserves the original task calculations.",
        "brain": {
            "real_only_auc": reference_auc,
            "scrambled_auc_mean": float(np.mean(aucs)),
            "scrambled_auc_range": [min(aucs), max(aucs)],
            "rows": brain_rows,
        },
        "weather": {
            "mean_relative_quadrant_variance_error": float(
                np.mean([r["mean_relative_error_quadrants"] for r in weather_rows])
            ),
            "rows": weather_rows,
        },
        "source_sha256": {
            str(p.relative_to(args.bench_root)): sha(p) for p in source_paths
        },
        "code_sha256": sha(Path(__file__)),
        "limitations": [
            "Diagnostic corruptions, not samples from a trained generator.",
            "One brain classifier and site; these are not generalization intervals.",
            "Weather uses unit-scale OAS targets and fixed quadrant definitions.",
            "Permutation-invariant scores remain appropriate for anonymous tasks.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "brain_predictions.npz", **predictions)
    print(
        json.dumps(
            {key: report[key] for key in ["scope", "brain", "weather"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
