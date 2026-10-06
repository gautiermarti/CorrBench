"""Evaluate fixed DiffeoCFM/Gaussian weather draws against matched replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from corrflow.diffeocfm import (
    CorrelationGaussian,
    DiffeoVectorField,
    sample_correlations,
)
from scipy.stats import wasserstein_distance

from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary
from corrbench.neuro import sha256
from corrbench.weather import aggregate_weights, spatial_kernel


def read_arrays(path):
    with np.load(path) as a:
        return {key: a[key] for key in a.files}


def variance(matrix, weights):
    return np.einsum("fi,nij,fj->nf", weights, matrix, weights, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    provenance = json.loads((args.run / "provenance.json").read_text())
    for split, expected in provenance["source_hashes"].items():
        assert sha256(args.data / f"{split}.npz") == expected
    history = json.loads((args.run / "history.json").read_text())
    assert [row["epoch"] for row in history] == list(range(1, 201))
    scored = [row for row in history if "structural" in row]
    selected = min(scored, key=lambda row: row["structural"]["feature_swd"])
    best = torch.load(
        args.run / "best_structural.pt", map_location="cpu", weights_only=True
    )
    assert best["epoch"] == selected["epoch"]
    stored = torch.load(args.run / "prior.pt", map_location="cpu", weights_only=True)
    prior = CorrelationGaussian(
        matrix_dim=stored["matrix_dim"], classes=stored["classes"]
    )
    prior.load_state_dict(stored["state"])
    assert prior.classes == 12
    train = read_arrays(args.data / "train.npz")
    test = read_arrays(args.data / "test.npz")
    settings = json.loads((args.baseline / "selection.json").read_text())
    assert settings["training_sha256"] == sha256(args.data / "train.npz")
    prior_report = json.loads((args.baseline / "report.json").read_text())
    labels = np.repeat(np.arange(12), 200)
    source_paths = [
        args.run / name
        for name in [
            "prior.pt",
            "best_structural.pt",
            "last.pt",
            "history.json",
            "config.yaml",
            "provenance.json",
        ]
    ]
    args.output.mkdir(parents=True)
    audit = {
        "status": "passed",
        "scope": "Post-baseline exploratory single-training-seed study",
        "selected_epoch": selected["epoch"],
        "completed_epochs": 200,
        "validation_feature_swd": selected["structural"]["feature_swd"],
        "code_sha256": sha256(Path(__file__)),
        "source_sha256": {
            s: sha256(args.data / f"{s}.npz") for s in ["train", "validation", "test"]
        },
        "run_sha256": {p.name: sha256(p) for p in source_paths},
    }
    (args.output / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    generated, metadata = {}, {}
    for family, weights in [
        ("gaussian", None),
        ("diffeocfm", "best_structural"),
        ("diffeocfm_last", "last"),
    ]:
        model = None
        if weights:
            checkpoint = torch.load(
                args.run / f"{weights}.pt", map_location="cpu", weights_only=True
            )
            model = DiffeoVectorField(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model"])
        for seed in range(2027, 2032):
            matrices, timing = sample_correlations(
                prior, model, labels, seed=seed, **config["sampling"]
            )
            name = f"{family}_{seed}"
            path = args.output / f"{name}.npz"
            np.savez_compressed(path, matrices=matrices, labels=labels)
            generated[name] = matrices
            metadata[name] = {
                "sha256": sha256(path),
                "timing": timing,
                "validity": validity_summary(matrices),
            }
            assert metadata[name]["validity"]["valid_fraction"] == 1
            print(
                json.dumps({"generated": name, "seconds": timing["sampling_seconds"]}),
                flush=True,
            )
    for seed in range(2027, 2032):
        rng = np.random.default_rng(seed)
        generated[f"replay_{seed}"] = np.concatenate(
            [
                train["matrices"][
                    rng.choice(
                        np.flatnonzero(train["labels"] == label), 200, replace=True
                    )
                ]
                for label in range(12)
            ]
        )
    real_features, _ = structural_features(test["matrices"])
    train_features, _ = structural_features(train["matrices"])
    feature_scale = np.maximum(train_features.std(0, ddof=1), 1e-6)
    rows = []
    for name in [*generated, "conditional_mean", "pooled_grid_mean", "spatial_kernel"]:
        contexts, fidelity = [], []
        for label in range(12):
            region, quarter = divmod(label, 4)
            mask = test["labels"] == label
            train_mask = train["labels"] == label
            coords = test["node_coordinates"][np.flatnonzero(mask)[0]]
            weights = aggregate_weights(coords)
            target = variance(test["empirical_matrices"][mask], weights)
            scale = np.maximum(
                variance(train["empirical_matrices"][train_mask], weights).mean(0),
                1e-6,
            )
            if name in generated:
                candidate = generated[name][labels == label]
            elif name == "conditional_mean":
                candidate = train["matrices"][train_mask].mean(0)[None]
            elif name == "pooled_grid_mean":
                candidate = train["matrices"][train["quarters"] == quarter].mean(0)[
                    None
                ]
            else:
                kernel = settings["kernels"][str(quarter)]
                candidate = spatial_kernel(
                    coords, kernel["length_km"], kernel["nugget"]
                )[None]
            functional = variance(candidate, weights)
            values = [
                wasserstein_distance(target[:, k], functional[:, k]) / scale[k]
                for k in range(5)
            ]
            contexts.append(
                {
                    "region": region,
                    "quarter": quarter,
                    "normalized_w1_by_function": values,
                    "mean": np.mean(values),
                }
            )
            if name in generated:
                features, _ = structural_features(candidate)
                fidelity.append(
                    sliced_wasserstein(
                        real_features[mask],
                        features,
                        scale=feature_scale,
                        seed=2027,
                        device="cpu",
                    )
                )
        row = {
            "name": name,
            "aggregate_w1": float(np.mean([c["mean"] for c in contexts])),
            "contexts": contexts,
        }
        if fidelity:
            row["macro_condition_structure_swd"] = float(np.mean(fidelity))
        if name in {"conditional_mean", "pooled_grid_mean", "spatial_kernel"}:
            expected = prior_report["summary"]["test"][name]["normalized_aggregate_w1"]
            np.testing.assert_allclose(row["aggregate_w1"], expected, atol=1e-12)
        rows.append(row)
    summary = {}
    for family in ["gaussian", "diffeocfm", "diffeocfm_last", "replay"]:
        selected_rows = [r for r in rows if r["name"].rsplit("_", 1)[0] == family]
        assert len(selected_rows) == 5
        values = [r["aggregate_w1"] for r in selected_rows]
        summary[family] = {
            "aggregate_w1_mean": float(np.mean(values)),
            "generation_sd": float(np.std(values, ddof=1)),
            "structure_swd_mean": float(
                np.mean([r["macro_condition_structure_swd"] for r in selected_rows])
            ),
        }
    report = {
        **audit,
        "summary": summary,
        "rows": rows,
        "generated": metadata,
        "generation_seeds": list(range(2027, 2032)),
        "per_condition": 200,
        "training_seed": 1701,
        "test_year_groups": 9,
        "limitations": "Exploratory inspected periods; one training seed. "
        "Generation SD is not uncertainty over future years. "
        "Structural SWD uses matched subsets of only nine "
        "matrices per condition; read utility separately.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
