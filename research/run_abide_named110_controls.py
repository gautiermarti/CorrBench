"""Run declared simple controls; seal both cohorts' selections before scoring."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import corrflow.diffeocfm as gaussian
import fmri_utility as utility
import numpy as np
import sklearn
import torch
from build_abide_named110 import PROTOCOL, PROTOCOL_SHA

import corrbench.neuro as neuro
from corrbench.neuro import sha256

COHORTS = ("same601", "expanded762")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def code_hashes():
    return {
        "runner": sha256(Path(__file__)),
        "utility": sha256(Path(utility.__file__)),
        "gaussian": sha256(Path(gaussian.__file__)),
        "estimator": sha256(Path(neuro.__file__)),
    }


def select(bench, root):
    output = root / "controls"
    if output.exists():
        raise FileExistsError(output)
    if sha256(bench / "docs" / f"{PROTOCOL}.md") != PROTOCOL_SHA:
        raise ValueError("Declared protocol changed")
    construction_path = root / "data/construction_audit.json"
    construction = json.loads(construction_path.read_text())
    if construction["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Construction uses another protocol")
    if construction["estimator_code_sha256"] != code_hashes()["estimator"]:
        raise ValueError("Estimator changed after construction")
    frozen_report = json.loads((bench / "docs/fmri_pilot_results.json").read_text())
    frozen_predictions = (
        bench / "artifacts/abide_complete_utility_20261005/test_predictions.npz"
    )
    if sha256(frozen_predictions) != frozen_report["sources"]["test_predictions.npz"]:
        raise ValueError("Frozen comparison predictions changed")
    output.mkdir()
    torch.set_num_threads(4)
    records = {}
    for cohort in COHORTS:
        data = root / "data" / cohort
        manifest = json.loads((data / "manifest.json").read_text())
        if (
            sha256(data / "manifest.json")
            != construction["cohorts"][cohort]["manifest_sha256"]
        ):
            raise ValueError("Construction manifest changed")
        for split in ("train", "validation"):
            if sha256(data / f"{split}.npz") != manifest["splits"][split]["sha256"]:
                raise ValueError("Selection data changed")
        destination = output / cohort
        destination.mkdir()
        with np.load(data / "train.npz") as arrays:
            train, labels, nodes = (
                arrays["matrices"],
                arrays["labels"],
                arrays["node_ids"],
            )
        start = time.perf_counter()
        prior, _ = gaussian.CorrelationGaussian.fit(train, labels)
        prior_seconds = time.perf_counter() - start
        prior_path = destination / "prior.pt"
        torch.save(
            {"matrix_dim": 110, "classes": 2, "state": prior.state_dict()}, prior_path
        )
        per_class = (len(train) + 1) // 2
        generated_labels = np.repeat(np.arange(2), per_class)
        arms, timings = [], []
        for seed in range(2027, 2032):
            matrices, timing = gaussian.sample_correlations(
                prior, None, generated_labels, seed=seed, batch_size=64
            )
            utility.features(matrices)  # Full finite/symmetry/unit-diagonal/PD checks.
            path = destination / f"gaussian_{seed}.npz"
            np.savez_compressed(
                path,
                matrices=matrices,
                labels=generated_labels,
                node_ids=nodes,
                seed=seed,
            )
            arms.append(
                {
                    "name": f"gaussian_{seed}",
                    "kind": "generated",
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
            )
            timings.append({"seed": seed, **timing})
            print(json.dumps({"cohort": cohort, "generated_seed": seed}), flush=True)
        generated_manifest = destination / "generated.json"
        write(generated_manifest, arms)
        write(
            destination / "generation_audit.json",
            {
                "train_sha256": sha256(data / "train.npz"),
                "node_ids": nodes.tolist(),
                "prior_sha256": sha256(prior_path),
                "prior_fit_seconds": prior_seconds,
                "shrinkage": prior.shrinkage,
                "generation": timings,
                "full_covariance": True,
                "device": "cpu",
                "code_sha256": code_hashes(),
                "all_generated_matrices_valid": True,
            },
        )
        del prior
        utility.select(data, destination / "utility", generated_manifest)
        records[cohort] = {
            "selection_sha256": sha256(destination / "utility/selection.json"),
            "generation_audit_sha256": sha256(destination / "generation_audit.json"),
            "generated_manifest_sha256": sha256(generated_manifest),
            "data_manifest_sha256": sha256(data / "manifest.json"),
        }
    write(
        output / "selection_registration.json",
        {
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA,
            "construction_sha256": sha256(construction_path),
            "cohorts": records,
            "code_sha256": code_hashes(),
            "frozen_predictions_sha256": sha256(frozen_predictions),
            "test_scored_in_this_study": False,
            "fresh_holdout": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
    )


def evaluate(bench, root):
    output = root / "controls"
    registration = json.loads((output / "selection_registration.json").read_text())
    if registration["code_sha256"] != code_hashes():
        raise ValueError("Implementation changed after selection")
    if registration["protocol_sha256"] != sha256(bench / "docs" / f"{PROTOCOL}.md"):
        raise ValueError("Study protocol changed")
    if registration["construction_sha256"] != sha256(
        root / "data/construction_audit.json"
    ):
        raise ValueError("Construction record changed")
    # Verify both sealed selections before either cohort's test evaluation.
    for cohort in COHORTS:
        record = registration["cohorts"][cohort]
        directory = output / cohort
        paths = {
            "selection": directory / "utility/selection.json",
            "generation_audit": directory / "generation_audit.json",
            "generated_manifest": directory / "generated.json",
            "data_manifest": root / "data" / cohort / "manifest.json",
        }
        for key, path in paths.items():
            if sha256(path) != record[f"{key}_sha256"]:
                raise ValueError(f"Changed {cohort} {key}")
        audit = json.loads(paths["generation_audit"].read_text())
        if sha256(directory / "prior.pt") != audit["prior_sha256"]:
            raise ValueError("Prior changed")
        for arm in json.loads(paths["generated_manifest"].read_text()):
            if sha256(Path(arm["path"])) != arm["sha256"]:
                raise ValueError("Generated matrices changed")
        manifest = json.loads(paths["data_manifest"].read_text())
        for split in ("train", "validation", "test"):
            if (
                sha256(root / "data" / cohort / f"{split}.npz")
                != manifest["splits"][split]["sha256"]
            ):
                raise ValueError("Data changed before evaluation")
    for cohort in COHORTS:
        utility.evaluate(root / "data" / cohort, output / cohort / "utility")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "evaluate"])
    p.add_argument(
        "--bench-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--study-root", type=Path, required=True)
    args = p.parse_args()
    (select if args.stage == "select" else evaluate)(
        args.bench_root.resolve(), args.study_root.resolve()
    )


if __name__ == "__main__":
    main()
