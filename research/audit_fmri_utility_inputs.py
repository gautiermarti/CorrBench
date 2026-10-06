"""Verify the complete ABIDE pilot artifact chain before held-out utility scoring."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from corrbench.neuro import sha256


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bench, flow = args.models_root / "corrbench", args.models_root / "corrflow"
    data = bench / "artifacts/abide_ho_20261005"
    generated = flow / "artifacts/abide_augmentation_20261005"
    utility = bench / "artifacts/abide_complete_utility_20261005"
    if (utility / "test_report.json").exists():
        raise ValueError("This audit must precede the held-out utility report")
    selection = read(utility / "selection.json")
    earlier_dir = bench / "artifacts/abide_real_replay_utility_20261005"
    earlier = read(earlier_dir / "selection.json")
    assert selection["arms"][:6] == earlier["arms"]
    assert len(selection["arms"]) == 35
    assert not selection["test_opened"]
    assert selection["code_sha256"] == sha256(bench / "research/fmri_utility.py")
    source_hashes = {s: sha256(data / f"{s}.npz") for s in ["train", "validation"]}
    assert selection["source_sha256"] == source_hashes
    for name, expected in selection["model_sha256"].items():
        assert sha256(utility / name) == expected, name
    with np.load(data / "train.npz") as a:
        nodes = a["node_ids"]
    with np.load(data / "validation.npz") as a:
        np.testing.assert_array_equal(nodes, a["node_ids"])
    assert len(nodes) == 111 and len(np.unique(nodes)) == 111
    methods = {
        "gaussian": "abide_ho_diffeocfm_seed1701_20261005",
        "diffeocfm": "abide_ho_diffeocfm_seed1701_20261005",
        "diffeocfm_last": "abide_ho_diffeocfm_seed1701_20261005",
        "nodeflow": "abide_ho_nodeflow_seed1701_20261005",
        "anonymous_flow": "abide_ho_anonymous_flow_seed1701_20261005",
    }
    runs = {}
    for directory in set(methods.values()):
        run = flow / "artifacts" / directory
        provenance = read(run / "provenance.json")
        assert provenance.get("source_sha256", provenance.get("source_hashes")) == (
            source_hashes
        )
        history = read(run / "history.json")
        evaluated = [
            {
                "epoch": r["epoch"],
                "validation_feature_swd": r.get(
                    "validation_feature_swd", r.get("structural", {}).get("feature_swd")
                ),
            }
            for r in history
            if "validation_feature_swd" in r or "structural" in r
        ]
        best = min(evaluated, key=lambda r: r["validation_feature_swd"])
        checkpoint = torch.load(
            run / "best_structural.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["epoch"] == best["epoch"]
        if "node_ids" in checkpoint:
            np.testing.assert_array_equal(nodes, checkpoint["node_ids"].numpy())
        runs[directory] = {
            "selected_epoch": best["epoch"],
            "validation_feature_swd": best["validation_feature_swd"],
            "complete": read(run / "complete.json"),
            "node_order": (
                "checkpoint node IDs match train and validation"
                if "node_ids" in checkpoint
                else "fixed input order; pinned trainer and dataset hashes"
            ),
        }
    ledger = read(generated / "generation_ledger.json")
    for filename, expected in ledger["source_hashes"].items():
        assert sha256(Path(filename)) == expected, filename
    manifest = read(generated / "utility_manifest.json")
    assert selection["arms"][6:] == [
        {**a, **{k: b[k] for k in ["C", "validation", "training_rows"]}}
        for a, b in zip(manifest, selection["arms"][6:], strict=True)
    ]
    samples = {}
    for method, directory in methods.items():
        for seed in range(2027, 2032):
            name = f"{method}_{seed}"
            path = generated / f"{name}.npz"
            job = ledger["jobs"][name]
            assert job["status"] == "complete"
            assert sha256(path) == job["sha256"]
            metadata_path = path.with_suffix(".json")
            assert sha256(metadata_path) == job["metadata_sha256"]
            metadata = read(metadata_path)
            for filename, expected in metadata["source_sha256"].items():
                assert sha256(flow / "artifacts" / directory / filename) == expected
            assert metadata["generation_seed"] == seed
            with np.load(path) as a:
                assert a["matrices"].shape == (350, 111, 111)
                np.testing.assert_array_equal(a["labels"], np.repeat([0, 1], 175))
                assert int(a["seed"]) == seed
                assert int(a["epoch"]) == metadata["epoch"]
            samples[name] = {
                "sha256": job["sha256"],
                "metadata_sha256": job["metadata_sha256"],
                "epoch": metadata["epoch"],
            }
    report = {
        "utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "test_utility_report_exists": False,
        "scope": "Input audit; test matrix construction already numerically audited",
        "source_sha256": source_hashes,
        "node_ids": nodes.tolist(),
        "generation_code_sha256": ledger["source_hashes"],
        "selection_sha256": sha256(utility / "selection.json"),
        "earlier_real_replay_selection_identical": True,
        "selected_classifier_count": len(selection["arms"]),
        "runs": runs,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "classifiers": 35, "samples": 25}))


if __name__ == "__main__":
    main()
