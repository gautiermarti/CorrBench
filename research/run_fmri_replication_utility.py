"""Wait for a frozen ABIDE family, audit its lineage, and run unchanged utility."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=["nodeflow", "anonymous_flow", "diffeocfm"], required=True
    )
    parser.add_argument("--execution-dir", type=Path)
    args = parser.parse_args()
    bench, flow = args.models_root / "corrbench", args.models_root / "corrflow"
    root = flow / "artifacts/abide_replication_20261005"
    data = bench / "artifacts/abide_ho_20261005"
    execution = args.execution_dir or root
    plan_path = execution / f"{args.family}_plan.json"
    ledger_path = execution / f"{args.family}_ledger.json"
    plan = read(plan_path)
    while True:
        ledger = read(ledger_path) if ledger_path.exists() else {"jobs": {}}
        states = [ledger["jobs"].get(j["name"], {}).get("status") for j in plan["jobs"]]
        if any(s in {"failed", "missing_artifact"} for s in states):
            raise RuntimeError(
                "A frozen generation job failed; explicit recovery needed"
            )
        if all(s == "complete" for s in states):
            break
        time.sleep(10)
    assert ledger["plan_sha256"] == sha(plan_path)
    for path, expected in plan["hashes"].items():
        assert sha(Path(path)) == expected, path
    for job in plan["jobs"]:
        for path, expected in job.get("hashes", {}).items():
            assert sha(Path(path)) == expected, path
        for path, expected in ledger["jobs"][job["name"]]["artifacts"].items():
            assert sha(Path(path)) == expected, path
    trainval = {split: sha(data / f"{split}.npz") for split in ["train", "validation"]}
    with np.load(data / "train.npz") as arrays:
        nodes = arrays["node_ids"]
    registry = [
        r
        for r in read(root / "registry.json")
        if Path(r["run"]).name.startswith(args.family + "_seed")
    ]
    runs, manifest, checked_samples = {}, [], {}
    for row in registry:
        run, path = Path(row["run"]), Path(row["path"])
        if str(run) not in runs:
            original_run = flow / f"artifacts/abide_ho_{args.family}_seed1701_20261005"
            np.testing.assert_array_equal(
                np.load(run / "selected_validation_indices.npy"),
                np.load(original_run / "selected_validation_indices.npy"),
            )
            history = read(run / "history.json")
            expected_epochs = 200 if args.family == "diffeocfm" else 20
            assert [h["epoch"] for h in history] == list(range(1, expected_epochs + 1))
            scored = [
                (
                    h["epoch"],
                    h.get(
                        "validation_feature_swd",
                        h.get("structural", {}).get("feature_swd"),
                    ),
                )
                for h in history
            ]
            scored = [(e, s) for e, s in scored if s is not None]
            best_epoch, best_score = min(scored, key=lambda x: x[1])
            checkpoint = torch.load(
                run / "best_structural.pt", map_location="cpu", weights_only=True
            )
            assert checkpoint["epoch"] == best_epoch
            if args.family != "diffeocfm":
                np.testing.assert_array_equal(checkpoint["node_ids"].numpy(), nodes)
                assert checkpoint["node_identity"] == (args.family == "nodeflow")
            else:
                initial = torch.load(
                    flow / "artifacts/abide_ho_diffeocfm_seed1701_20261005/prior.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                current = torch.load(
                    run / "prior.pt", map_location="cpu", weights_only=True
                )
                assert set(initial["state"]) == set(current["state"])
                for key in initial["state"]:
                    torch.testing.assert_close(
                        initial["state"][key],
                        current["state"][key],
                        atol=1e-12,
                        rtol=1e-12,
                    )
            provenance = read(run / "provenance.json")
            assert (
                provenance.get("source_sha256", provenance.get("source_hashes"))
                == trainval
            )
            runs[str(run)] = {
                "epochs": len(history),
                "selected_epoch": best_epoch,
                "validation_score": best_score,
                "training_seed": row["training_seed"],
            }
        metadata = read(path.with_suffix(".json"))
        for filename, expected in metadata["source_sha256"].items():
            assert sha(run / filename) == expected, filename
        assert metadata["generation_seed"] == row["generation_seed"]
        assert metadata["weights"] == row["weights"]
        expected_epoch = (
            runs[str(run)]["epochs"]
            if row["weights"] == "last"
            else runs[str(run)]["selected_epoch"]
        )
        assert metadata["epoch"] == expected_epoch
        with np.load(path) as arrays:
            assert arrays["matrices"].shape == (350, 111, 111)
            np.testing.assert_array_equal(arrays["labels"], np.repeat([0, 1], 175))
            assert int(arrays["seed"]) == row["generation_seed"]
            assert int(arrays["epoch"]) == expected_epoch
        arm = {
            "name": row["name"],
            "seed": row["generation_seed"],
            "kind": "generated",
            "path": str(path),
            "sha256": sha(path),
        }
        manifest.append(arm)
        if row["generation_seed"] == 2027:
            manifest.append(
                {
                    **arm,
                    "name": arm["name"] + "_synthetic_only",
                    "kind": "synthetic_only",
                }
            )
        checked_samples[row["name"]] = {
            "sha256": sha(path),
            "metadata_sha256": sha(path.with_suffix(".json")),
        }
    # The initial pilot did not classify from final DiffeoCFM weights alone.
    # Add that explicitly declared diagnostic from its already frozen draws.
    if args.family == "diffeocfm":
        path = flow / "artifacts/abide_augmentation_20261005/diffeocfm_last_2027.npz"
        manifest.append(
            {
                "name": "diffeocfm_last_train1701_draw2027_synthetic_only",
                "seed": 2027,
                "kind": "synthetic_only",
                "path": str(path),
                "sha256": sha(path),
            }
        )
    manifest_path = root / f"{args.family}_utility_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = bench / "artifacts/abide_replication_utility_20261005" / args.family
    if output.exists():
        raise FileExistsError(output)
    utility = bench / "research/fmri_utility.py"
    command = [
        sys.executable,
        str(utility),
        "select",
        "--data",
        str(data),
        "--output",
        str(output),
        "--generated-manifest",
        str(manifest_path),
    ]
    print(
        json.dumps(
            {
                "family": args.family,
                "stage": "validation_selection",
                "arms": len(manifest) + 6,
            }
        ),
        flush=True,
    )
    subprocess.run(command, check=True, cwd=bench)
    selection = read(output / "selection.json")
    original = read(bench / "artifacts/abide_complete_utility_20261005/selection.json")
    assert selection["arms"][:6] == original["arms"][:6]
    assert selection["source_sha256"] == trainval
    assert selection["code_sha256"] == sha(utility)
    assert not selection["test_opened"]
    for filename, expected in selection["model_sha256"].items():
        assert sha(output / filename) == expected
    audit = {
        "status": "passed",
        "time_utc": datetime.now(UTC).isoformat(),
        "scope": "Fixed-protocol training replication at a previously inspected site",
        "family": args.family,
        "selection_sha256": sha(output / "selection.json"),
        "plan_sha256": sha(plan_path),
        "ledger_sha256": sha(ledger_path),
        "source_sha256": trainval,
        "audit_code_sha256": sha(Path(__file__)),
        "runs": runs,
        "samples": checked_samples,
        "original_real_replay_selection_identical": True,
        "prior_matches_original": args.family == "diffeocfm",
    }
    (output / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps({"family": args.family, "stage": "held_out_evaluation"}), flush=True
    )
    subprocess.run(
        [
            sys.executable,
            str(utility),
            "evaluate",
            "--data",
            str(data),
            "--output",
            str(output),
        ],
        check=True,
        cwd=bench,
    )
    print(json.dumps({"family": args.family, "stage": "complete"}), flush=True)


if __name__ == "__main__":
    main()
