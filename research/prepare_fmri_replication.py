"""Freeze additional training-seed jobs for the unchanged ABIDE pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bench, flow = args.models_root / "corrbench", args.models_root / "corrflow"
    data = bench / "artifacts/abide_ho_20261005"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source_paths = [
        *[data / f"{split}.npz" for split in ["train", "validation", "test"]],
        bench / "src/corrbench/geometry.py",
        bench / "src/corrbench/metrics.py",
        bench / "research/run_frozen_jobs.py",
        bench / "research/fmri_utility.py",
        bench / "docs/fmri_replication_protocol.md",
        flow / "src/corrflow/cli/train_neuro_flow.py",
        flow / "src/corrflow/cli/train_diffeocfm.py",
        flow / "src/corrflow/cli/sample_neuro_augmentation.py",
        flow / "src/corrflow/models/node_conditioned.py",
        flow / "src/corrflow/models/equivariant.py",
        flow / "src/corrflow/sampling.py",
        flow / "src/corrflow/flow.py",
        flow / "src/corrflow/diffeocfm.py",
        Path(__file__),
    ]
    hashes = {str(path): sha(path) for path in source_paths}
    registry = []
    for family in ["nodeflow", "anonymous_flow", "diffeocfm"]:
        plan = {
            "protocol": "fmri_replication_protocol_20261005",
            "previous_test_inspection": True,
            "hashes": hashes,
            "environment": {
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": "",
            },
            "jobs": [],
        }
        config_path = (
            flow
            / "configs"
            / (
                "diffeocfm_abide_ho_pilot.yaml"
                if family == "diffeocfm"
                else "corrflow_abide_ho_pilot.yaml"
            )
        )
        plan["hashes"] = {**hashes, str(config_path): sha(config_path)}
        for seed in range(1702, 1706):
            name = f"{family}_seed{seed}"
            run = args.output / name
            config = yaml.safe_load(config_path.read_text())
            config["training"]["seed"] = seed
            frozen_config = args.output / f"{name}_config.yaml"
            frozen_config.write_text(yaml.safe_dump(config))
            module = "train_diffeocfm" if family == "diffeocfm" else "train_neuro_flow"
            command = [
                str(args.python),
                "-m",
                f"corrflow.cli.{module}",
                "--data",
                str(data),
                "--config",
                str(frozen_config),
                "--output",
                str(run),
                "--device",
                "cpu",
            ]
            if family == "anonymous_flow":
                command.append("--anonymous")
            plan["jobs"].append(
                {
                    "name": name + "_train",
                    "cwd": str(flow),
                    "command": command,
                    "hashes": {str(frozen_config): sha(frozen_config)},
                    "artifacts": [
                        str(run / filename)
                        for filename in [
                            "complete.json",
                            "history.json",
                            "best_structural.pt",
                            "last.pt",
                            "provenance.json",
                            "selected_validation_indices.npy",
                        ]
                    ],
                }
            )
            methods = [family] + (["diffeocfm_last"] if family == "diffeocfm" else [])
            for method in methods:
                weights = "last" if method.endswith("_last") else "best_structural"
                for draw in range(2027, 2032):
                    sample_name = f"{method}_train{seed}_draw{draw}"
                    path = args.output / "generated" / f"{sample_name}.npz"
                    plan["jobs"].append(
                        {
                            "name": sample_name,
                            "cwd": str(flow),
                            "command": [
                                str(args.python),
                                "-m",
                                "corrflow.cli.sample_neuro_augmentation",
                                "--run",
                                str(run),
                                "--family",
                                "diffeocfm" if family == "diffeocfm" else "gft",
                                "--output",
                                str(path),
                                "--per-class",
                                "175",
                                "--seed",
                                str(draw),
                                "--device",
                                "cpu",
                                "--weights",
                                weights,
                            ],
                            "artifacts": [str(path), str(path.with_suffix(".json"))],
                        }
                    )
                    registry.append(
                        {
                            "name": sample_name,
                            "method": method,
                            "training_seed": seed,
                            "generation_seed": draw,
                            "path": str(path),
                            "run": str(run),
                            "weights": weights,
                        }
                    )
        (args.output / f"{family}_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n"
        )
    (args.output / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(
        json.dumps(
            {"additional_training_runs": 12, "additional_sample_batches": len(registry)}
        )
    )


if __name__ == "__main__":
    main()
