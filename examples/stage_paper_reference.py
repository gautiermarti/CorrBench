"""Stage both financial releases and verify every original array, without upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from corrbench.reference import export_reference, read_reference_split, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=Path("configs/releases"))
    args = parser.parse_args()
    audit = {
        "scope": (
            "All frozen financial matrix arrays, not upstream prices or model training"
        ),
        "publication": "local staging only; financial data rights pending",
        "releases": {},
    }
    for name, dataset in (("taiwan", "taiwan_final"), ("japan", "japan_jpx")):
        metadata_path = args.metadata_root / f"{name}_paper.yaml"
        metadata = yaml.safe_load(metadata_path.read_text())
        output = args.output / metadata["config_name"]
        source = args.data_root / dataset
        print(f"Exporting {metadata['config_name']}", flush=True)
        manifest = export_reference(source, output, metadata)
        report = {
            "metadata_sha256": sha256(metadata_path),
            "release_manifest_sha256": sha256(output / "manifest.json"),
            "splits": {},
        }
        for split, info in manifest["splits"].items():
            restored = read_reference_split(output, split)
            with np.load(source / f"{split}.npz", allow_pickle=False) as original:
                if set(original.files) != set(restored):
                    raise AssertionError("Array names changed during export")
                for field in original.files:
                    if original[field].dtype != restored[field].dtype:
                        raise AssertionError(f"Dtype mismatch: {field}")
                    np.testing.assert_array_equal(original[field], restored[field])
            if sha256(source / f"{split}.npz") != info["input_npz_sha256"]:
                raise AssertionError("Source changed during round-trip verification")
            report["splits"][split] = {
                "samples": info["samples"],
                "arrays_equal": sorted(restored),
                "input_npz_sha256": info["input_npz_sha256"],
                "parquet_sha256": info["parquet_sha256"],
                "minimum_eigenvalue_float64": info["minimum_eigenvalue_float64"],
            }
            print(f"Verified {dataset}/{split}: {info['samples']} matrices", flush=True)
        audit["releases"][metadata["config_name"]] = report
    audit["total_matrices"] = sum(
        split["samples"]
        for release in audit["releases"].values()
        for split in release["splits"].values()
    )
    (args.output / "roundtrip_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(
        f"All {audit['total_matrices']} matrices and provenance arrays match",
        flush=True,
    )


if __name__ == "__main__":
    main()
