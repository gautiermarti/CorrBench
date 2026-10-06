"""Check all local ABIDE configurations with the independent HF datasets reader."""

import argparse
import hashlib
import json
from pathlib import Path

import datasets
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {
        "brain_abide_ho_n111": "artifacts/abide_ho_20261005",
        "brain_abide_ho_n110_same601": "artifacts/abide_named110_20261006/data/same601",
        "brain_abide_ho_n110_expanded762":
            "artifacts/abide_named110_20261006/data/expanded762",
    }
    report = {}
    for config, source in sources.items():
        manifest = json.loads((args.release / config / "manifest.json").read_text())
        loaded = datasets.load_dataset(
            str(args.release.resolve()), name=config,
            cache_dir=str(args.output / "cache"),
        )
        splits = {}
        for split, spec in manifest["splits"].items():
            source_file = args.root / source / f"{split}.npz"
            if sha(source_file) != spec["input_npz_sha256"]:
                raise ValueError("Frozen NPZ mismatch")
            rows = len(loaded[split])
            if rows != spec["samples"]:
                raise ValueError("HF row count mismatch")
            with np.load(source_file, allow_pickle=False) as original:
                for name, layout in spec["arrays"].items():
                    value = np.asarray(loaded[split][layout["column"]],
                                       dtype=layout["dtype"])
                    shape = (rows, *layout["sample_shape"])
                    value = value.reshape(shape)
                    expected = original[name]
                    if layout["static"]:
                        expected = np.broadcast_to(expected, shape)
                    if not np.array_equal(
                        value, expected, equal_nan=expected.dtype.kind in "fc"
                    ):
                        raise ValueError(f"HF values differ: {config}/{split}/{name}")
            if loaded[split]["source_row_index"] != list(range(rows)):
                raise ValueError("Source row order differs")
            splits[split] = {"rows": rows, "arrays_checked": len(spec["arrays"]),
                             "all_values_exact": True}
        report[config] = splits
    audit = {
        "scope": "Independent HF client local loading, not hosted delivery",
        "datasets_version": datasets.__version__, "configurations": report,
        "release_audit_sha256": sha(args.release / "audit.json"),
        "publicly_published": False,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
