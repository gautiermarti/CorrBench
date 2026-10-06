#!/usr/bin/env python3
"""Build a Ward-ordered copy of a matrix dataset for image baselines."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from corrbench.ordering import ward_order


def order_one(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered, permutation = ward_order(matrix)
    return ordered.astype(np.float32), permutation.astype(np.uint16)


def copy_metadata(source: Any, excluded: set[str]) -> dict[str, np.ndarray]:
    return {name: source[name] for name in source.files if name not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new ordered dataset directory: {args.output}")
    if args.workers < 1:
        parser.error("--workers must be positive")

    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "input": str(args.input),
        "ordering": "Ward linkage with optimal leaves on condensed 1-C",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        with np.load(args.input / f"{split}.npz") as source:
            matrices = source["matrices"]
            metadata = copy_metadata(source, {"matrices", "permutations"})
        print(f"ordering {split}: {matrices.shape}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(order_one, matrices, chunksize=16))
        ordered = np.stack([result[0] for result in results])
        permutations = np.stack([result[1] for result in results])
        output = args.output / f"{split}.npz"
        np.savez_compressed(
            output,
            matrices=ordered,
            permutations=permutations,
            **metadata,
        )
        sampled = np.linspace(0, matrices.shape[0] - 1, 32, dtype=np.int64)
        reconstruction_error = max(
            float(
                np.max(
                    np.abs(
                        ordered[index]
                        - matrices[index][
                            np.ix_(permutations[index], permutations[index])
                        ]
                    )
                )
            )
            for index in sampled
        )
        report["splits"][split] = {
            "matrices": int(matrices.shape[0]),
            "dimension": int(matrices.shape[-1]),
            "sampled_permutation_error": reconstruction_error,
            "sampled_minimum_eigenvalue": float(
                np.linalg.eigvalsh(ordered[sampled]).min()
            ),
        }
        print(report["splits"][split], flush=True)
    (args.output / "ordering_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
