#!/usr/bin/env python3
"""Measure eigenvalue conditioning across a nested matrix-size benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    dims = sorted(
        int(path.name.removeprefix("n"))
        for path in args.data.glob("n*")
        if path.is_dir() and path.name.removeprefix("n").isdigit()
    )
    if not dims:
        raise ValueError(f"No n<size> directories found under {args.data}")

    report = {
        "data": str(args.data),
        "device": args.device,
        "matrix_dims": dims,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        split_report = {}
        for dim in dims:
            with np.load(args.data / f"n{dim}" / f"{split}.npz") as source:
                matrices = source["matrices"]
            minimum_eigenvalues = []
            maximum_eigenvalues = []
            for start in range(0, matrices.shape[0], args.batch_size):
                batch = torch.from_numpy(matrices[start : start + args.batch_size]).to(
                    args.device
                )
                eigenvalues = torch.linalg.eigvalsh(batch)
                minimum_eigenvalues.append(eigenvalues[:, 0].cpu().numpy())
                maximum_eigenvalues.append(eigenvalues[:, -1].cpu().numpy())
            minimum = np.concatenate(minimum_eigenvalues).astype(np.float64)
            maximum = np.concatenate(maximum_eigenvalues).astype(np.float64)
            condition = maximum / minimum
            split_report[str(dim)] = {
                "samples": int(matrices.shape[0]),
                "minimum_eigenvalue": float(minimum.min()),
                "condition_number": summarize(condition),
            }
            print(
                f"{split} n={dim}: median={np.median(condition):.2f} "
                f"p99={np.quantile(condition, 0.99):.2f} "
                f"max={condition.max():.2f}",
                flush=True,
            )
        report["splits"][split] = split_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
