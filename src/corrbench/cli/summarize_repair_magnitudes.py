#!/usr/bin/env python3
"""Summarize interpretable nearest-correlation repair magnitudes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with np.load(path) as source:
            prepared = source["prepared"].astype(np.float64)
            repaired = source["matrices"].astype(np.float64)
        distance = np.linalg.norm(repaired - prepared, axis=(1, 2))
        base_norm = np.linalg.norm(prepared, axis=(1, 2))
        matrix_dim = prepared.shape[-1]
        rows.append(
            {
                "source": str(path),
                "mean_frobenius": float(distance.mean()),
                "mean_relative_frobenius": float(
                    np.mean(distance / np.maximum(base_norm, 1e-12))
                ),
                "mean_offdiagonal_rms": float(
                    np.mean(distance / np.sqrt(matrix_dim * (matrix_dim - 1)))
                ),
            }
        )

    report = {"runs": rows, "summary": {}}
    for metric in (
        "mean_frobenius",
        "mean_relative_frobenius",
        "mean_offdiagonal_rms",
    ):
        values = [row[metric] for row in rows]
        report["summary"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
