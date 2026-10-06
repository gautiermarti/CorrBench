#!/usr/bin/env python3
"""Cache normalized-Cholesky coordinates for the DiffeoCFM baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from corrbench.geometry import corr_to_normalized_cholesky


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with np.load(args.input) as source:
        matrices = source["matrices"]
        passthrough = {
            name: source[name] for name in source.files if name != "matrices"
        }
    coordinates = []
    for start in range(0, matrices.shape[0], args.batch_size):
        batch = torch.from_numpy(matrices[start : start + args.batch_size]).to(
            args.device
        )
        coordinates.append(
            corr_to_normalized_cholesky(batch).cpu().numpy().astype(np.float32)
        )
    coordinates = np.concatenate(coordinates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, coordinates=coordinates, **passthrough)
    print(
        f"wrote {coordinates.shape} to {args.output}; "
        f"mean={coordinates.mean():.6f}, std={coordinates.std():.6f}"
    )


if __name__ == "__main__":
    main()
