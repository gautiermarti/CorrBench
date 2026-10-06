#!/usr/bin/env python3
"""Cache generalized-Fisher coordinates for an NPZ correlation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from corrbench.geometry import corr_to_gft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        raise FileExistsError(f"Choose a new cache file: {args.output}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    with np.load(args.input) as source:
        matrices = source["matrices"]
        passthrough = {
            name: source[name] for name in source.files if name != "matrices"
        }
    vectors = []
    minimum_input_eigenvalue = np.inf
    for start in range(0, matrices.shape[0], args.batch_size):
        batch = torch.from_numpy(matrices[start : start + args.batch_size]).to(
            args.device
        )
        minimum_input_eigenvalue = min(
            minimum_input_eigenvalue, float(torch.linalg.eigvalsh(batch)[:, 0].min())
        )
        vectors.append(corr_to_gft(batch).cpu().numpy().astype(np.float32))
        if start % (10 * args.batch_size) == 0:
            print(f"processed {start + batch.shape[0]}/{matrices.shape[0]}", flush=True)

    gft = np.concatenate(vectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gft=gft,
        minimum_input_eigenvalue=np.asarray(minimum_input_eigenvalue),
        **passthrough,
    )
    print(
        f"wrote {gft.shape} to {args.output}; "
        f"mean={gft.mean():.6f}, std={gft.std():.6f}"
    )


if __name__ == "__main__":
    main()
