#!/usr/bin/env python3
"""Convert the original CorrGAN files into fixed stratified NPZ splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

FILENAME = re.compile(r"corr_(?P<index>\d+)_class_(?P<label>[012])\.npy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20200805)
    args = parser.parse_args()

    records = []
    for path in args.source.rglob("*.npy"):
        match = FILENAME.fullmatch(path.name)
        if match:
            records.append((int(match["index"]), int(match["label"]), path))
    records.sort()
    if len(records) != 18_000:
        raise ValueError(f"Expected 18,000 matrices, found {len(records)}.")

    labels = np.asarray([record[1] for record in records], dtype=np.uint8)
    all_indices = np.arange(len(records))
    train_indices, remaining = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=args.seed,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        remaining,
        test_size=0.50,
        random_state=args.seed + 1,
        stratify=labels[remaining],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        indices = np.sort(indices)
        matrices = np.stack(
            [np.load(records[index][2]).astype(np.float32) for index in indices]
        )
        split_labels = labels[indices]
        source_ids = np.asarray([records[index][0] for index in indices])
        output_path = args.output / f"{name}.npz"
        np.savez_compressed(
            output_path,
            matrices=matrices,
            labels=split_labels,
            source_ids=source_ids,
        )
        reports[name] = {
            "samples": int(indices.size),
            "label_counts": {
                str(label): int(np.sum(split_labels == label)) for label in range(3)
            },
            "minimum_eigenvalue": float(np.linalg.eigvalsh(matrices).min()),
            "sha256": sha256(output_path),
        }
        print(name, reports[name], flush=True)

    manifest = {
        "source": str(args.source),
        "seed": args.seed,
        "split": "fixed stratified 70/15/15",
        "ordering": "original CorrGAN Ward optimal-leaf ordering",
        "limitations": [
            "source dates and asset identities are unavailable",
            "random splits do not measure temporal generalization",
        ],
        "splits": reports,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
