"""Training-only categorical replay with independent reference-case selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def replay_indices(train_labels, requested_labels, seed):
    rng = np.random.default_rng(seed)
    result = np.empty(len(requested_labels), dtype=np.int64)
    for label in np.unique(requested_labels):
        candidates = np.flatnonzero(train_labels == label)
        if not len(candidates):
            raise ValueError(f"No training rows for class {label}")
        mask = requested_labels == label
        result[mask] = rng.choice(candidates, int(mask.sum()), replace=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=44301)
    p.add_argument("--selection-seed", type=int, default=44301)
    args = p.parse_args()
    if args.output.suffix != ".npz" or args.per_class < 1:
        p.error("Require .npz output and positive per-class count")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    with np.load(args.data / f"{args.split}.npz") as a:
        labels = a["labels"]
    rng = np.random.default_rng(args.selection_seed)
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == c), args.per_class, replace=False)
            for c in np.unique(labels)
        ]
    )
    labels = labels[selected]
    train_path = args.data / "train.npz"
    with np.load(train_path) as a:
        train, train_labels = a["matrices"], a["labels"]
    started = time.perf_counter()
    drawn = replay_indices(train_labels, labels, args.seed)
    matrices = train[drawn]
    seconds = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrices=matrices,
        labels=labels,
        selected_test_indices=selected,
        replay_train_indices=drawn,
        split=args.split,
        sampling_seconds=seconds,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "training-only categorical replay with replacement",
                "seed": args.seed,
                "selection_seed": args.selection_seed,
                "split": args.split,
                "samples": len(labels),
                "unique_training_rows": len(np.unique(drawn)),
                "sampling_seconds_excluding_io": seconds,
                "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(labels)} replay samples to {args.output}")


if __name__ == "__main__":
    main()
