"""Public-data-only conditional replay starter for the local Kaggle prototype."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def replay(train, cases, seed=2027):
    names = [name for name in train.columns if name.startswith("c_")]
    if not names:
        raise ValueError("No lower-triangle matrix columns in training data")
    rng = np.random.default_rng(seed)
    output = pd.DataFrame({"id": cases["id"]})
    values = np.empty((len(cases), len(names)))
    for label in sorted(cases["condition"].unique()):
        pool = train.loc[train["condition"] == label, names].to_numpy()
        if not len(pool):
            raise ValueError(f"No training examples for condition {label}")
        indices = np.flatnonzero(cases["condition"].to_numpy() == label)
        values[indices] = pool[rng.integers(0, len(pool), len(indices))]
    return pd.concat([output, pd.DataFrame(values, columns=names)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    train = pd.read_csv(args.data / "train.csv")
    cases = pd.read_csv(args.data / "test.csv")
    replay(train, cases, args.seed).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
