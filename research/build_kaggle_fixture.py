"""Build and exercise an entirely synthetic, offline Kaggle readiness bundle."""

import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(groups, n, seed):
    """Independent synthetic groups, each containing the three declared regimes."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(3), groups)
    matrices = []
    # Independent blocks, common market factor, heterogeneous idiosyncratic risk.
    # No observed financial record is used to construct this systems-test fixture.
    blocks = np.arange(n) // max(n // 4, 1)
    for label in labels:
        loading = np.zeros((n, 6))
        loading[:, 0] = rng.uniform(0.15, 0.4, n) + 0.24 * label
        loading[np.arange(n), 1 + blocks.clip(max=3)] = rng.uniform(0.15, 0.6, n)
        loading[:, -1] = rng.normal(0, 0.12, n)
        covariance = loading @ loading.T + np.diag(rng.uniform(0.25, 0.8, n))
        scale = np.sqrt(np.diag(covariance))
        matrices.append(covariance / scale[:, None] / scale[None, :])
    return np.asarray(matrices), labels


def frame(matrices, labels, prefix, metric):
    n = matrices.shape[-1]
    indices = np.tril_indices(n, -1)
    result = pd.DataFrame(
        matrices[:, indices[0], indices[1]], columns=metric.columns(n)
    )
    result.insert(0, "id", [f"{prefix}_{i:06d}" for i in range(len(labels))])
    result["condition"] = labels
    return result


def notebook(path, title, source):
    document = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "overview",
                "metadata": {},
                "source": [title],
            },
            {
                "cell_type": "code",
                "id": "implementation",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(document, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--train-groups", type=int, default=100)
    parser.add_argument("--test-groups", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve earlier evidence")
    public = args.output / "public"
    organizer = args.output / "organizer_only"
    public.mkdir(parents=True)
    organizer.mkdir()
    metric = load_module("fixture_metric", ROOT / "competition/metric.py")
    starter = load_module("fixture_starter", ROOT / "competition/starter_replay.py")
    train, train_labels = synthetic(args.train_groups, args.n, 1103)
    hidden, hidden_labels = synthetic(args.test_groups, args.n, 8107)
    metric.CONFIG = metric.fit_configuration(train)
    train_frame = frame(train, train_labels, "train", metric)
    solution = frame(hidden, hidden_labels, "case", metric)
    public_groups = max(20, int(0.2 * args.test_groups))
    if args.test_groups - public_groups < 20:
        raise ValueError("Both subsets need at least 20 groups")
    group = np.repeat(np.arange(args.test_groups), 3)
    solution["Usage"] = np.where(group < public_groups, "Public", "Private")
    cases = solution[["id", "condition"]]
    names = metric.columns(args.n)
    train_frame.to_csv(public / "train.csv", index=False)
    cases.to_csv(public / "test.csv", index=False)
    solution.to_csv(organizer / "solution.csv", index=False)
    replay = starter.replay(
        pd.read_csv(public / "train.csv"), pd.read_csv(public / "test.csv")
    )
    replay.to_csv(public / "sample_submission.csv", index=False)
    configured_source = (
        (ROOT / "competition/metric.py")
        .read_text()
        .replace("CONFIG = None", "CONFIG = " + repr(metric.CONFIG), 1)
    )
    (public / "metric.py").write_text(configured_source)
    notebook(
        public / "metric.ipynb",
        "# Candidate distribution metric — synthetic local fixture\n"
        "Not a launched competition or approved final metric.",
        configured_source,
    )
    starter_source = (
        (ROOT / "competition/starter_replay.py")
        .read_text()
        .split("\ndef main():", 1)[0]
    )
    starter_source += (
        "\n# Point DATA to the competition input directory when hosted.\n"
        "DATA = Path('.')\n"
        "train = pd.read_csv(DATA / 'train.csv')\n"
        "cases = pd.read_csv(DATA / 'test.csv')\n"
        "replay(train, cases).to_csv('submission.csv', index=False)\n"
    )
    notebook(
        public / "starter.ipynb",
        "# Conditional replay starter\n"
        "Reads public training data and requested condition labels only.",
        starter_source,
    )
    solution = pd.read_csv(organizer / "solution.csv")
    exact = solution[["id", *names]].copy()
    conditional_mean = replay.copy()
    for label in metric.CONFIG["classes"]:
        conditional_mean.loc[cases.condition == label, names] = (
            train_frame.loc[train_frame.condition == label, names].mean().to_numpy()
        )
    identity = replay.copy()
    identity[names] = 0.0
    wrong = replay.copy()
    # Consecutive cases have conditions 0,1,2; roll by one mixes these regimes.
    wrong[names] = np.roll(replay[names].to_numpy(), 1, axis=0)
    candidates = {
        "perfect_reference_organizer_check": exact,
        "conditional_replay": pd.read_csv(public / "sample_submission.csv"),
        "conditional_training_mean": conditional_mean,
        "identity": identity,
        "wrong_condition_replay": wrong,
    }
    scores, runtimes = {}, {}
    for name, submission in candidates.items():
        scores[name], runtimes[name] = {}, {}
        for usage in ["Public", "Private"]:
            mask = solution["Usage"] == usage
            start = time.perf_counter()
            scores[name][usage] = metric.score(solution[mask], submission[mask], "id")
            runtimes[name][usage] = time.perf_counter() - start
    assert scores["perfect_reference_organizer_check"] == {
        "Public": 0.0,
        "Private": 0.0,
    }
    assert all(
        scores["conditional_replay"][part] < scores["identity"][part]
        for part in ["Public", "Private"]
    )
    assert all(
        scores["conditional_replay"][part] < scores["wrong_condition_replay"][part]
        for part in ["Public", "Private"]
    )
    # Exercise ID alignment and distributional invariance after actual CSV parsing.
    assert metric.score(solution, exact.sample(frac=1, random_state=99), "id") == 0.0
    permuted = exact.copy()
    for label in metric.CONFIG["classes"]:
        rows = np.flatnonzero(hidden_labels == label)
        permuted.loc[rows, names] = permuted.loc[rows[::-1], names].to_numpy()
    assert metric.score(solution, permuted, "id") == 0.0
    isolated = load_module("embedded_metric", public / "metric.py")
    assert isolated.score(solution, replay, "id") == metric.score(
        solution, replay, "id"
    )
    (organizer / "group_ids.json").write_text(json.dumps(group.tolist()) + "\n")
    files = {
        str(path.relative_to(args.output)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and "__pycache__" not in str(path)
    }
    record = {
        "status": "local_synthetic_readiness_test_only",
        "dimension": args.n,
        "train_cases": len(train),
        "evaluation_cases": len(hidden),
        "public_cases": int((solution.Usage == "Public").sum()),
        "private_cases": int((solution.Usage == "Private").sum()),
        "independent_test_groups": args.test_groups,
        "all_group_members_share_usage": True,
        "scores": scores,
        "seconds_per_score_call": runtimes,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "machine": platform.machine(),
        },
        "checks": {
            "csv_roundtrip_perfect_score_zero": True,
            "id_order_invariance": True,
            "within_condition_row_invariance": True,
            "public_private_subsets": True,
            "standalone_metric_parity": True,
        },
        "limitations": [
            "Synthetic systems test: these are not scientific model-comparison scores.",
            "Twenty-one features do not determine the full matrix distribution.",
            "Kaggle hosted scoring, data rights and protected holdout remain pending.",
            "No actual held-out research matrix is distributed in this fixture.",
            "Local runtime with concurrent GPU training; not Kaggle hardware.",
        ],
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                ROOT / "competition/metric.py",
                ROOT / "competition/starter_replay.py",
            ]
        },
        "files": files,
    }
    (args.output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in [
                    "status",
                    "dimension",
                    "evaluation_cases",
                    "scores",
                    "seconds_per_score_call",
                    "peak_process_rss_mib",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
