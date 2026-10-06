"""Adversarial submission contract tests, independent of the frozen paper score."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "competition" / "metric.py"
SPEC = importlib.util.spec_from_file_location("candidate_metric", PATH)
metric = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metric)


@pytest.fixture
def frames():
    n = 6
    labels = np.repeat(np.arange(3), 40)
    rho = 0.1 + 0.25 * labels + np.tile(np.linspace(0, 0.1, 40), 3)
    matrices = rho[:, None, None] * np.ones((len(labels), n, n))
    matrices += (1 - rho)[:, None, None] * np.eye(n)
    metric.CONFIG = metric.fit_configuration(matrices)
    names = metric.columns(n)
    solution = pd.DataFrame(matrices[:, *np.tril_indices(n, -1)], columns=names)
    solution.insert(0, "id", [f"case_{i}" for i in range(len(labels))])
    solution["condition"] = labels
    solution["Usage"] = np.tile(["Public", "Private"], len(labels) // 2)
    submission = solution[["id", *names]].copy()
    return solution, submission


def test_perfect_row_order_and_within_condition_distribution(frames):
    solution, submission = frames
    assert metric.score(solution, submission.sample(frac=1, random_state=9), "id") == 0
    names = metric.columns(6)
    for label in range(3):
        rows = np.flatnonzero(solution["condition"].to_numpy() == label)
        submission.loc[rows, names] = submission.loc[rows[::-1], names].to_numpy()
    assert metric.score(solution, submission, "id") == 0


def test_public_private_are_valid_scoring_subsets(frames):
    solution, submission = frames
    for usage in ["Public", "Private"]:
        mask = solution["Usage"] == usage
        assert metric.score(solution[mask], submission[mask], "id") == 0


def test_wrong_condition_and_collapse_penalized(frames):
    solution, submission = frames
    names = metric.columns(6)
    wrong = submission.copy()
    wrong[names] = np.roll(wrong[names].to_numpy(), 40, axis=0)
    assert metric.score(solution, wrong, "id") > 0.1
    collapsed = submission.copy()
    for label in range(3):
        mask = solution["condition"] == label
        collapsed.loc[mask, names] = collapsed.loc[mask, names].mean().to_numpy()
    assert metric.score(solution, collapsed, "id") > 0


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "infinity",
        "duplicate",
        "missing",
        "extra",
        "indefinite",
        "text",
        "duplicate_column",
    ],
)
def test_bad_submission_rejected(frames, kind):
    solution, submission = frames
    if kind in ["nan", "infinity"]:
        submission.loc[0, "c_0000"] = np.nan if kind == "nan" else np.inf
    elif kind == "duplicate":
        submission.loc[0, "id"] = submission.loc[1, "id"]
    elif kind == "missing":
        submission = submission.iloc[:-1]
    elif kind == "extra":
        submission["condition"] = 0
    elif kind == "indefinite":
        submission.loc[0, ["c_0000", "c_0001", "c_0002"]] = [0.9, 0.9, -0.9]
    elif kind == "text":
        submission["c_0000"] = submission["c_0000"].astype(object)
        submission.loc[0, "c_0000"] = "wrong"
    elif kind == "duplicate_column":
        submission = pd.concat([submission, submission[["c_0000"]]], axis=1)
    with pytest.raises(metric.ParticipantVisibleError):
        metric.score(solution, submission, "id")
