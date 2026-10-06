import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "research/audit_market_windows.py"
SPEC = importlib.util.spec_from_file_location("market_windows", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_actual_sessions_and_inclusive_boundaries():
    # Deliberate calendar gaps: purging must use observed sessions.
    days = np.array([1, 2, 5, 8, 9, 12, 13, 14, 18, 20, 21, 22, 25, 28, 29])
    dates = np.datetime64("2020-01-01") + days
    ends = {
        "train": dates[[2, 2, 3, 4]],
        "validation": dates[[5, 6, 7, 8]],
        "test": dates[[9, 10, 11, 12, 13, 14]],
    }
    views, report = MODULE.window_views(dates, ends, 3)
    np.testing.assert_array_equal(views["boundary_purged"]["validation"], [2, 3])
    np.testing.assert_array_equal(views["boundary_purged"]["test"], [2, 3, 4, 5])
    np.testing.assert_array_equal(views["disjoint_blocks"]["train"], [0, 1])
    np.testing.assert_array_equal(views["disjoint_blocks"]["test"], [2, 5])
    assert report["original"]["shared_return_sessions"]["train__validation"] == 2
    assert report["disjoint_blocks"]["splits"]["train"]["parent_windows"] == 1
    assert report["disjoint_blocks"]["splits"]["train"]["matrices"] == 2


def test_short_validation_is_explicitly_empty():
    dates = np.arange(12).astype("datetime64[D]")
    ends = {
        "train": dates[[3, 4]],
        "validation": dates[[5, 6]],
        "test": dates[[7, 8, 9, 10, 11]],
    }
    views, report = MODULE.window_views(dates, ends, 4)
    assert not len(views["boundary_purged"]["validation"])
    assert report["boundary_purged"]["all_splits_nonempty"] is False
    assert (
        report["boundary_purged"]["splits"]["validation"]["first_return_date"] is None
    )


@pytest.mark.parametrize("fault", ["missing_end", "early_window", "overlap_ends"])
def test_invalid_window_provenance_rejected(fault):
    dates = np.arange(20).astype("datetime64[D]")
    ends = {
        "train": dates[[4, 5]],
        "validation": dates[[10, 11]],
        "test": dates[[17, 18]],
    }
    if fault == "missing_end":
        ends["test"] = np.array([np.datetime64("2030-01-01")])
    elif fault == "early_window":
        ends["train"] = dates[[0, 1]]
    else:
        ends["validation"] = dates[[5, 11]]
    with pytest.raises(ValueError):
        MODULE.window_views(dates, ends, 3)
