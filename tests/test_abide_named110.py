import importlib.util
from pathlib import Path

import numpy as np
import pytest

from corrbench.neuro import read_roi_series, scan_correlations


def named_reader():
    path = Path(__file__).parents[1] / "research/build_abide_named110.py"
    spec = importlib.util.spec_from_file_location("named_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_named_series


def test_external_label_selection_precedes_constant_check_and_estimation(tmp_path):
    read_named = named_reader()
    source = tmp_path / "series.1D"
    rng = np.random.default_rng(82)
    values = rng.normal(size=(70, 4))
    values[:, 1] += 2 * values[:, 0]
    values[:, 3] = 0
    np.savetxt(source, values, header="#10 #11 #12 #3455", comments="")
    with pytest.raises(ValueError, match="Constant"):
        read_roi_series(source)
    selected, bad = read_named(source, [10, 11, 12, 3455], [10, 11, 12])
    assert bad.size == 0
    np.testing.assert_array_equal(selected, values[:, :3])
    assert scan_correlations(selected)["minimum_target_eigenvalue"] > 0
    values[:, 3] = rng.normal(size=70)
    full = scan_correlations(values)
    reduced = scan_correlations(values[:, :3])
    assert abs(full["shrinkage"] - reduced["shrinkage"]) > 1e-5
    assert np.max(np.abs(full["matrices"][:3, :3] - reduced["matrices"])) > 1e-5
    values[:, 1] = 0
    np.savetxt(source, values, header="#10 #11 #12 #3455", comments="")
    _, bad = read_named(source, [10, 11, 12, 3455], [10, 11, 12])
    np.testing.assert_array_equal(bad, [11])
    with pytest.raises(ValueError, match="header differs"):
        read_named(source, [11, 10, 12, 3455], [10, 11, 12])
    with pytest.raises(ValueError, match="ordered subset"):
        read_named(source, [10, 11, 12, 3455], [11, 10, 12])
