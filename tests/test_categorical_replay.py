import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_replay_preserves_classes_and_uses_only_training_rows():
    path = Path(__file__).parents[1] / "research/sample_categorical_replay.py"
    spec = importlib.util.spec_from_file_location("replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = np.array([0, 0, 1, 1, 2])
    requested = np.array([2, 0, 1, 2, 1, 0] * 20)
    drawn = module.replay_indices(labels, requested, seed=19)
    np.testing.assert_array_equal(labels[drawn], requested)
    np.testing.assert_array_equal(
        drawn, module.replay_indices(labels, requested, seed=19)
    )
    assert not np.array_equal(drawn, module.replay_indices(labels, requested, seed=20))
    with pytest.raises(ValueError, match="No training rows"):
        module.replay_indices(labels, np.array([3]), seed=19)
