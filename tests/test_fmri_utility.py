import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def utility_module():
    path = Path(__file__).parents[1] / "research/fmri_utility.py"
    spec = importlib.util.spec_from_file_location("fmri_utility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_runs_without_test_data_and_uses_training_only_scaler(tmp_path):
    utility = utility_module()
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(14)
    for split, count, start in [("train", 20, 0), ("validation", 8, 100)]:
        x = rng.normal(size=(count, 30, 4))
        matrices = np.stack([np.corrcoef(row, rowvar=False) for row in x])
        np.savez(
            data / f"{split}.npz",
            matrices=matrices,
            labels=np.arange(count) % 2,
            subject_ids=np.arange(start, start + count),
        )
    output = tmp_path / "selection"
    utility.select(data, output, None)
    record = json.loads((output / "selection.json").read_text())
    assert not record["test_opened"]
    assert len(record["arms"]) == 6
    assert record["augmentation_per_class"] == 10
    assert record["arms"][0]["training_rows"] == 20
    assert all(a["training_rows"] == 40 for a in record["arms"][1:])
    import joblib

    scaler = joblib.load(output / "feature_scaler.joblib")
    with np.load(data / "train.npz") as a:
        np.testing.assert_allclose(
            scaler.mean_, utility.features(a["matrices"]).mean(0)
        )
    with pytest.raises(FileNotFoundError):
        utility.evaluate(data, output)


def test_paired_bootstrap_and_invalid_matrix_guard():
    utility = utility_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.8, 0.7, 0.8, 0.9])
    result = utility.stratified_bootstrap_difference(
        y, probabilities, probabilities.copy(), draws=20
    )
    assert result == {"roc_auc": [0.0, 0.0], "balanced_accuracy": [0.0, 0.0]}
    invalid = np.array([[[1.0, 1.2], [1.2, 1.0]]])
    with pytest.raises(ValueError, match="PD"):
        utility.features(invalid)
