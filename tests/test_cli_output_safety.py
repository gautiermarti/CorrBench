"""Protect existing scientific artifacts before attempting to read new inputs."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module,inputs,is_directory",
    [
        ("build_market_dataset", ["--config", "missing.yaml"], True),
        ("build_ordered_image_dataset", ["--input", "missing"], True),
        ("cache_gft", ["--input", "missing.npz"], False),
        (
            "evaluate_matrix_samples",
            ["--data", "missing", "--generated", "missing.npz"],
            False,
        ),
    ],
)
def test_existing_output_preserved(tmp_path, monkeypatch, module, inputs, is_directory):
    suffix = ".json" if module == "evaluate_matrix_samples" else ".npz"
    output = tmp_path / ("dataset" if is_directory else "output" + suffix)
    if is_directory:
        output.mkdir()
    artifact = output / "train.npz" if is_directory else output
    artifact.write_bytes(b"previous frozen scientific artifact")
    monkeypatch.setattr(sys, "argv", [module, *inputs, "--output", str(output)])
    with pytest.raises(FileExistsError):
        importlib.import_module(f"corrbench.cli.{module}").main()
    assert artifact.read_bytes() == b"previous frozen scientific artifact"


def test_evaluation_preserves_arrays_when_report_is_missing(tmp_path, monkeypatch):
    from corrbench.cli.evaluate_matrix_samples import main

    output = tmp_path / "evaluation.json"
    arrays = output.with_suffix(".npz")
    arrays.write_bytes(b"saved evaluation arrays")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--data",
            "missing",
            "--generated",
            "missing.npz",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        main()
    assert not output.exists()
    assert arrays.read_bytes() == b"saved evaluation arrays"


@pytest.mark.parametrize(
    "module,inputs,suffix",
    [
        ("cache_gft", ["--input", "missing.npz"], ".txt"),
        (
            "evaluate_matrix_samples",
            ["--data", "missing", "--generated", "missing.npz"],
            ".npz",
        ),
    ],
)
def test_output_suffix_cannot_cause_silent_numpy_rename_or_overwrite(
    tmp_path, monkeypatch, module, inputs, suffix
):
    output = tmp_path / ("output" + suffix)
    monkeypatch.setattr(sys, "argv", [module, *inputs, "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(f"corrbench.cli.{module}").main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
