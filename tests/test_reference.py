import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from corrbench.reference import export_reference, read_reference_split, sha256


@pytest.fixture
def reference_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "market": "synthetic",
        "source": "/private/secret/returns.parquet",
        "source_sha256": "a" * 64,
        "sampling": {"matrix_dim": 3, "lookback": 12},
        "regime_definition": {"frozen_thresholds": [-0.2, 0.4]},
        "splits": {},
    }
    rng = np.random.default_rng(10)
    for i, split in enumerate(("train", "validation", "test")):
        # Estimate independent correlation matrices from known synthetic signals.
        matrices = np.stack(
            [np.corrcoef(rng.normal(size=(12, 3)), rowvar=False) for _ in range(3)]
        ).astype(np.float32)
        np.savez_compressed(
            source / f"{split}.npz",
            matrices=matrices,
            assets=np.array([["one", "two", "three"]] * 3),
            labels=np.array([0, 1, 2], dtype=np.uint8),
            original_labels=np.array([1, 1, 2], dtype=np.uint8),
            sharpes=np.array([-1.2, 0.3, 1.1], dtype=np.float32),
            end_dates=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
            + np.timedelta64(30 * i, "D"),
            minimum_eigenvalues=np.linalg.eigvalsh(matrices)[:, 0],
        )
        manifest["splits"][split] = {
            "sha256": sha256(source / f"{split}.npz"),
            "samples": 3,
            "private_note": "not for the release",
        }
    (source / "manifest.json").write_text(json.dumps(manifest))
    metadata = {
        "config_name": "synthetic_n3",
        "dataset_version": "test-v1",
        "domain": "synthetic",
        "protocol_id": "fixture-v1",
        "source_reference": "Generated test signals",
        "observation_definition": "Independent Gaussian draws",
        "builder_reference": "tests/test_reference.py",
        "split_description": "Separate synthetic draws; not a scientific result",
        "matrix_space": "real_correlation_pd",
        "rights_status": "pending",
    }
    return source, metadata


def test_complete_roundtrip_preserves_values_types_and_ids(reference_source, tmp_path):
    source, metadata = reference_source
    output = tmp_path / "release"
    manifest = export_reference(source, output, metadata, batch_size=2)
    for split in ("train", "validation", "test"):
        restored = read_reference_split(output, split)
        with np.load(source / f"{split}.npz") as original:
            assert set(restored) == set(original.files)
            for name in original.files:
                assert restored[name].dtype == original[name].dtype
                np.testing.assert_array_equal(restored[name], original[name])
        table = pq.read_table(output / manifest["splits"][split]["path"])
        assert table["source_row_index"].to_pylist() == [0, 1, 2]
        assert len(set(table["sample_id"].to_pylist())) == 3
        assert table["n_observations"].to_pylist() == [12, 12, 12]
    serialized = (output / "manifest.json").read_text()
    assert "/private/secret" not in serialized
    assert "private_note" not in serialized
    assert manifest["distribution_blocked"]
    card = (output / "README.md").read_text()
    front = yaml.safe_load(card.split("---", 2)[1])
    assert [s["split"] for s in front["configs"][0]["data_files"]] == [
        "train",
        "validation",
        "test",
    ]
    assert "license" not in front  # Never label pending financial data as MIT.
    assert "Public distribution is blocked" in card


def test_competition_export_never_reads_or_exposes_test(reference_source, tmp_path):
    source, metadata = reference_source
    original = json.loads((source / "manifest.json").read_text())
    hidden_hash = original["splits"]["test"]["sha256"]
    (source / "test.npz").unlink()
    output = tmp_path / "training"
    result = export_reference(source, output, metadata, purpose="competition_training")
    assert set(result["splits"]) == {"train", "validation"}
    assert not (output / "data/test.parquet").exists()
    assert hidden_hash not in (output / "manifest.json").read_text()


def test_hash_mismatch_leaves_no_partial_release(reference_source, tmp_path):
    source, metadata = reference_source
    with (source / "validation.npz").open("ab") as handle:
        handle.write(b"modified")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="hash mismatch"):
        export_reference(source, output, metadata)
    assert not output.exists()


@pytest.mark.parametrize(
    "problem", ["asymmetric", "indefinite", "nan", "unknown", "ids"]
)
def test_bad_scientific_inputs_are_rejected(reference_source, tmp_path, problem):
    source, metadata = reference_source
    with np.load(source / "train.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    if problem == "asymmetric":
        arrays["matrices"][0, 0, 1] = 0.95
    elif problem == "indefinite":
        arrays["matrices"][0] = [[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]
    elif problem == "nan":
        arrays["matrices"][0, 0, 1] = np.nan
    elif problem == "unknown":
        arrays["parent_ids"] = np.array([0, 1, 2])
    else:
        arrays["assets"][0, 0] = arrays["assets"][0, 1]
    np.savez_compressed(source / "train.npz", **arrays)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["splits"]["train"]["sha256"] = sha256(source / "train.npz")
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        export_reference(source, tmp_path / "release", metadata)
    assert not (tmp_path / "release").exists()


def test_immutable_release_and_reader_integrity(reference_source, tmp_path):
    source, metadata = reference_source
    output = tmp_path / "release"
    export_reference(source, output, metadata)
    with pytest.raises(FileExistsError):
        export_reference(source, output, metadata)
    with (output / "data/train.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Parquet hash mismatch"):
        read_reference_split(output, "train")


def test_source_rights_require_explicit_evidence(reference_source, tmp_path):
    source, metadata = reference_source
    metadata["rights_status"] = "cleared"
    with pytest.raises(ValueError, match="Cleared rights require"):
        export_reference(source, tmp_path / "release", metadata)


def test_restricted_source_is_blocked_and_discloses_evidence(
    reference_source, tmp_path
):
    source, metadata = reference_source
    metadata.update(rights_status="restricted", license="mit")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Restricted rights require"):
        export_reference(source, output, metadata)
    assert not output.exists()
    evidence = "The source permits competition participation only."
    metadata["rights_evidence"] = evidence
    manifest = export_reference(source, output, metadata)
    card = (output / "README.md").read_text()
    front = yaml.safe_load(card.split("---", 2)[1])
    assert manifest["distribution_blocked"] is True
    assert "license" not in front
    assert "blocked by the recorded source terms" in card
    assert evidence in card


def test_pending_source_evidence_is_visible(reference_source, tmp_path):
    source, metadata = reference_source
    evidence = "The exact historical source coverage remains under review."
    metadata["rights_evidence"] = evidence
    output = tmp_path / "release"
    export_reference(source, output, metadata)
    assert evidence in (output / "README.md").read_text()
