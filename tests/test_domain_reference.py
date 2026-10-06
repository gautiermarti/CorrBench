import json

import numpy as np
import pytest

from corrbench.domain_reference import (
    export_domain_reference,
    read_domain_reference_split,
)
from corrbench.reference import sha256


@pytest.fixture
def brain_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {"private_note": "/private/hidden", "splits": {}}
    for split in ["train", "validation", "test"]:
        np.savez_compressed(
            source / f"{split}.npz",
            matrices=np.tile(np.eye(3), (2, 1, 1)),
            empirical_matrices=np.tile(np.ones((3, 3)), (2, 1, 1)),
            labels=np.array([0, 1], dtype=np.int16),
            node_ids=np.array([17, 3, 92], dtype=np.int32),
            subject_ids=np.array([1001, 1002]),
            site_ids=np.array(["a", "b"]),
        )
        manifest["splits"][split] = {
            "subjects": 2,
            "sha256": sha256(source / f"{split}.npz"),
        }
    (source / "manifest.json").write_text(json.dumps(manifest))
    metadata = {
        "config_name": "test_brain",
        "dataset_version": "test",
        "domain": "neuroscience",
        "protocol_id": "synthetic-test",
        "source_reference": "Synthetic",
        "observation_definition": "Synthetic",
        "builder_reference": "Test fixture",
        "split_description": "Test only",
        "matrix_space": "real_correlation_pd",
        "rights_status": "pending",
        "independent_units": "Synthetic test rows",
        "limitations": "Not real subjects",
    }
    return source, metadata


def test_static_atlas_and_singular_diagnostic_roundtrip(brain_source, tmp_path):
    source, metadata = brain_source
    output = tmp_path / "release"
    export_domain_reference(source, output, metadata, adapter="abide_pcp")
    for split in ["train", "validation", "test"]:
        restored = read_domain_reference_split(output, split)
        with np.load(source / f"{split}.npz") as original:
            assert set(restored) == set(original.files)
            for key in original.files:
                assert restored[key].dtype == original[key].dtype
                np.testing.assert_array_equal(restored[key], original[key])
    assert "/private/hidden" not in (output / "manifest.json").read_text()


def test_domain_training_export_does_not_read_test(brain_source, tmp_path):
    source, metadata = brain_source
    (source / "test.npz").unlink()
    output = tmp_path / "release"
    result = export_domain_reference(
        source, output, metadata, adapter="abide_pcp", purpose="competition_training"
    )
    assert set(result["splits"]) == {"train", "validation"}
    assert result["distribution_blocked"]


def test_custom_license_identifier_is_accepted_by_hub_schema(brain_source, tmp_path):
    """A locally valid card must not fail the Hub's custom-license slug rule."""
    import yaml

    source, metadata = brain_source
    metadata.update(
        rights_status="cleared",
        rights_evidence="Synthetic fixture created for this test",
        license="other",
        license_name="Human readable license title",
        license_link="https://example.org/license",
    )
    invalid = tmp_path / "invalid_license"
    with pytest.raises(ValueError, match="license_name"):
        export_domain_reference(source, invalid, metadata, adapter="abide_pcp")
    assert not invalid.exists()
    metadata["license_name"] = "synthetic-test-1.0"
    output = tmp_path / "valid_license"
    export_domain_reference(source, output, metadata, adapter="abide_pcp")
    front = yaml.safe_load((output / "README.md").read_text().split("---")[1])
    assert front["license_name"] == "synthetic-test-1.0"
    assert front["license_link"] == "https://example.org/license"


def test_unknown_field_and_bad_atlas_are_rejected_atomically(brain_source, tmp_path):
    source, metadata = brain_source
    with np.load(source / "train.npz") as a:
        arrays = {k: a[k] for k in a.files}
    for problem in ["unknown", "duplicate_id"]:
        current = dict(arrays)
        if problem == "unknown":
            current["unrecorded_parent"] = np.array([1, 2])
        else:
            current["node_ids"] = np.array([1, 1, 2])
        np.savez_compressed(source / "train.npz", **current)
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["splits"]["train"]["sha256"] = sha256(source / "train.npz")
        (source / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            export_domain_reference(
                source, tmp_path / problem, metadata, adapter="abide_pcp"
            )
        assert not (tmp_path / problem).exists()
