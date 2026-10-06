import json

import numpy as np
import pandas as pd
import pytest

from corrbench.neuro import (
    eligible_subjects,
    read_roi_series,
    scan_correlations,
    subject_split,
)


def test_subject_and_site_groups_and_quality_rules():
    rows = []
    for i, site in enumerate(["PITT", "UM_1", "UM_2", "NYU", "YALE"]):
        rows.append(
            {
                "SUB_ID": i,
                "FILE_ID": f"{site}_0000{i}",
                "SITE_ID": site,
                "DX_GROUP": 1 if i % 2 else 2,
                "qc_rater_1": "OK",
                "qc_anat_rater_2": "maybe",
                "qc_func_rater_2": "OK",
                "qc_anat_rater_3": "OK",
                "qc_func_rater_3": "OK",
                "func_mean_fd": 0.25 if i == 4 else 0.1,
            }
        )
    frame = pd.DataFrame(rows)
    kept = eligible_subjects(frame)
    assert list(kept.split) == ["train", "validation", "validation", "test"]
    assert list(kept.label) == [0, 1, 0, 1]
    assert subject_split("UM_1") == subject_split("UM_2")
    with pytest.raises(ValueError, match="unique"):
        eligible_subjects(pd.concat([frame, frame.iloc[:1]]))


def test_regularization_handles_short_scans_and_is_permutation_consistent():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(50, 80))
    result = scan_correlations(x)
    assert result["empirical_rank"] <= 49
    assert result["minimum_target_eigenvalue"] > 0
    assert 0 < result["shrinkage"] <= 1
    np.testing.assert_allclose(np.diag(result["matrices"]), 1)
    p = rng.permutation(80)
    permuted = scan_correlations(x[:, p])
    np.testing.assert_allclose(
        permuted["matrices"], result["matrices"][np.ix_(p, p)], atol=1e-14
    )
    expected = np.corrcoef(x, rowvar=False)
    np.testing.assert_allclose(result["empirical_matrices"], expected, atol=1e-14)


def test_roi_header_is_preserved_and_bad_observations_rejected(tmp_path):
    path = tmp_path / "series.1D"
    x = np.random.default_rng(4).normal(size=(50, 3))
    np.savetxt(path, x, header="#10\t#101\t#102", comments="")
    values, ids = read_roi_series(path)
    np.testing.assert_array_equal(values, x)
    np.testing.assert_array_equal(ids, [10, 101, 102])
    x[:, 0] = 0
    np.savetxt(path, x, header="#10\t#101\t#102", comments="")
    with pytest.raises(ValueError, match="Constant"):
        read_roi_series(path)


def test_builder_preserves_subjects_and_rejects_changed_source(tmp_path, monkeypatch):
    from corrbench.cli import build_abide_dataset as builder
    from corrbench.neuro import ABIDE_DERIVATIVE, sha256

    series = tmp_path / "series"
    series.mkdir()
    rows, files = [], []
    rng = np.random.default_rng(50)
    for i, site in enumerate(["PITT", "PITT", "UM_1", "UM_2", "NYU", "NYU"]):
        file_id = f"{site}_0000{i}"
        row = {
            "SUB_ID": i,
            "FILE_ID": file_id,
            "SITE_ID": site,
            "DX_GROUP": 1 if i % 2 else 2,
            "qc_rater_1": "OK",
            "qc_anat_rater_2": "OK",
            "qc_func_rater_2": "OK",
            "qc_anat_rater_3": "OK",
            "qc_func_rater_3": "OK",
            "func_mean_fd": 0.1,
            "AGE_AT_SCAN": 20,
            "SEX": 1,
        }
        rows.append(row)
        path = series / f"{file_id}_rois_ho.1D"
        np.savetxt(
            path,
            rng.normal(size=(50, 111)),
            header="\t".join(f"#{j}" for j in range(111)),
            comments="",
        )
        files.append(
            {
                "subject_id": i,
                "site": site,
                "split": subject_split(site),
                "status": "available",
                "path": path.name,
                "sha256": sha256(path),
                "url": "synthetic-test-fixture",
            }
        )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    digest = sha256(metadata)
    monkeypatch.setattr(builder, "ABIDE_PHENOTYPE_SHA256", digest)
    (series / "download_manifest.json").write_text(
        json.dumps(
            {
                "metadata_sha256": digest,
                "derivative": ABIDE_DERIVATIVE,
                "files": files,
            }
        )
    )
    output = tmp_path / "dataset"
    manifest = builder.build_dataset(metadata, series, output)
    assert not manifest["excluded"]
    groups = []
    for split in ["train", "validation", "test"]:
        with np.load(output / f"{split}.npz") as a:
            assert a["matrices"].shape == (2, 111, 111)
            assert a["empirical_rank"].max() <= 49
            assert a["minimum_target_eigenvalue"].min() > 0
            groups.append(set(a["subject_ids"].tolist()))
    assert len(set.union(*groups)) == 6
    changed = series / files[-1]["path"]
    changed.write_text(changed.read_text() + "\n")
    with pytest.raises(RuntimeError, match="Source file changed"):
        builder.build_dataset(metadata, series, tmp_path / "changed")
