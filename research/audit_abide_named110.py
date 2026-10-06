"""Independently check reconstruction and reproduce all selected validation scores."""

import argparse
import io
import json
from pathlib import Path

import fmri_utility as utility
import joblib
import numpy as np
from sklearn.covariance import oas
from sklearn.preprocessing import StandardScaler

from corrbench.neuro import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--study-root", type=Path, required=True)
    args = p.parse_args()
    root, bench = args.study_root.resolve(), args.bench_root.resolve()
    destination = root / "input_selection_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    registration_path = root / "controls/selection_registration.json"
    registration = json.loads(registration_path.read_text())
    construction = json.loads((root / "data/construction_audit.json").read_text())
    download_path = (
        bench / "artifacts/abide_source_20261005/rois_ho/download_manifest.json"
    )
    download = json.loads(download_path.read_text())
    records = {r["subject_id"]: r for r in download["files"]}
    datasets, max_error, max_shrinkage_error = {}, 0.0, 0.0
    seen = set()
    for cohort in ("same601", "expanded762"):
        datasets[cohort] = {}
        for split in ("train", "validation", "test"):
            path = root / "data" / cohort / f"{split}.npz"
            if (
                sha256(path)
                != construction["cohorts"][cohort]["splits"][split]["sha256"]
            ):
                raise ValueError("Dataset changed")
            with np.load(path) as arrays:
                data = {k: arrays[k] for k in arrays.files}
            datasets[cohort][split] = data
            if cohort != "expanded762":
                continue
            for i, subject in enumerate(data["subject_ids"]):
                record = records[int(subject)]
                if record["split"] != split or subject in seen:
                    raise ValueError("Subject mismatch or duplicate")
                seen.add(int(subject))
                source = download_path.parent / record["path"]
                if sha256(source) != record["sha256"]:
                    raise ValueError("Source changed")
                text = source.read_text()
                ids = np.array([int(x[1:]) for x in text.splitlines()[0].split()])
                raw = np.loadtxt(io.StringIO(text), comments="#")
                keep = ids != 3455
                np.testing.assert_array_equal(ids[keep], data["node_ids"])
                x = raw[:, keep]
                z = (x - x.mean(0)) / x.std(0, ddof=1)
                covariance, shrinkage = oas(z, assume_centered=True)
                scales = np.sqrt(np.diag(covariance))
                target = covariance / scales[:, None] / scales[None, :]
                max_error = max(
                    max_error, float(np.abs(target - data["matrices"][i]).max())
                )
                max_shrinkage_error = max(
                    max_shrinkage_error, abs(shrinkage - data["shrinkage"][i])
                )
                np.testing.assert_allclose(
                    target, data["matrices"][i], atol=2e-14, rtol=0
                )
                np.testing.assert_allclose(
                    np.corrcoef(x, rowvar=False),
                    data["empirical_matrices"][i],
                    atol=2e-14,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    shrinkage, data["shrinkage"][i], atol=1e-15, rtol=0
                )
                if len(x) != data["n_timepoints"][i]:
                    raise ValueError("Time-point count differs")
    common = 0
    for split in ("train", "validation", "test"):
        small, large = datasets["same601"][split], datasets["expanded762"][split]
        index = {int(s): i for i, s in enumerate(large["subject_ids"])}
        take = [index[int(s)] for s in small["subject_ids"]]
        for key in small:
            np.testing.assert_array_equal(
                small[key], large[key] if key == "node_ids" else large[key][take]
            )
        common += len(take)
    rows = []
    for cohort in ("same601", "expanded762"):
        directory = root / "controls" / cohort / "utility"
        selection_path = directory / "selection.json"
        if (
            sha256(selection_path)
            != registration["cohorts"][cohort]["selection_sha256"]
        ):
            raise ValueError("Selection changed")
        selection = json.loads(selection_path.read_text())
        if selection["code_sha256"] != sha256(Path(utility.__file__)):
            raise ValueError("Utility implementation changed")
        for name, expected in selection["model_sha256"].items():
            if sha256(directory / name) != expected:
                raise ValueError("Selected model changed")
        train, val = (datasets[cohort][s] for s in ("train", "validation"))
        scaler = joblib.load(directory / "feature_scaler.joblib")
        reconstructed = StandardScaler().fit(utility.features(train["matrices"]))
        np.testing.assert_array_equal(scaler.mean_, reconstructed.mean_)
        np.testing.assert_array_equal(scaler.scale_, reconstructed.scale_)
        features = scaler.transform(utility.features(val["matrices"]))
        for arm in selection["arms"]:
            best = max(arm["validation"], key=lambda row: row["roc_auc"])
            if best["C"] != arm["C"]:
                raise ValueError("Selection or tie-breaking rule violated")
            model = joblib.load(directory / f"{arm['name']}.joblib")
            score = utility.metrics(val["labels"], model.predict_proba(features)[:, 1])
            for metric, value in score.items():
                if value != best[metric]:
                    raise ValueError(
                        "Native validation score does not reproduce exactly"
                    )
            rows.append(
                {
                    "cohort": cohort,
                    "arm": arm["name"],
                    "C": arm["C"],
                    "validation_auc": score["roc_auc"],
                    "exact": True,
                }
            )
    report = {
        "status": "passed",
        "source_reconstructions": len(seen),
        "maximum_target_entry_error": max_error,
        "maximum_shrinkage_error": max_shrinkage_error,
        "common_subjects_all_arrays_identical": common,
        "native_validation_selections": rows,
        "all_native_validation_scores_exact": True,
        "all_scalers_fit_real_training_only": True,
        "selection_registration_sha256": sha256(registration_path),
        "construction_sha256": sha256(root / "data/construction_audit.json"),
        "audit_code_sha256": sha256(Path(__file__)),
        "test_outcomes_scored_by_this_audit": False,
        "scope": (
            "Independent adapter calculation using the same sklearn OAS library; "
            "same acquired sources, not an external-person replication"
        ),
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "native_validation_selections"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
