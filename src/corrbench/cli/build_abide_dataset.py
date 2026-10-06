"""Build the frozen subject/site-disjoint ABIDE PCP pilot from local ROI series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrbench.neuro import (
    ABIDE_DERIVATIVE,
    ABIDE_PHENOTYPE_SHA256,
    eligible_subjects,
    read_roi_series,
    scan_correlations,
    sha256,
)


def build_dataset(metadata: Path, series: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if sha256(metadata) != ABIDE_PHENOTYPE_SHA256:
        raise ValueError("Metadata differs from declared source")
    frame = eligible_subjects(pd.read_csv(metadata))
    download = json.loads((series / "download_manifest.json").read_text())
    if (
        download["metadata_sha256"] != ABIDE_PHENOTYPE_SHA256
        or download["derivative"] != ABIDE_DERIVATIVE
        or len(download["files"]) != len(frame)
    ):
        raise ValueError("Incomplete or incompatible download manifest")
    records = {r["subject_id"]: r for r in download["files"]}
    if len(records) != len(frame) or set(records) != set(frame.SUB_ID):
        raise ValueError("Download manifest subject membership differs")
    # Establish atlas order using a training subject only. Never select ROIs
    # based on held-out series; reject incompatible/partial atlas headers.
    node_ids = None
    for row in frame[frame.split == "train"].itertuples():
        record = records[row.SUB_ID]
        if record["status"] == "available":
            path = series / record["path"]
            if sha256(path) != record["sha256"]:
                raise ValueError("Source file changed after download")
            try:
                _, node_ids = read_roi_series(path)
                break
            except ValueError:
                pass
    if node_ids is None or len(node_ids) != 111:
        raise ValueError("Could not establish the declared full 111-ROI atlas")
    accepted = {s: [] for s in ["train", "validation", "test"]}
    exclusions, provenance = [], []
    for i, row in enumerate(frame.itertuples()):
        record = records[row.SUB_ID]
        try:
            if record["status"] != "available":
                raise ValueError("Source download unavailable")
            path = series / record["path"]
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"Source file changed: {path}")
            values, ids = read_roi_series(path)
            if not np.array_equal(ids, node_ids):
                raise ValueError("Atlas header differs from fixed training header")
            result = scan_correlations(values)
            result.update(
                {
                    "labels": int(row.label),
                    "subject_ids": int(row.SUB_ID),
                    "site_ids": row.SITE_ID,
                    "ages": float(row.AGE_AT_SCAN),
                    "sex": int(row.SEX),
                    "mean_framewise_displacement": float(row.func_mean_fd),
                }
            )
            accepted[row.split].append(result)
            provenance.append(
                {k: record[k] for k in ["subject_id", "site", "split", "sha256", "url"]}
            )
        except ValueError as error:
            exclusions.append(
                {
                    "subject_id": int(row.SUB_ID),
                    "site": row.SITE_ID,
                    "split": row.split,
                    "reason": str(error),
                }
            )
        if (i + 1) % 100 == 0:
            print(f"Constructed {i + 1}/{len(frame)} subjects", flush=True)
    for split, rows in accepted.items():
        if not rows or {r["labels"] for r in rows} != {0, 1}:
            raise ValueError(f"Need both classes in {split}")
    groups = {s: {r["subject_ids"] for r in rows} for s, rows in accepted.items()}
    if any(
        groups[a] & groups[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise AssertionError("Subject overlap")
    output.mkdir(parents=True)
    summary = {}
    for split, rows in accepted.items():
        arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
        arrays["node_ids"] = node_ids
        path = output / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        summary[split] = {
            "subjects": len(rows),
            "classes": {
                str(label): int((arrays["labels"] == label).sum()) for label in [0, 1]
            },
            "sites": sorted(set(arrays["site_ids"].tolist())),
            "timepoints_min_median_max": [
                int(arrays["n_timepoints"].min()),
                float(np.median(arrays["n_timepoints"])),
                int(arrays["n_timepoints"].max()),
            ],
            "rank_deficient_empirical_count": int(
                (arrays["empirical_rank"] < 111).sum()
            ),
            "minimum_target_eigenvalue": float(
                arrays["minimum_target_eigenvalue"].min()
            ),
            "median_shrinkage": float(np.median(arrays["shrinkage"])),
            "sha256": sha256(path),
        }
    manifest = {
        "protocol": "abide_pcp_ho_site_holdout_v1_20261005",
        "source": ABIDE_DERIVATIVE,
        "source_metadata_sha256": ABIDE_PHENOTYPE_SHA256,
        "independent_unit": "subject; complete scan; no windows",
        "target": "per-scan standardized-series OAS correlation (PD)",
        "diagnostic_target": "empirical Pearson correlation (possibly rank deficient)",
        "condition_mapping": {"0": "typical control", "1": "ASD"},
        "matrix_dim": 111,
        "node_ids": node_ids.tolist(),
        "eligible_metadata_rows": len(frame),
        "splits": summary,
        "excluded": exclusions,
        "source_files": provenance,
        "redistribution": "Not cleared; local non-commercial research pilot only",
        "builder_sha256": sha256(Path(__file__)),
    }
    import corrbench.neuro as implementation

    manifest["estimator_code_sha256"] = sha256(Path(implementation.__file__))
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"splits": summary, "excluded": len(exclusions)}, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build_dataset(args.metadata, args.series, args.output)


if __name__ == "__main__":
    main()
