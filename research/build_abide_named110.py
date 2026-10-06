"""Re-estimate the separately declared named-region sensitivity datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import corrbench.neuro as neuro
from corrbench.neuro import eligible_subjects, scan_correlations, sha256

PROTOCOL = "abide_named110_protocol_20261006"
PROTOCOL_SHA = "1743c7d4409cbfb150bedc9b9abc1f49660202f19b9d8b6f15f55706a9fd6f17"
LABEL_SHA = "b719f40d03414a61e8cfc12f9fd377a493521e3624bdd00ee088e7f613f31214"
SPLITS = ("train", "validation", "test")


def read_named_series(path, source_ids, named_ids):
    """Validate the complete source before applying a fixed, external label list."""
    text = path.read_text()
    header = text.splitlines()[0].split()
    if not header or not all(re.fullmatch(r"#\d+", x) for x in header):
        raise ValueError("Malformed atlas header")
    ids = np.asarray([int(x[1:]) for x in header], dtype=np.int64)
    if not np.array_equal(ids, source_ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Atlas header differs from frozen source order")
    named_ids = np.asarray(named_ids, dtype=np.int64)
    keep = np.isin(ids, named_ids)
    if not np.array_equal(ids[keep], named_ids):
        raise ValueError("Named IDs must be a unique, ordered subset of source IDs")
    values = np.loadtxt(io.StringIO(text), comments="#", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ids) or len(values) < 50:
        raise ValueError("Invalid observations-by-regions shape or short scan")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observations")
    selected = values[:, keep]
    bad = named_ids[selected.std(axis=0, ddof=1) <= 1e-12]
    return selected, bad


def split_summary(arrays, path):
    return {
        "subjects": len(arrays["labels"]),
        "classes": {str(c): int((arrays["labels"] == c).sum()) for c in (0, 1)},
        "sites": {
            s: int((arrays["site_ids"] == s).sum())
            for s in sorted(set(arrays["site_ids"].tolist()))
        },
        "minimum_target_eigenvalue": float(arrays["minimum_target_eigenvalue"].min()),
        "median_shrinkage": float(np.median(arrays["shrinkage"])),
        "rank_deficient_empirical_count": int((arrays["empirical_rank"] < 110).sum()),
        "sha256": sha256(path),
    }


def build(bench, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = bench / "docs" / f"{PROTOCOL}.md"
    if sha256(protocol) != PROTOCOL_SHA:
        raise ValueError("Declared protocol changed")
    original = bench / "artifacts/abide_ho_20261005"
    source = bench / "artifacts/abide_source_20261005"
    series = source / "rois_ho"
    labels_path = bench / "artifacts/abide_provenance_20261006/ho_labels.csv"
    if sha256(labels_path) != LABEL_SHA:
        raise ValueError("Official label table changed")
    audit = json.loads((bench / "docs/abide_atlas_audit_20261006.json").read_text())
    parent_path = original / "manifest.json"
    download_path = series / "download_manifest.json"
    for key, path in [("pilot", parent_path), ("downloads", download_path)]:
        if sha256(path) != audit["input_manifests_sha256"][key]:
            raise ValueError(f"Audited {key} manifest changed")
    parent = json.loads(parent_path.read_text())
    download = json.loads(download_path.read_text())
    metadata = source / "phenotypic.csv"
    if sha256(metadata) != neuro.ABIDE_PHENOTYPE_SHA256:
        raise ValueError("Source metadata changed")
    frame = eligible_subjects(pd.read_csv(metadata))
    records = {r["subject_id"]: r for r in download["files"]}
    if len(records) != len(download["files"]) or set(records) != set(frame.SUB_ID):
        raise ValueError("Source membership differs")
    source_ids = np.asarray(parent["node_ids"])
    with labels_path.open() as stream:
        names = {
            int(r[0]): r[1]
            for r in csv.reader(stream)
            if r and not r[0].startswith("#")
        }
    node_ids = source_ids[np.isin(source_ids, list(names))]
    if len(node_ids) != 110 or set(source_ids) - set(node_ids) != {3455}:
        raise ValueError("Source atlas does not match the declared 110 named regions")
    frozen = {}
    for split in SPLITS:
        path = original / f"{split}.npz"
        if sha256(path) != parent["splits"][split]["sha256"]:
            raise ValueError("Frozen data changed")
        with np.load(path) as data:
            frozen[split] = {k: data[k] for k in data.files}
    accepted = {s: [] for s in SPLITS}
    exclusions, provenance = [], []
    for i, row in enumerate(frame.itertuples()):
        record = records[row.SUB_ID]
        if record["status"] != "available" or record["split"] != row.split:
            raise ValueError("Unexpected unavailable source or split mismatch")
        path = series / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Source bytes changed")
        values, bad = read_named_series(path, source_ids, node_ids)
        if len(bad):
            exclusions.append(
                {
                    "subject_id": int(row.SUB_ID),
                    "site": row.SITE_ID,
                    "split": row.split,
                    "constant_ids": bad.tolist(),
                }
            )
            continue
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
        if (i + 1) % 100 == 0:
            print(f"Reconstructed {i + 1}/{len(frame)} sources", flush=True)
    if len(exclusions) != 18:
        raise ValueError("Unexpected exclusion count")
    originally_excluded = {r["subject_id"] for r in parent["excluded"]}
    if not {r["subject_id"] for r in exclusions} <= originally_excluded:
        raise ValueError("New exclusion among frozen accepted subjects")
    datasets = {"same601": {}, "expanded762": {}}
    differences = {}
    keep = np.isin(source_ids, node_ids)
    for split in SPLITS:
        rows = accepted[split]
        arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        arrays["node_ids"] = node_ids
        if len(rows) != audit["proposed_110_label_nonconstant_counts"][split]:
            raise ValueError("Unexpected reconstructed cohort count")
        index = {int(v): i for i, v in enumerate(arrays["subject_ids"])}
        take = [index[int(v)] for v in frozen[split]["subject_ids"]]
        common = {k: v if k == "node_ids" else v[take] for k, v in arrays.items()}
        for key in [
            "labels",
            "subject_ids",
            "site_ids",
            "ages",
            "sex",
            "n_timepoints",
            "mean_framewise_displacement",
        ]:
            np.testing.assert_array_equal(common[key], frozen[split][key])
        deleted = frozen[split]["matrices"][:, keep][:, :, keep]
        changed = common["matrices"] - deleted
        shrinkage = common["shrinkage"] - frozen[split]["shrinkage"]
        differences[split] = {
            "max_absolute_entry_change_vs_cached_deletion": float(
                np.abs(changed).max()
            ),
            "mean_absolute_entry_change_vs_cached_deletion": float(
                np.abs(changed).mean()
            ),
            "shrinkage_difference_min_median_max": [
                float(shrinkage.min()),
                float(np.median(shrinkage)),
                float(shrinkage.max()),
            ],
            "all_common_subject_metadata_identical": True,
        }
        datasets["same601"][split], datasets["expanded762"][split] = common, arrays
    output.mkdir(parents=True)
    reports = {}
    for cohort, data in datasets.items():
        destination = output / cohort
        destination.mkdir()
        subject_groups = [set(data[s]["subject_ids"].tolist()) for s in SPLITS]
        site_groups = [set(data[s]["site_ids"].tolist()) for s in SPLITS]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            if subject_groups[a] & subject_groups[b] or site_groups[a] & site_groups[b]:
                raise ValueError("Subject/site overlap")
        summary = {}
        for split, arrays in data.items():
            path = destination / f"{split}.npz"
            np.savez_compressed(path, **arrays)
            summary[split] = split_summary(arrays, path)
        membership = set.union(*subject_groups)
        manifest = {
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA,
            "cohort": cohort,
            "matrix_dim": 110,
            "node_ids": node_ids.tolist(),
            "target": parent["target"],
            "source": parent["source"],
            "source_metadata_sha256": neuro.ABIDE_PHENOTYPE_SHA256,
            "source_files": [r for r in provenance if r["subject_id"] in membership],
            "splits": summary,
            "excluded_remaining_constant_regions": exclusions,
            "parent_manifest_sha256": sha256(parent_path),
            "label_table_sha256": LABEL_SHA,
            "builder_sha256": sha256(Path(__file__)),
            "estimator_code_sha256": sha256(Path(neuro.__file__)),
            "redistribution": parent["redistribution"],
            "fresh_holdout": False,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        reports[cohort] = {
            "splits": summary,
            "manifest_sha256": sha256(destination / "manifest.json"),
        }
    report = {
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA,
        "cohorts": reports,
        "remaining_exclusions": len(exclusions),
        "reconstructed_common_subjects_equal_between_cohorts": True,
        "target_change": differences,
        "original_data_unchanged": True,
        "source_hashes_checked": len(records),
        "fresh_holdout": False,
        "builder_sha256": sha256(Path(__file__)),
        "estimator_code_sha256": sha256(Path(neuro.__file__)),
        "numpy": np.__version__,
    }
    for split in SPLITS:
        if sha256(original / f"{split}.npz") != parent["splits"][split]["sha256"]:
            raise ValueError("Original data changed during reconstruction")
    (output / "construction_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.bench_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
