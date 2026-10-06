"""Audit pinned PCP atlas identity and constant-series exclusions after the pilot."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

EXPECTED = {
    "ho_mask_pad.nii.gz": (
        "e2ef01f7394a5d0dde7c8cd57f8cc72079b2115b4180d6466aafb0831e08f8c1"
    ),
    "ho_roi_atlas.nii.gz": (
        "ff91cb7b6cb496950f365bb4a2906b9c277cc9a382fa950aef295455893e153e"
    ),
    "ho_labels.csv": "b719f40d03414a61e8cfc12f9fd377a493521e3624bdd00ee088e7f613f31214",
}
PCP = "fda1ec5d3f53eb97d94352e779f37c3ce46caee1"
TEMPLATES = "b64c29d83a015e92966ca0371370b0f2ceda7365"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name, digest in EXPECTED.items():
        if sha(args.resources / name) != digest:
            raise ValueError(f"Pinned atlas resource differs: {name}")
    mask = nib.load(args.resources / "ho_mask_pad.nii.gz")
    hosted = nib.load(args.resources / "ho_roi_atlas.nii.gz")
    voxels = np.asanyarray(mask.dataobj)
    assert np.array_equal(voxels, np.asanyarray(hosted.dataobj))
    assert np.array_equal(mask.affine, hosted.affine)
    ids = sorted(int(x) for x in np.unique(voxels) if x != 0)
    with (args.resources / "ho_labels.csv").open() as stream:
        names = {
            int(r[0]): r[1]
            for r in csv.reader(stream)
            if r and not r[0].startswith("#")
        }
    pilot_path = args.pilot / "manifest.json"
    pilot = json.loads(pilot_path.read_text())
    assert ids == pilot["node_ids"]
    unnamed = sorted(set(ids) - names.keys())
    assert not names.keys() - set(ids)
    assert unnamed == [3455]
    pointer = (args.resources / "cpac_mask_pointer").read_text()
    assert f"oid sha256:{EXPECTED['ho_mask_pad.nii.gz']}" in pointer
    download_path = args.series / "download_manifest.json"
    download = json.loads(download_path.read_text())
    assert len(download["files"]) == pilot["eligible_metadata_rows"] == 780
    assert download["metadata_sha256"] == pilot["source_metadata_sha256"]
    assert download["derivative"] == pilot["source"]
    rejected = []
    constant_counts, recovered, other = Counter(), Counter(), Counter()
    sites = {}
    for record in download["files"]:
        assert record["status"] == "available"
        source = args.series / record["path"]
        assert sha(source) == record["sha256"]
        text = source.read_text()
        columns = [int(v[1:]) for v in text.splitlines()[0].split()]
        assert columns == ids
        values = np.loadtxt(io.StringIO(text), comments="#")
        assert values.shape[1] == len(ids) and np.isfinite(values).all()
        bad = np.array(ids)[values.std(axis=0, ddof=1) <= 1e-12].tolist()
        site = sites.setdefault(record["site"], Counter())
        site["eligible"] += 1
        if not bad:
            site["frozen_retained"] += 1
            continue
        constant_counts.update(bad)
        rejected.append({"subject_id": record["subject_id"], "constant_ids": bad})
        if set(bad) <= set(unnamed):
            recovered[record["split"]] += 1
            site["excluded_only_for_unnamed_label"] += 1
        else:
            other[record["split"]] += 1
            site["has_other_constant_region"] += 1
    assert {r["subject_id"] for r in rejected} == {
        r["subject_id"] for r in pilot["excluded"]
    }
    report = {
        "schema": "corrbench.abide-atlas-audit.v1",
        "date": "2026-10-06",
        "scope": ("Post-pilot source diagnostic; frozen arrays and results unchanged"),
        "pcp_revision": PCP,
        "cpac_templates_revision": TEMPLATES,
        "mask_sha256": EXPECTED["ho_mask_pad.nii.gz"],
        "resource_sha256": EXPECTED,
        "hosted_atlas_voxels_and_affine_identical": True,
        "current_cpac_template_points_to_same_bytes": True,
        "shape": list(mask.shape),
        "voxel_sizes_mm": [float(v) for v in mask.header.get_zooms()],
        "atlas_label_count": len(ids),
        "official_named_label_count": len(names),
        "node_ids_match_frozen_pilot": True,
        "unnamed_labels": [
            {
                "id": code,
                "voxel_count": int(np.sum(voxels == code)),
                "volume_mm3": float(
                    np.sum(voxels == code) * np.prod(mask.header.get_zooms())
                ),
            }
            for code in unnamed
        ],
        "eligible_subjects": len(download["files"]),
        "all_input_hashes_verified": True,
        "constant_subjects": len(rejected),
        "constant_id_subject_counts": dict(sorted(constant_counts.items())),
        "excluded_only_for_unnamed_label_by_split": dict(recovered),
        "other_constant_region_by_split": dict(other),
        "per_site": {k: dict(v) for k, v in sorted(sites.items())},
        "frozen_exclusion_membership_reproduced": True,
        "proposed_110_label_nonconstant_counts": {
            k: v["subjects"] + recovered[k] for k, v in pilot["splits"].items()
        },
        "proposed_110_label_dataset_built": False,
        "fresh_holdout": False,
        "input_manifests_sha256": {
            "pilot": sha(pilot_path),
            "downloads": sha(download_path),
        },
        "audit_code_sha256": sha(Path(__file__)),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "atlas_label_count",
                    "official_named_label_count",
                    "unnamed_labels",
                    "constant_subjects",
                    "excluded_only_for_unnamed_label_by_split",
                    "proposed_110_label_nonconstant_counts",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
