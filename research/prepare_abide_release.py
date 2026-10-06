"""Prepare three explicit ABIDE reference views and verify every restored array."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from corrbench.domain_reference import (
    export_domain_reference,
    read_domain_reference_split,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((root / "configs/releases/abide_pilot.yaml").read_text())
    configs, report = [], {}
    for name, source, cohort, counts in [
        ("brain_abide_ho_n111", "artifacts/abide_ho_20261005", "original111",
         [349, 92, 160]),
        ("brain_abide_ho_n110_same601",
         "artifacts/abide_named110_20261006/data/same601", "same601",
         [349, 92, 160]),
        ("brain_abide_ho_n110_expanded762",
         "artifacts/abide_named110_20261006/data/expanded762", "expanded762",
         [488, 108, 166]),
    ]:
        metadata = dict(base)
        metadata["config_name"] = name
        metadata["dataset_version"] = "review-20261006"
        metadata["rights_status"] = "pending"
        metadata["rights_evidence"] = (
            "ABIDE I links CC BY-NC-SA 3.0. Section 3 permits adaptations and "
            "distribution subject to section 4. Dataset, investigator and "
            "funder attribution is now complete in docs/abide_attribution_20261006.md. "
            "FSL separately identifies Harvard-Oxford as CC BY-SA 4.0. "
            "The unresolved scope is whether those atlas terms apply to these "
            "numerical matrix derivatives beyond attribution; atlas images and "
            "label-name tables are excluded. No unrestricted or commercial "
            "redistribution permission is claimed. See docs/abide_release_review.md."
        )
        if cohort != "original111":
            metadata["protocol_id"] = "abide_named110_protocol_20261006_" + cohort
            metadata["observation_definition"] = (
                "PCP CPAC filt_noglobal rois_ho; remove unnamed ID 3455 before "
                "per-scan standardization and OAS estimation; 110 named ROI IDs."
            )
            metadata["builder_reference"] = (
                "research/build_abide_named110.py; "
                "docs/abide_named110_protocol_20261006.md"
            )
            metadata["independent_units"] = (
                f"{sum(counts)} subjects across disjoint sites: {counts[0]} train, "
                f"{counts[1]} validation, {counts[2]} test. The three configurations "
                "overlap and must not be pooled as independent observations."
            )
            metadata["limitations"] = (
                "Separately declared post-pilot sensitivity, not a fresh holdout. "
                "Same601 matches the original subject cohort; expanded762 includes "
                "161 additional scans. It is not a cached 111D submatrix: "
                "standardization and OAS are re-estimated from regional series. "
                "Real/replay/Gaussian controls establish no augmentation gain; "
                "no learned 110-region model was fitted. QC, site, scan length and "
                "regularization shape the target. No clinical, causal or privacy claim."
            )
        source = root / source
        target = args.output / name
        manifest = export_domain_reference(
            source, target, metadata, adapter="abide_pcp"
        )
        if not manifest["distribution_blocked"]:
            raise ValueError("This reviewed staging does not clear redistribution")
        splits = {}
        for split, count in zip(["train", "validation", "test"], counts, strict=True):
            restored = read_domain_reference_split(target, split)
            with np.load(source / f"{split}.npz", allow_pickle=False) as arrays:
                if set(restored) != set(arrays.files):
                    raise ValueError("Fields changed during export")
                for key in arrays.files:
                    a, b = arrays[key], restored[key]
                    if a.dtype != b.dtype or not np.array_equal(
                        a, b, equal_nan=a.dtype.kind in "fc"
                    ):
                        raise ValueError(f"Array changed: {name}/{split}/{key}")
                if len(arrays["matrices"]) != count:
                    raise ValueError("Unexpected split count")
            splits[split] = {
                "rows": count, "every_array_and_dtype_exact": True,
                "parquet_sha256": sha(target / f"data/{split}.parquet"),
            }
        report[name] = {
            "cohort": cohort, "matrix_dim": manifest["matrix_dim"],
            "manifest_sha256": sha(target / "manifest.json"),
            "distribution_blocked": True, "splits": splits,
        }
        configs.append({
            "config_name": name,
            "data_files": [{"split": s, "path": f"{name}/data/{s}.parquet"}
                           for s in splits],
        })
    docs = args.output / "docs"
    docs.mkdir()
    for name in [
        "abide_attribution_20261006.md", "abide_attribution_20261006.json",
        "abide_release_review.md", "abide_atlas_audit_20261006.md",
        "abide_atlas_audit_20261006.json", "fmri_pilot_protocol.md",
        "abide_named110_protocol_20261006.md", "abide_named110_results_20261006.md",
        "abide_named110_results_20261006.json", "abide_named110_execution_20261006.md",
    ]:
        shutil.copy2(root / "docs" / name, docs / name)
    front = {"pretty_name": "CorrBench ABIDE / original and named-region views",
             "tags": ["corrbench", "neuroscience", "correlation-matrices"],
             "configs": configs}
    body = """# CorrBench ABIDE reference candidate

Local review staging; no Hub upload and no redistribution clearance. The
upstream ABIDE license is CC BY-NC-SA 3.0; software licensing does not override it.
[Attribution and funding](docs/abide_attribution_20261006.md) are complete for
all included source sites. The remaining scope question is documented in
[the release review](docs/abide_release_review.md).

| Configuration | Dimension | Train / validation / test | Meaning |
| --- | --- | --- | --- |
| brain_abide_ho_n111 | 111 | 349 / 92 / 160 | Original; includes ID 3455 |
| brain_abide_ho_n110_same601 | 110 | 349 / 92 / 160 | Re-estimated, same people |
| brain_abide_ho_n110_expanded762 | 110 | 488 / 108 / 166 | All eligible scans |

These are overlapping views of at most 762 people, not 1,964 independent
observations. Never concatenate configurations or mix their matrix dimensions.
Named110 real/replay/Gaussian controls do not establish an augmentation gain.
No learned 110-region fit, clinical validation, causal or privacy guarantee is
claimed. Test sites were already inspected. Full protocol and source details
are in each configuration's card and manifest.

Rows retain regularized OAS correlation, empirical Pearson diagnostic, diagnosis,
public source subject/site codes, age, sex, motion and estimator diagnostics.
Regional time series, raw MRI, atlas images and label-name tables are excluded.
This is an explicit transformation of ABIDE/PCP observations, not an official
ABIDE dataset release. It does not create a protected competition holdout.

Each 2D matrix is flattened in row-major order. Static integer ROI IDs follow
matrix order; the CorrBench reader restores the original arrays and dtypes.
All nine splits round-trip every array and dtype exactly in the attached audit.
"""
    (args.output / "README.md").write_text(
        "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n" + body
    )
    audit = {
        "scope": "Local three-configuration packaging; no Hub upload",
        "corrbench_version": importlib.metadata.version("corrbench"),
        "publicly_published": False, "distribution_blocked": True,
        "configurations": report,
        "files": {str(p.relative_to(args.output)): sha(p)
                  for p in sorted(args.output.rglob("*")) if p.is_file()},
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: {"dimension": v["matrix_dim"], "rows": sum(
        s["rows"] for s in v["splits"].values())}
        for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
