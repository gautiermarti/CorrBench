"""Lossless local Parquet adapters for the declared brain and weather pilots."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from corrbench.reference import (
    _arrow_array,
    _validate_metadata,
    read_reference_split,
    sha256,
)

COMMON = {
    "matrices",
    "empirical_matrices",
    "labels",
    "node_ids",
    "shrinkage",
    "empirical_rank",
}
FIELDS = {
    "abide_pcp": COMMON
    | {
        "minimum_empirical_eigenvalue",
        "minimum_target_eigenvalue",
        "n_timepoints",
        "subject_ids",
        "site_ids",
        "ages",
        "sex",
        "mean_framewise_displacement",
    },
    "ncep_temperature": COMMON
    | {
        "region_ids",
        "quarters",
        "years",
        "window_start",
        "window_end",
        "observation_count",
        "minimum_eigenvalue",
        "empirical_minimum_eigenvalue",
        "node_coordinates",
        "anomaly_standard_deviation",
        "group_ids",
        "sample_ids",
    },
}
REQUIRED = {
    "abide_pcp": {"matrices", "labels", "node_ids", "subject_ids", "site_ids"},
    "ncep_temperature": {
        "matrices",
        "labels",
        "node_ids",
        "node_coordinates",
        "years",
        "region_ids",
        "quarters",
        "window_start",
        "window_end",
        "sample_ids",
    },
}
RENAMES = {
    "matrices": "matrix",
    "empirical_matrices": "empirical_matrix",
    "labels": "condition",
    "node_ids": "variable_ids",
}


def validate(arrays, adapter):
    if REQUIRED[adapter] - arrays.keys() or arrays.keys() - FIELDS[adapter]:
        raise ValueError("Missing required fields or unknown arrays in domain adapter")
    matrices = arrays["matrices"]
    if (
        matrices.ndim != 3
        or matrices.shape[-1] != matrices.shape[-2]
        or len(matrices) < 1
    ):
        raise ValueError("Require nonempty square matrix batches")
    count, n, _ = matrices.shape
    static = {"node_ids"} if adapter == "abide_pcp" else set()
    for name, a in arrays.items():
        if a.dtype.kind not in "fiubU":
            raise ValueError(f"Unsupported array dtype: {name}")
        if name in static:
            if a.shape != (n,):
                raise ValueError("Incorrect static atlas IDs")
        elif a.ndim < 1 or a.shape[0] != count:
            raise ValueError(f"Row count differs for {name}")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"Nonfinite values: {name}")
        if name in {"matrices", "empirical_matrices"}:
            if a.shape != (count, n, n):
                raise ValueError("Matrix target shapes differ")
            if not np.allclose(a, a.swapaxes(-1, -2), atol=1e-6, rtol=0):
                raise ValueError("Asymmetric target")
            if not np.allclose(
                np.diagonal(a, axis1=-2, axis2=-1), 1, atol=1e-6, rtol=0
            ):
                raise ValueError("Nonunit diagonal")
            minimum = float(np.linalg.eigvalsh(a.astype(float)).min())
            if minimum < -1e-6 or (name == "matrices" and minimum <= 0):
                raise ValueError("Matrix lies outside declared PD/PSD space")
    ids = arrays["node_ids"]
    if ids.shape != ((n,) if static else (count, n)):
        raise ValueError("Node IDs do not align with matrices")
    if any(len(np.unique(row)) != n for row in (ids[None] if static else ids)):
        raise ValueError("Duplicate node identity")
    if adapter == "ncep_temperature" and arrays["node_coordinates"].shape != (
        count,
        n,
        2,
    ):
        raise ValueError("Coordinates do not align with nodes")
    if arrays["labels"].shape != (count,) or arrays["labels"].dtype.kind not in "iu":
        raise ValueError("Conditions must be integer labels")
    return count, n, static


def export_domain_reference(
    data, output, metadata, *, adapter, purpose="paper_reproduction"
):
    _validate_metadata(metadata)
    for key in ["independent_units", "limitations"]:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Require domain-specific metadata: {key}")
    if metadata.get("license") == "other":
        for key in ["license_name", "license_link"]:
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"Custom license requires {key}")
        if not re.fullmatch(r"[a-z0-9.-]+", metadata["license_name"]):
            raise ValueError("Custom license_name must be a lowercase Hub identifier")
    if adapter not in FIELDS or purpose not in {
        "paper_reproduction",
        "competition_training",
    }:
        raise ValueError("Unknown adapter or purpose")
    if metadata["matrix_space"] != "real_correlation_pd":
        raise ValueError("These adapters preserve the declared regularized PD target")
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((data / "manifest.json").read_text())
    splits = ["train", "validation"]
    if purpose == "paper_reproduction":
        splits += ["test"]
        if adapter == "ncep_temperature":
            splits += ["spatial_transfer"]
    manifest = {
        "schema_version": "corrbench.reference.v1",
        "adapter": f"{adapter}.v1",
        "purpose": purpose,
        "metadata": metadata,
        "distribution_blocked": metadata["rights_status"] != "cleared",
        "construction_manifest_sha256": sha256(data / "manifest.json"),
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".corrbench-domain-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "release"
        (stage / "data").mkdir(parents=True)
        for split in splits:
            path = data / f"{split}.npz"
            expected = original["splits"][split]
            digest = sha256(path)
            if digest != expected.get("sha256", expected.get("source_sha256")):
                raise ValueError(f"Frozen source hash mismatch: {split}")
            with np.load(path, allow_pickle=False) as a:
                arrays = {k: a[k] for k in a.files}
            count, n, static = validate(arrays, adapter)
            if count != expected.get("subjects", expected.get("matrices")):
                raise ValueError("Construction row count mismatch")
            if "matrix_dim" in manifest and manifest["matrix_dim"] != n:
                raise ValueError("Matrix dimension differs across splits")
            manifest["matrix_dim"] = n
            layout, columns = {}, {}
            for name, a in arrays.items():
                repeated = (
                    np.broadcast_to(a, (count, *a.shape)) if name in static else a
                )
                column = RENAMES.get(name, name)
                columns[column] = _arrow_array(repeated)
                layout[name] = {
                    "column": column,
                    "dtype": a.dtype.str,
                    "sample_shape": list(repeated.shape[1:]),
                    "static": name in static,
                }
            columns["source_row_index"] = pa.array(np.arange(count, dtype=np.int64))
            relative = f"data/{split}.parquet"
            pq.write_table(pa.table(columns), stage / relative, compression="zstd")
            manifest["splits"][split] = {
                "path": relative,
                "samples": count,
                "input_npz_sha256": digest,
                "parquet_sha256": sha256(stage / relative),
                "arrays": layout,
            }
        front = {
            "pretty_name": f"CorrBench / {metadata['config_name']}",
            "tags": ["corrbench", "correlation-matrices", metadata["domain"]],
            "configs": [
                {
                    "config_name": metadata["config_name"],
                    "data_files": [
                        {"split": s, "path": r["path"]}
                        for s, r in manifest["splits"].items()
                    ],
                }
            ],
        }
        if metadata["rights_status"] == "cleared":
            front["license"] = metadata["license"]
            if metadata["license"] == "other":
                front["license_name"] = metadata["license_name"]
                front["license_link"] = metadata["license_link"]
        body = [
            f"# CorrBench / {metadata['config_name']}",
            "",
            "Local reference staging. This tool performs no Hub upload.",
            "",
            f"Version: {metadata['dataset_version']}. "
            f"Protocol: {metadata['protocol_id']}.",
            f"Data rights: **{metadata['rights_status']}**. Purpose: {purpose}.",
            "",
            f"Source: {metadata['source_reference']}",
            "",
            f"Observations: {metadata['observation_definition']}",
            "",
            f"Builder: {metadata['builder_reference']}",
            "",
            f"Splits: {metadata['split_description']}",
            "",
            f"Rights evidence: {metadata.get('rights_evidence', 'Review pending')}",
            "",
            "## Representation",
            "",
            f"Each matrix is a flattened {n}-by-{n} array in row-major order.",
            "The primary matrix is the declared regularized PD estimator.",
            "The empirical matrix is a separate, possibly singular target.",
            "variable_ids and node_coordinates follow matrix order.",
            "All original arrays, types and layouts are preserved in manifest.json.",
            "Static atlas IDs repeat per record; the domain reader restores them.",
            "No source arrays are silently discarded, repaired or re-estimated.",
            "",
            "## Counts and independent units",
            "",
            *[f"- {s}: {r['samples']} matrices" for s, r in manifest["splits"].items()],
            "",
            metadata["independent_units"],
            "",
            "## Intended use and limitations",
            "",
            metadata["limitations"],
            "",
            "This is a matrix benchmark, not a complete simulator of observations.",
            "Previously inspected pilot periods are not an untouched holdout.",
            "Public observations can reconstruct targets absent from test exports.",
            "The software license does not override source-specific data restrictions.",
        ]
        card = (
            "---\n"
            + yaml.safe_dump(front, sort_keys=False)
            + "---\n\n"
            + "\n".join(body)
            + "\n"
        )
        (stage / "README.md").write_text(card)
        manifest["dataset_card_sha256"] = sha256(stage / "README.md")
        path = stage / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(sha256(path) + "  manifest.json\n")
        stage.rename(output)
    return manifest


def read_domain_reference_split(directory, split):
    restored = read_reference_split(directory, split)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, spec in manifest["splits"][split]["arrays"].items():
        if spec["static"]:
            a = restored[name]
            if not np.all(a == a[0]):
                raise ValueError("Repeated static metadata differs between rows")
            restored[name] = a[0]
    return restored
