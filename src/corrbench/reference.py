"""Stage dated-market reference data for the Hub without any network writes."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REQUIRED = {"matrices", "labels", "end_dates", "assets"}
OPTIONAL = {"original_labels", "sharpes", "minimum_eigenvalues"}
FIELD_NAMES = {
    "matrices": "matrix",
    "labels": "condition",
    "end_dates": "window_end",
    "assets": "variable_ids",
}
PURPOSES = {
    "paper_reproduction": ("train", "validation", "test"),
    "competition_training": ("train", "validation"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_metadata(metadata: dict[str, Any]) -> None:
    required = (
        "config_name",
        "dataset_version",
        "domain",
        "protocol_id",
        "source_reference",
        "observation_definition",
        "builder_reference",
        "split_description",
        "matrix_space",
        "rights_status",
    )
    for key in required:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Metadata requires a nonempty string: {key}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", metadata["config_name"]):
        raise ValueError("config_name must use lowercase letters, digits, _ or -")
    if metadata["matrix_space"] not in {"real_correlation_pd", "real_correlation_psd"}:
        raise ValueError("Unsupported matrix_space")
    if metadata["rights_status"] not in {"pending", "restricted", "cleared"}:
        raise ValueError("rights_status must be pending, restricted or cleared")
    if metadata["rights_status"] == "restricted" and (
        not isinstance(metadata.get("rights_evidence"), str)
        or not metadata["rights_evidence"].strip()
    ):
        raise ValueError("Restricted rights require rights_evidence")
    if metadata["rights_status"] == "cleared":
        for key in ("license", "rights_evidence"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"Cleared rights require {key}")


def _validate_arrays(arrays: dict[str, np.ndarray], n: int) -> int:
    if missing := REQUIRED - arrays.keys():
        raise ValueError(f"Missing dated-market arrays: {sorted(missing)}")
    if unknown := arrays.keys() - REQUIRED - OPTIONAL:
        raise ValueError(
            f"Unsupported arrays; add an explicit adapter: {sorted(unknown)}"
        )
    m = arrays["matrices"]
    if m.ndim != 3 or m.shape[1:] != (n, n) or m.dtype.kind != "f":
        raise ValueError("matrices must have floating shape (samples, n, n)")
    count = len(m)
    if count == 0:
        raise ValueError("Empty splits are not reference datasets")
    for name, a in arrays.items():
        expected = (count, n) if name == "assets" else (count,)
        if name != "matrices" and a.shape != expected:
            raise ValueError(f"Incorrect row coverage or shape for {name}")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"Nonfinite values in {name}")
    if arrays["end_dates"].dtype.kind != "M" or np.isnat(arrays["end_dates"]).any():
        raise ValueError("end_dates must contain finite datetime64 values")
    if arrays["assets"].dtype.kind != "U":
        raise ValueError("assets must be Unicode identifiers, not pickled objects")
    if any(len(set(row)) != n or any(not x for x in row) for row in arrays["assets"]):
        raise ValueError("Variable identifiers must be nonempty and unique per matrix")
    for name in ("labels", "original_labels"):
        if name in arrays and arrays[name].dtype.kind not in "iu":
            raise ValueError(f"{name} must be integer conditions")
    return count


def _arrow_array(a: np.ndarray) -> pa.Array:
    flat = pa.array(a.reshape(-1))
    if a.ndim == 1:
        return flat
    return pa.FixedSizeListArray.from_arrays(flat, int(np.prod(a.shape[1:])))


def _write_split(
    source: Path,
    destination: Path,
    report: dict,
    config_name: str,
    split: str,
    n: int,
    lookback: int,
    space: str,
    batch_size: int,
) -> dict[str, Any]:
    input_hash = sha256(source)
    if input_hash != report["sha256"]:
        raise ValueError(f"Frozen source hash mismatch for {split}")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = _validate_arrays(arrays, n)
    if count != report["samples"]:
        raise ValueError(f"Frozen sample count mismatch for {split}")
    layout = {
        key: {
            "column": FIELD_NAMES.get(key, key),
            "dtype": a.dtype.str,
            "sample_shape": list(a.shape[1:]),
        }
        for key, a in arrays.items()
    }
    minimum = float("inf")
    writer = None
    try:
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            matrices = arrays["matrices"][start:stop].astype(np.float64)
            if not np.allclose(matrices, matrices.swapaxes(-1, -2), atol=1e-6, rtol=0):
                raise ValueError(f"Asymmetric matrix in {split}")
            if not np.allclose(
                np.diagonal(matrices, axis1=1, axis2=2), 1, atol=1e-6, rtol=0
            ):
                raise ValueError(f"Non-unit diagonal in {split}")
            eigenvalues = np.linalg.eigvalsh((matrices + matrices.swapaxes(-1, -2)) / 2)
            local_min = float(eigenvalues[:, 0].min())
            if local_min < -1e-6 or (space == "real_correlation_pd" and local_min <= 0):
                raise ValueError(f"Matrix outside declared space in {split}")
            minimum = min(minimum, local_min)
            columns = {
                "sample_id": pa.array(
                    [f"{config_name}:{split}:{i}" for i in range(start, stop)]
                ),
                "source_row_index": pa.array(np.arange(start, stop, dtype=np.int64)),
                "n_observations": pa.array(
                    np.full(stop - start, lookback, dtype=np.int32)
                ),
                **{
                    FIELD_NAMES.get(key, key): _arrow_array(a[start:stop])
                    for key, a in arrays.items()
                },
            }
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {
        "samples": count,
        "input_npz_sha256": input_hash,
        "parquet_sha256": sha256(destination),
        "arrays": layout,
        "minimum_eigenvalue_float64": minimum,
        "window_end_min": str(arrays["end_dates"].min()),
        "window_end_max": str(arrays["end_dates"].max()),
    }


def _dataset_card(metadata: dict, manifest: dict) -> str:
    config = metadata["config_name"]
    front = {
        "pretty_name": f"CorrBench / {config}",
        "tags": ["correlation-matrices", "corrbench", metadata["domain"]],
        "configs": [
            {
                "config_name": config,
                "data_files": [
                    {"split": split, "path": data["path"]}
                    for split, data in manifest["splits"].items()
                ],
            }
        ],
    }
    if metadata["rights_status"] == "cleared":
        front["license"] = metadata["license"]
    counts = "\n".join(
        f"- {k}: {v['samples']} matrices" for k, v in manifest["splits"].items()
    )
    return (
        "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n"
        f"# CorrBench / {config}\n\n"
        f"Local staging export, version `{metadata['dataset_version']}`. "
        "No Hub upload is performed by this tool.\n\n"
        f"**Data rights: {metadata['rights_status']}.** "
        + (
            "Public distribution is blocked by the recorded source terms.\n\n"
            if metadata["rights_status"] == "restricted"
            else (
                "Public distribution is blocked pending source-specific review.\n\n"
                if metadata["rights_status"] == "pending"
                else ""
            )
        )
        + (
            f"Evidence: {metadata['rights_evidence']}\n\n"
            if metadata.get("rights_evidence")
            else ""
        )
        + f"Purpose: `{manifest['purpose']}`. "
        + f"Protocol: `{metadata['protocol_id']}`.\n\n"
        + f"Source: {metadata['source_reference']}\n\n"
        + f"Observations: {metadata['observation_definition']}\n\n"
        + f"Builder: {metadata['builder_reference']}\n\n"
        + f"Splits: {metadata['split_description']}\n\n{counts}\n\n"
        + "## Representation and provenance\n\n"
        + f"`matrix` is the complete {manifest['matrix_dim']} "
        + f"by {manifest['matrix_dim']} "
        + "matrix flattened in row-major order. Reshape using `matrix_dim` from "
        + "`manifest.json`. All entries are retained; no repair or re-estimation "
        + "is performed. `variable_ids` follows that same order. `condition` and "
        + "optional source diagnostics retain their original types and values.\n\n"
        + "`window_end` is the observation-window endpoint; `n_observations` is "
        + "the lookback in source sessions. Recover the start from the source "
        + "calendar, not by subtracting calendar days. `sample_id` and "
        + "`source_row_index` identify each original matrix. The manifest records "
        + "exact array layouts, source hashes, frozen conditions and output hashes.\n\n"
        + "## Intended use and limits\n\n"
        + "Subject to the recorded source permissions, these files describe "
        + "the declared matrix-generation benchmark. These matrices "
        + "are estimates, not return trajectories or guaranteed independent "
        + "observations. Group related dates/subjects/parent samples for new "
        + "evaluation. Frozen paper splits must not be described as a fresh "
        + "competition holdout. Public observations can reconstruct hidden "
        + "correlations; omitting test files does not establish a blind contest.\n\n"
        + "Source-specific data rights are separate from the software's MIT "
        + "license. Final paper/repository citation links and any source-required "
        + "attribution must be reviewed before publication.\n"
    )


def export_reference(
    data: Path,
    output: Path,
    metadata: dict[str, Any],
    *,
    purpose: str = "paper_reproduction",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Write an atomic, explicitly local staging bundle from frozen market splits."""
    _validate_metadata(metadata)
    if purpose not in PURPOSES or batch_size < 1:
        raise ValueError("Invalid purpose or batch_size")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing release: {output}")
    original = json.loads((data / "manifest.json").read_text())
    n, lookback = original["sampling"]["matrix_dim"], original["sampling"]["lookback"]
    if not isinstance(n, int) or n < 2 or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("Invalid matrix_dim or lookback in construction manifest")
    manifest = {
        "schema_version": "corrbench.reference.v1",
        "adapter": "dated_market_npz.v1",
        "purpose": purpose,
        "metadata": metadata,
        "matrix_dim": n,
        "distribution_blocked": metadata["rights_status"] != "cleared",
        "construction_manifest_sha256": sha256(data / "manifest.json"),
        "construction": {
            key: original[key]
            for key in (
                "market",
                "source_sha256",
                "source_rows",
                "source_assets",
                "source_sessions",
                "sampling",
                "regime_definition",
            )
            if key in original
        },
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".corrbench-export-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "release"
        stage.mkdir()
        (stage / "data").mkdir()
        for split in PURPOSES[purpose]:
            relative = f"data/{split}.parquet"
            result = _write_split(
                data / f"{split}.npz",
                stage / relative,
                original["splits"][split],
                metadata["config_name"],
                split,
                n,
                lookback,
                metadata["matrix_space"],
                batch_size,
            )
            manifest["splits"][split] = {"path": relative, **result}
        card = stage / "README.md"
        card.write_text(_dataset_card(metadata, manifest))
        manifest["dataset_card_sha256"] = sha256(card)
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(
            sha256(manifest_path) + "  manifest.json\n"
        )
        stage.rename(output)
    return manifest


def read_reference_split(directory: Path, split: str) -> dict[str, np.ndarray]:
    """Restore original NPZ arrays from a staged split, checking file integrity."""
    manifest_file = directory / "manifest.json"
    expected = (directory / "manifest.sha256").read_text().split()[0]
    if sha256(manifest_file) != expected:
        raise ValueError("Release manifest hash mismatch")
    manifest = json.loads(manifest_file.read_text())
    info = manifest["splits"][split]
    path = (directory / info["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Split path escapes the release")
    if sha256(path) != info["parquet_sha256"]:
        raise ValueError("Parquet hash mismatch")
    table = pq.read_table(path)
    if table.num_rows != info["samples"]:
        raise ValueError("Parquet row count mismatch")
    restored = {}
    for name, spec in info["arrays"].items():
        column = table[spec["column"]].combine_chunks()
        flat = column.values if spec["sample_shape"] else column
        a = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=spec["dtype"])
        restored[name] = a.reshape((info["samples"], *spec["sample_shape"]))
    return restored
