#!/usr/bin/env python3
"""Audit the curated JPX panel and its conventional/projective datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _dates(values: np.ndarray) -> tuple[str, str]:
    days = values.astype("datetime64[D]")
    return str(days.min()), str(days.max())


def _matrix_split(path: Path) -> tuple[dict[str, Any], set[str], np.ndarray]:
    with np.load(path) as data:
        matrices = data["matrices"]
        assets = data["assets"]
        end_dates = data["end_dates"]
        labels = data["labels"]
        minimum_eigenvalues = data["minimum_eigenvalues"]
        report = {
            "samples": int(matrices.shape[0]),
            "matrix_dim": int(matrices.shape[1]),
            "date_min": _dates(end_dates)[0],
            "date_max": _dates(end_dates)[1],
            "unique_end_dates": int(np.unique(end_dates).size),
            "unique_assets": int(np.unique(assets).size),
            "label_counts": {
                str(label): int(np.count_nonzero(labels == label)) for label in range(3)
            },
            "maximum_diagonal_error": float(
                np.max(np.abs(np.diagonal(matrices, axis1=1, axis2=2) - 1.0))
            ),
            "maximum_symmetry_error": float(
                np.max(np.abs(matrices - matrices.transpose(0, 2, 1)))
            ),
            "minimum_eigenvalue": float(minimum_eigenvalues.min()),
        }
        return report, set(np.unique(assets).tolist()), end_dates.copy()


def _gft_split(path: Path, matrix_path: Path) -> dict[str, Any]:
    with np.load(path) as gft, np.load(matrix_path) as matrices:
        metadata_equal = all(
            np.array_equal(gft[key], matrices[key])
            for key in ("labels", "end_dates", "assets")
        )
        return {
            "rows": int(gft["gft"].shape[0]),
            "coordinates": int(gft["gft"].shape[1]),
            "all_finite": bool(np.isfinite(gft["gft"]).all()),
            "metadata_equal_to_matrix_split": bool(metadata_equal),
            "minimum_input_eigenvalue": float(gft["minimum_input_eigenvalue"]),
        }


def _projective_split(root: Path, split: str, dims: list[int]) -> dict[str, Any]:
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for dim in dims:
        with np.load(root / f"n{dim}" / f"{split}.npz") as data:
            arrays[dim] = {
                key: data[key].copy()
                for key in (
                    "matrices",
                    "assets",
                    "end_dates",
                    "labels",
                    "sharpes",
                    "parent_minimum_eigenvalues",
                    "parent_condition_numbers",
                )
            }

    parent_dim = max(dims)
    parent = arrays[parent_dim]
    maximum_nesting_error = 0.0
    all_metadata_equal = True
    for dim in dims:
        child = arrays[dim]
        maximum_nesting_error = max(
            maximum_nesting_error,
            float(
                np.max(np.abs(child["matrices"] - parent["matrices"][:, :dim, :dim]))
            ),
        )
        all_metadata_equal &= (
            np.array_equal(child["assets"], parent["assets"][:, :dim])
            and np.array_equal(child["end_dates"], parent["end_dates"])
            and np.array_equal(child["labels"], parent["labels"])
            and np.array_equal(child["sharpes"], parent["sharpes"])
        )
    return {
        "samples": int(parent["matrices"].shape[0]),
        "matrix_dims": dims,
        "maximum_nesting_error": maximum_nesting_error,
        "paired_metadata_exact": bool(all_metadata_equal),
        "parent_minimum_eigenvalue": float(parent["parent_minimum_eigenvalues"].min()),
        "parent_condition_number_median": float(
            np.median(parent["parent_condition_numbers"])
        ),
        "parent_condition_number_maximum": float(
            parent["parent_condition_numbers"].max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--projective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_names = ("train", "validation", "test")
    conventional: dict[str, Any] = {}
    selected_by_split: dict[str, set[str]] = {}
    dates_by_split: dict[str, np.ndarray] = {}
    for split in split_names:
        report, selected, dates = _matrix_split(args.matrices / f"{split}.npz")
        conventional[split] = report
        selected_by_split[split] = selected
        dates_by_split[split] = dates

    temporal_order = (
        dates_by_split["train"].max() < dates_by_split["validation"].min()
        and dates_by_split["validation"].max() < dates_by_split["test"].min()
    )

    panel = pd.read_parquet(
        args.panel,
        columns=["date", "instrument_id", "supervision_flag"],
    )
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    archive_start = panel["date"].min()
    archive_end = panel["date"].max()
    life = panel.groupby("instrument_id")["date"].agg(["min", "max"])
    ever_supervised = set(
        panel.loc[panel["supervision_flag"].fillna(False), "instrument_id"].unique()
    )

    metadata = pd.read_parquet(args.metadata)
    metadata["instrument_id"] = metadata["instrument_id"].astype(str)
    metadata = metadata.set_index("instrument_id", drop=False)
    selected_all = set().union(*selected_by_split.values())
    selected_meta = metadata.reindex(sorted(selected_all))
    valid_sector = selected_meta["33SectorName"].notna() & selected_meta[
        "33SectorName"
    ].ne("-")
    ending_early = life.index[life["max"] < archive_end]
    selected_ending_early = sorted(selected_all.intersection(ending_early))
    examples = metadata.reindex(selected_ending_early).loc[
        :, ["instrument_id", "Name", "33SectorName"]
    ]
    examples = examples.head(20).fillna("").to_dict(orient="records")

    for split in split_names:
        selected = selected_by_split[split]
        conventional[split]["assets_ending_before_archive_end"] = len(
            selected.intersection(ending_early)
        )
        conventional[split]["assets_ever_under_supervision"] = len(
            selected.intersection(ever_supervised)
        )

    gft = {
        split: _gft_split(
            args.matrices / "gft" / f"{split}.npz",
            args.matrices / f"{split}.npz",
        )
        for split in split_names
    }
    dims = sorted(
        int(path.name.removeprefix("n"))
        for path in args.projective.glob("n*")
        if path.is_dir() and path.name.removeprefix("n").isdigit()
    )
    projective = {
        split: _projective_split(args.projective, split, dims) for split in split_names
    }

    report = {
        "panel": {
            "rows": int(len(panel)),
            "assets": int(panel["instrument_id"].nunique()),
            "date_min": str(archive_start.date()),
            "date_max": str(archive_end.date()),
            "assets_starting_after_archive_start": int(
                (life["min"] > archive_start).sum()
            ),
            "assets_ending_before_archive_end": int((life["max"] < archive_end).sum()),
            "assets_ever_under_supervision": len(ever_supervised),
        },
        "conventional": conventional,
        "chronological_splits_strictly_disjoint": bool(temporal_order),
        "selected_assets": {
            "unique": len(selected_all),
            "metadata_name_coverage": float(selected_meta["Name"].notna().mean()),
            "snapshot_sector_coverage": float(valid_sector.mean()),
            "ending_before_archive_end": len(selected_ending_early),
            "ending_before_archive_end_examples": examples,
        },
        "gft_cache": gft,
        "projective": projective,
        "metadata_caveat": (
            "The supplied security master is a 2021-12 snapshot. Sector fields are "
            "node attributes, but are not point-in-time historical classifications."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
