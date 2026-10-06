"""Measure session overlap and construct diagnostic purged/disjoint row indices.

This does not mutate a paper split, relabel examples, fit a model or create a
fresh holdout. Session disjointness is a stronger criterion than excluding only
shared date/asset observations; it does not imply statistical independence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrbench.reference import sha256

SPLITS = ("train", "validation", "test")


def window_views(session_dates, end_dates, lookback):
    """Return row masks using actual observed sessions, never calendar offsets."""
    sessions = np.asarray(session_dates, dtype="datetime64[ns]")
    if lookback < 1 or len(sessions) < lookback:
        raise ValueError("Insufficient sessions or invalid lookback")
    if np.isnat(sessions).any() or np.any(sessions[1:] <= sessions[:-1]):
        raise ValueError("Source sessions must be unique, finite and sorted")
    positions = {}
    for split in SPLITS:
        dates = np.asarray(end_dates[split], dtype="datetime64[ns]")
        if dates.ndim != 1 or not len(dates) or np.isnat(dates).any():
            raise ValueError("Each original split needs finite row end dates")
        pos = np.searchsorted(sessions, dates)
        if np.any(pos >= len(sessions)) or np.any(sessions[pos] != dates):
            raise ValueError("Window end absent from source sessions")
        if np.any(pos < lookback - 1):
            raise ValueError("Window extends before source data")
        positions[split] = pos
    for earlier, later in zip(SPLITS[:-1], SPLITS[1:], strict=True):
        if positions[earlier].max() >= positions[later].min():
            raise ValueError("Original end-date splits are not chronological")
    views = {key: {} for key in ["original", "boundary_purged", "disjoint_blocks"]}
    prior_original_end = -1
    for split in SPLITS:
        pos = positions[split]
        allowed = pos - lookback + 1 > prior_original_end
        chosen = []
        previous_end = prior_original_end
        for end in np.unique(pos[allowed]):
            if end - lookback + 1 > previous_end:
                chosen.append(end)
                previous_end = end
        views["original"][split] = np.arange(len(pos))
        views["boundary_purged"][split] = np.flatnonzero(allowed)
        views["disjoint_blocks"][split] = np.flatnonzero(np.isin(pos, chosen))
        # Keep identical boundary rules in both diagnostic views, even if the
        # disjoint view retained an earlier final parent window.
        prior_original_end = int(pos.max())
    report = {}
    for name, view in views.items():
        covered, rows = {}, {}
        for split in SPLITS:
            unique = np.unique(positions[split][view[split]])
            mask = np.zeros(len(sessions), dtype=bool)
            for end in unique:
                mask[end - lookback + 1 : end + 1] = True
            covered[split] = mask
            rows[split] = {
                "matrices": len(view[split]),
                "parent_windows": len(unique),
                "return_sessions": int(mask.sum()),
                "first_return_date": str(sessions[mask][0]) if mask.any() else None,
                "last_return_date": str(sessions[mask][-1]) if mask.any() else None,
                "within_split_parent_windows_disjoint": bool(
                    len(unique) < 2 or np.all(np.diff(unique) >= lookback)
                ),
            }
        overlaps = {
            f"{a}__{b}": int(np.sum(covered[a] & covered[b]))
            for a, b in [
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ]
        }
        if name != "original":
            assert not any(overlaps.values())
        if name == "disjoint_blocks":
            assert all(v["within_split_parent_windows_disjoint"] for v in rows.values())
        report[name] = {
            "splits": rows,
            "shared_return_sessions": overlaps,
            "all_splits_nonempty": all(len(v) for v in view.values()),
        }
    return views, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit directory")
    manifest = json.loads((args.data / "manifest.json").read_text())
    assert sha256(args.source) == manifest["source_sha256"]
    dates = pd.read_parquet(args.source, columns=[args.date_column])[args.date_column]
    sessions = np.sort(pd.to_datetime(dates).unique().to_numpy(dtype="datetime64[ns]"))
    ends, source_hashes = {}, {}
    for split in SPLITS:
        path = args.data / f"{split}.npz"
        source_hashes[split] = sha256(path)
        assert source_hashes[split] == manifest["splits"][split]["sha256"]
        with np.load(path) as arrays:
            ends[split] = arrays["end_dates"]
    views, summary = window_views(sessions, ends, manifest["sampling"]["lookback"])
    args.output.mkdir(parents=True)
    for name, selections in views.items():
        np.savez_compressed(args.output / f"{name}_indices.npz", **selections)
    result = {
        "status": "passed",
        "market": manifest["market"],
        "scope": "Construction diagnostic on inspected paper periods; no new holdout",
        "source_sha256": sha256(args.source),
        "split_sha256": source_hashes,
        "manifest_sha256": sha256(args.data / "manifest.json"),
        "code_sha256": sha256(Path(__file__)),
        "lookback_sessions": manifest["sampling"]["lookback"],
        "views": summary,
        "indices_sha256": {p.name: sha256(p) for p in args.output.glob("*.npz")},
        "limits": "Date-disjoint observations are not independent market regimes. "
        "Asset subsets at one parent window still share returns. Empty validation "
        "requires redesigning dates, not silently retaining overlapping rows. "
        "These indices preserve paper labels; no new training/calibration is claimed.",
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
