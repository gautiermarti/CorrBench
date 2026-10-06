"""Build the frozen NCEP regional-temperature correlation pilot from annual NetCDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrbench.neuro import sha256
from corrbench.weather import (
    REGIONS,
    correlation_targets,
    patch_indices,
    remove_training_seasonality,
)


def main():
    from netCDF4 import Dataset, num2date

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.source / "download_manifest.json").read_text())
    if set(manifest["files"]) != {str(y) for y in range(1981, 2025)}:
        raise ValueError("Require every year in the frozen protocol")
    panels = {name: [] for name, *_ in REGIONS}
    all_dates, coordinates, node_ids = [], {}, {}
    latitudes, longitudes = None, None
    for year in range(1981, 2025):
        entry = manifest["files"][str(year)]
        path = args.source / entry["filename"]
        if path.stat().st_size != entry["expected_content_length"]:
            raise ValueError(f"Truncated source: {path}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed source: {path}")
        with Dataset(path) as dataset:
            dates = np.asarray(
                [
                    np.datetime64(f"{d.year:04d}-{d.month:02d}-{d.day:02d}")
                    for d in num2date(dataset["time"][:], dataset["time"].units)
                ]
            )
            expected = np.arange(
                np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
            )
            np.testing.assert_array_equal(dates, expected)
            if dataset["air"].units not in {"degK", "K"}:
                raise ValueError("Unexpected temperature units")
            if latitudes is None:
                latitudes, longitudes = dataset["lat"][:], dataset["lon"][:]
            else:
                np.testing.assert_array_equal(latitudes, dataset["lat"][:])
                np.testing.assert_array_equal(longitudes, dataset["lon"][:])
            for name, lat, lon in REGIONS:
                rows, cols = patch_indices(latitudes, longitudes, lat, lon)
                values = dataset["air"][:, rows, cols]
                if np.ma.getmaskarray(values).any():
                    raise ValueError(f"Masked source values: {year}, {name}")
                values = np.asarray(values, dtype=np.float64).reshape(len(dates), 64)
                if (
                    not np.isfinite(values).all()
                    or values.min() < 150
                    or values.max() > 400
                ):
                    raise ValueError(f"Invalid temperature: {year}, {name}")
                panels[name].append(values)
                yy, xx = np.meshgrid(rows, cols, indexing="ij")
                node_ids[name] = (yy * len(longitudes) + xx).ravel()
                coordinates[name] = np.column_stack(
                    [
                        latitudes[yy].ravel(),
                        ((longitudes[xx] + 180) % 360 - 180).ravel(),
                    ]
                )
        all_dates.append(dates)
        print("verified", year, flush=True)
    dates = np.concatenate(all_dates)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    quarters = (months - 1) // 3
    arrays = {s: [] for s in ["train", "validation", "test", "spatial_transfer"]}
    coefficients = {}
    for region, (name, *_rest) in enumerate(REGIONS):
        values = np.concatenate(panels[name])
        anomalies, coefficients[name] = remove_training_seasonality(values, dates)
        for year in range(1981, 2025):
            if region == 3 and year < 2016:
                continue
            split = (
                "spatial_transfer"
                if region == 3
                else "train"
                if year <= 2010
                else "validation"
                if year <= 2015
                else "test"
            )
            for quarter in range(4):
                selected = (years == year) & (quarters == quarter)
                block_dates = dates[selected]
                empirical, matrix, standard_deviation, shrinkage = correlation_targets(
                    anomalies[selected]
                )
                eig = np.linalg.eigvalsh(empirical)
                arrays[split].append(
                    {
                        "matrices": matrix,
                        "empirical_matrices": empirical,
                        "labels": region * 4 + quarter,
                        "region_ids": region,
                        "quarters": quarter,
                        "years": year,
                        "window_start": str(block_dates[0]),
                        "window_end": str(block_dates[-1]),
                        "observation_count": len(block_dates),
                        "shrinkage": shrinkage,
                        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix)[0]),
                        "empirical_minimum_eigenvalue": float(eig[0]),
                        "empirical_rank": int((eig > eig[-1] * 1e-10).sum()),
                        "node_ids": node_ids[name],
                        "node_coordinates": coordinates[name],
                        "anomaly_standard_deviation": standard_deviation,
                        "group_ids": str(year),
                        "sample_ids": f"{name}:{year}:Q{quarter + 1}",
                    }
                )
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "climatology.npz", **coefficients)
    summary = {}
    for split, rows in arrays.items():
        output = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
        path = args.output / f"{split}.npz"
        np.savez_compressed(path, **output)
        summary[split] = {
            "matrices": len(rows),
            "year_groups": len(np.unique(output["years"])),
            "minimum_eigenvalue": float(output["minimum_eigenvalue"].min()),
            "source_sha256": sha256(path),
            "first_date": min(output["window_start"].tolist()),
            "last_date": max(output["window_end"].tolist()),
        }
    audit = {
        "protocol": "weather_pilot_protocol_20261005",
        "source_manifest_sha256": sha256(args.source / "download_manifest.json"),
        "builder_sha256": sha256(Path(__file__)),
        "transforms_sha256": sha256(Path(__file__).parents[1] / "weather.py"),
        "climatology_fit_end": "2010-12-31",
        "climatology_model": "Intercept plus four annual Fourier harmonics",
        "source_target": "Reanalysis daily mean 2m temperature, degK",
        "regularized_target": "Per-quarter OAS of standardized anomalies",
        "spatial_transfer": "South America; local 1981-2010 climatology calibration",
        "regions": REGIONS,
        "splits": summary,
        "observations_per_region": len(dates),
        "source_files": manifest["files"],
    }
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
