"""Download the frozen 1981--2024 NOAA PSL daily 2 m temperature source files."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from corrbench.neuro import sha256

BASE = (
    "https://psl.noaa.gov/thredds/fileServer/Datasets/"
    "ncep.reanalysis/Dailies/surface_gauss"
)


def fetch(year, output, previous):
    filename = f"air.2m.gauss.{year}.nc"
    path, url = output / filename, f"{BASE}/{filename}"
    if path.exists():
        expected = previous.get(str(year))
        if not expected or sha256(path) != expected["sha256"]:
            raise ValueError(f"Unverified existing source: {path}")
        return expected
    temporary = path.with_suffix(".nc.partial")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=60) as response:
                expected_bytes = int(response.headers["Content-Length"])
                last_modified = response.headers.get("Last-Modified")
            if not 1_000_000 < expected_bytes < 100_000_000:
                raise ValueError("Unexpected NetCDF source size")
            # Large single responses are being truncated by the transfer path.
            # Verified byte ranges avoid accepting a clean EOF as a complete file.
            with temporary.open("wb") as handle:
                for start in range(0, expected_bytes, 2 * 1024 * 1024):
                    end = min(start + 2 * 1024 * 1024, expected_bytes) - 1
                    request = urllib.request.Request(
                        url, headers={"Range": f"bytes={start}-{end}"}
                    )
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status != 206:
                            raise ValueError("Source did not honor byte range")
                        expected_range = f"bytes {start}-{end}/{expected_bytes}"
                        if response.headers.get("Content-Range") != expected_range:
                            raise ValueError("Unexpected source byte range")
                        chunk = response.read()
                    if len(chunk) != end - start + 1:
                        raise ValueError("Truncated byte range")
                    handle.write(chunk)
            if temporary.stat().st_size != expected_bytes:
                raise ValueError("Truncated NetCDF download")
            with temporary.open("rb") as handle:
                if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                    raise ValueError("Expected NetCDF4/HDF5 source")
            record = {
                "year": year,
                "url": url,
                "filename": filename,
                "bytes": temporary.stat().st_size,
                "expected_content_length": expected_bytes,
                "last_modified": last_modified,
                "transfer": "verified contiguous HTTP byte ranges",
                "sha256": sha256(temporary),
                "retrieved_utc": datetime.now(UTC).isoformat(),
            }
            temporary.replace(path)
            return record
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "download_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"source": BASE, "files": {}}
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {
            pool.submit(fetch, year, args.output, manifest["files"]): year
            for year in range(1981, 2025)
        }
        for job in as_completed(jobs):
            result = job.result()
            manifest["files"][str(result["year"])] = result
            temporary = manifest_path.with_suffix(".json.partial")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(manifest_path)
            print(result["year"], result["bytes"], result["sha256"], flush=True)


if __name__ == "__main__":
    main()
