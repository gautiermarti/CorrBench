"""Download declared public PCP regional series; preserve an audited local cache."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from corrbench.neuro import (
    ABIDE_DERIVATIVE,
    ABIDE_PHENOTYPE_SHA256,
    ABIDE_SOURCE,
    eligible_subjects,
    sha256,
)


def fetch_one(row, root):
    filename = f"{row.FILE_ID}_rois_ho.1D"
    path = root / filename
    url = f"{ABIDE_SOURCE}/{ABIDE_DERIVATIVE}/{filename}"
    record = {
        "subject_id": int(row.SUB_ID),
        "site": row.SITE_ID,
        "split": row.split,
        "path": filename,
        "url": url,
    }
    for attempt in range(3):
        try:
            if not path.exists():
                with urlopen(url, timeout=30) as response:
                    payload = response.read(5 * 1024 * 1024 + 1)
                if len(payload) > 5 * 1024 * 1024 or not payload.startswith(b"#"):
                    raise ValueError("Unexpected size or missing ROI header")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
            return {
                **record,
                "status": "available",
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        except Exception as error:
            if attempt == 2:
                return {**record, "status": "failed", "error": str(error)}
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if sha256(args.metadata) != ABIDE_PHENOTYPE_SHA256:
        raise ValueError("Metadata differs from the declared pilot source")
    if not 1 <= args.workers <= 8:
        p.error("--workers must be between 1 and 8")
    frame = eligible_subjects(pd.read_csv(args.metadata))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "download_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        for row in old["files"]:
            if row["status"] == "available":
                if sha256(args.output / row["path"]) != row["sha256"]:
                    raise ValueError(f"Cached source changed: {row['path']}")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fetch_one, row, args.output) for row in frame.itertuples()]
        for job in as_completed(jobs):
            rows.append(job.result())
            state = {
                "source": ABIDE_SOURCE,
                "derivative": ABIDE_DERIVATIVE,
                "metadata_sha256": ABIDE_PHENOTYPE_SHA256,
                "eligible": len(frame),
                "files": sorted(rows, key=lambda x: x["subject_id"]),
            }
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(manifest)
            if len(rows) % 50 == 0 or len(rows) == len(frame):
                print(
                    json.dumps(
                        {
                            "done": len(rows),
                            "total": len(frame),
                            "failed": sum(r["status"] == "failed" for r in rows),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
