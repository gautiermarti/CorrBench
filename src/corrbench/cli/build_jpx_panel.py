#!/usr/bin/env python3
"""Build a curated point-in-time Japanese equity panel from the JPX archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corrbench.jpx import build_jpx_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_jpx_panel(
        args.archive,
        args.output / "daily_prices.parquet",
        args.output / "asset_metadata.parquet",
        args.output / "manifest.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
