"""Stage an HF-compatible reference bundle locally; never uploads data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from corrbench.reference import PURPOSES, export_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=PURPOSES, default="paper_reproduction")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    metadata = yaml.safe_load(args.metadata.read_text())
    if not isinstance(metadata, dict):
        parser.error("metadata must be a YAML mapping")
    manifest = export_reference(
        args.data,
        args.output,
        metadata,
        purpose=args.purpose,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "distribution_blocked": manifest["distribution_blocked"],
                "samples": {k: v["samples"] for k, v in manifest["splits"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
