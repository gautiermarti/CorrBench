"""Stage lossless brain/weather reference data locally without uploading."""

import argparse
import json
from pathlib import Path

import yaml

from corrbench.domain_reference import FIELDS, export_domain_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--adapter", choices=FIELDS, required=True)
    p.add_argument(
        "--purpose",
        choices=["paper_reproduction", "competition_training"],
        default="paper_reproduction",
    )
    a = p.parse_args()
    result = export_domain_reference(
        a.data,
        a.output,
        yaml.safe_load(a.metadata.read_text()),
        adapter=a.adapter,
        purpose=a.purpose,
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "distribution_blocked": result["distribution_blocked"],
                "splits": {s: r["samples"] for s, r in result["splits"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
