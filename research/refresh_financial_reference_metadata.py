"""Refresh local reference cards without decoding or changing financial arrays."""

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from corrbench.reference import _dataset_card, _validate_metadata


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prior_path = args.root / "docs/financial_reference_metadata_audit.json"
    prior = json.loads(prior_path.read_text())
    audit = {
        "scope": "Metadata-only refresh; no arrays decoded or science rerun",
        "prior_metadata_audit_sha256": sha(prior_path),
        "full_roundtrip_audit_sha256": prior["full_roundtrip_audit_sha256"],
        "configurations": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        for config in ["taiwan_paper", "japan_paper"]:
            metadata = yaml.safe_load(
                (args.root / f"configs/releases/{config}.yaml").read_text()
            )
            _validate_metadata(metadata)
            name = metadata["config_name"]
            source, target = args.source / name, stage / name
            old_path = source / "manifest.json"
            assert sha(old_path) == prior["configurations"][name]["manifest_sha256"]
            assert sha(old_path) == (source / "manifest.sha256").read_text().split()[0]
            old = json.loads(old_path.read_text())
            assert sha(source / "README.md") == old["dataset_card_sha256"]
            # A card update must never change construction/protocol/array meaning.
            allowed = {"rights_status", "rights_evidence", "license", "license_name"}
            assert {k: v for k, v in old["metadata"].items() if k not in allowed} == {
                k: v for k, v in metadata.items() if k not in allowed
            }
            target.mkdir()
            splits = {}
            for split, info in old["splits"].items():
                source_file = (source / info["path"]).resolve()
                assert source_file.is_relative_to(source.resolve())
                expected = prior["configurations"][name]["splits"][split]
                assert (
                    sha(source_file)
                    == info["parquet_sha256"]
                    == expected["parquet_sha256"]
                )
                target_file = target / info["path"]
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                assert sha(target_file) == info["parquet_sha256"]
                splits[split] = {
                    "rows": info["samples"],
                    "parquet_sha256": sha(target_file),
                    "unchanged_from_full_array_audit": True,
                }
            manifest = copy.deepcopy(old)
            manifest["metadata"] = metadata
            manifest["distribution_blocked"] = metadata["rights_status"] != "cleared"
            assert manifest["distribution_blocked"]
            card = target / "README.md"
            card.write_text(_dataset_card(metadata, manifest))
            manifest["dataset_card_sha256"] = sha(card)
            manifest_path = target / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            (target / "manifest.sha256").write_text(
                sha(manifest_path) + "  manifest.json\n"
            )
            audit["configurations"][name] = {
                "rights_status": metadata["rights_status"],
                "distribution_blocked": manifest["distribution_blocked"],
                "manifest_sha256": sha(manifest_path),
                "card_sha256": sha(card),
                "splits": splits,
            }
        audit["total_rows"] = sum(
            s["rows"]
            for c in audit["configurations"].values()
            for s in c["splits"].values()
        )
        (stage / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        stage.rename(args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
