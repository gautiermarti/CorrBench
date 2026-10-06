"""Integrity and comparability checks for curated static leaderboard records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from corrbench.reference import sha256


def _file(directory, name):
    if not isinstance(name, str):
        raise ValueError("An evidence/protocol file path is required")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Evidence must be an existing file inside the bundle")
    return path


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 fingerprint")


def validate_leaderboard(directory: Path):
    """Reject incompatible or corrupted records; do not certify scientific truth."""
    payload = json.loads((directory / "results.json").read_text())
    if payload["schema_version"] != "corrbench.leaderboard.v1":
        raise ValueError("Unsupported leaderboard schema")
    protocols = {p["id"]: p for p in payload["protocols"]}
    if len(protocols) != len(payload["protocols"]):
        raise ValueError("Duplicate protocol IDs")
    ids = [r["id"] for r in payload["records"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate result IDs")
    evidence = {}
    for key, protocol in protocols.items():
        if protocol["direction"] not in {"higher", "lower"}:
            raise ValueError("Unknown metric direction")
        path = _file(directory, protocol["protocol_file"])
        if sha256(path) != protocol["protocol_sha256"]:
            raise ValueError("Protocol file changed")
        if protocol["evidence_file"]:
            evidence[key] = sha256(_file(directory, protocol["evidence_file"]))
            _hash(protocol["dataset_hash"])
        elif any(r["protocol"] == key for r in payload["records"]):
            raise ValueError("A protocol with results requires evidence")
    for row in payload["records"]:
        protocol = protocols.get(row["protocol"])
        if protocol is None:
            raise ValueError("Unknown result protocol")
        for field in ["dataset_hash", "split", "metric_version", "protocol_sha256"]:
            if row[field] != protocol[field]:
                raise ValueError(f"Incompatible result {field}")
        if row["evidence_sha256"] != evidence[row["protocol"]]:
            raise ValueError("Result evidence fingerprint differs")
        # External verification needs a future explicit review/receipt contract.
        if row["status"] not in {"locally_audited", "author_reported"}:
            raise ValueError("Unsupported verification claim")
        for field in ["samples", "training_seeds", "generation_seeds"]:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Counts must be nonnegative integers")
        if row["samples"] == 0:
            raise ValueError("A scored result requires reference samples")
        if protocol["metric"] not in row["metrics"]:
            raise ValueError("Missing primary metric")
        for value in row["metrics"].values():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError("Metrics must be finite numbers")
    return {
        "status": "passed",
        "protocols": len(protocols),
        "records": len(ids),
        "scope": "Bundle integrity and protocol comparability; no external "
        "scientific verification inferred",
        "results_sha256": sha256(directory / "results.json"),
    }
