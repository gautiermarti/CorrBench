import json

import pytest

from corrbench.leaderboard import validate_leaderboard
from corrbench.reference import sha256


def bundle(tmp_path):
    (tmp_path / "protocol.md").write_text("Frozen example protocol")
    (tmp_path / "evidence.json").write_text('{"score": 0.3}')
    protocol = {
        "id": "example",
        "direction": "lower",
        "metric": "distance",
        "protocol_file": "protocol.md",
        "protocol_sha256": sha256(tmp_path / "protocol.md"),
        "evidence_file": "evidence.json",
        "dataset_hash": "a" * 64,
        "split": "test",
        "metric_version": "v1",
    }
    row = {
        **{
            k: protocol[k]
            for k in ["protocol_sha256", "dataset_hash", "split", "metric_version"]
        },
        "id": "method",
        "protocol": "example",
        "status": "locally_audited",
        "samples": 100,
        "training_seeds": 5,
        "generation_seeds": 5,
        "metrics": {"distance": 0.3},
        "evidence_sha256": sha256(tmp_path / "evidence.json"),
    }
    return {
        "schema_version": "corrbench.leaderboard.v1",
        "protocols": [protocol],
        "records": [row],
    }


def test_valid_evidence_bundle(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(bundle(tmp_path)))
    assert validate_leaderboard(tmp_path)["records"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "data",
        "split",
        "version",
        "evidence",
        "nonfinite",
        "verification",
        "duplicate",
        "outside_path",
        "negative_count",
    ],
)
def test_incompatible_or_unsubstantiated_records_rejected(tmp_path, corruption):
    payload = bundle(tmp_path)
    row, protocol = payload["records"][0], payload["protocols"][0]
    if corruption == "data":
        row["dataset_hash"] = "b" * 64
    elif corruption == "split":
        row["split"] = "validation"
    elif corruption == "version":
        row["metric_version"] = "v2"
    elif corruption == "evidence":
        (tmp_path / "evidence.json").write_text('{"score": 100}')
    elif corruption == "nonfinite":
        row["metrics"]["distance"] = float("nan")
    elif corruption == "verification":
        row["status"] = "independently_verified"
    elif corruption == "duplicate":
        payload["records"].append(dict(row))
    elif corruption == "outside_path":
        protocol["protocol_file"] = "../outside.md"
    elif corruption == "negative_count":
        row["training_seeds"] = -1
    (tmp_path / "results.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_leaderboard(tmp_path)
