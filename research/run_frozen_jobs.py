"""Execute trusted, frozen local experiment commands and persist a restart ledger.

The manifest supplies argument arrays, never shell expressions. Completed jobs
are skipped only if their declared completion artifacts still match the ledger.
An interrupted running job requires explicit recovery; it is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    fingerprint = sha256(args.plan)
    state = {"plan_sha256": fingerprint, "jobs": {}}
    if args.ledger.exists():
        state = json.loads(args.ledger.read_text())
        if state["plan_sha256"] != fingerprint:
            raise ValueError("The plan changed; use a new ledger")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    for job in plan["jobs"]:
        name = job["name"]
        for path, expected in {
            **plan.get("hashes", {}),
            **job.get("hashes", {}),
        }.items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen input changed: {path}")
        previous = state["jobs"].get(name)
        if previous:
            if previous["status"] != "complete":
                raise RuntimeError(
                    f"Explicit recovery required for {name}: {previous['status']}"
                )
            for path, expected in previous["artifacts"].items():
                if sha256(path) != expected:
                    raise ValueError(f"Completed artifact changed: {path}")
            continue
        log_path = args.ledger.parent / f"{name}.log"
        if log_path.exists():
            raise FileExistsError(log_path)
        command = job["command"]
        if not isinstance(command, list) or not all(
            isinstance(x, str) for x in command
        ):
            raise ValueError("Command must be an argument list")
        row = {"status": "running", "started_unix": time.time(), "command": command}
        state["jobs"][name] = row
        save(args.ledger, state)
        print(json.dumps({"job": name, "status": "running"}), flush=True)
        environment = {
            **os.environ,
            **plan.get("environment", {}),
            **job.get("environment", {}),
        }
        with log_path.open("x") as log:
            result = subprocess.run(
                command,
                cwd=job["cwd"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        row["seconds"] = time.time() - row["started_unix"]
        row["returncode"] = result.returncode
        row["status"] = "complete" if result.returncode == 0 else "failed"
        row["log_sha256"] = sha256(log_path)
        if result.returncode == 0:
            try:
                row["artifacts"] = {path: sha256(path) for path in job["artifacts"]}
            except FileNotFoundError:
                row["status"] = "missing_artifact"
                save(args.ledger, state)
                raise
        save(args.ledger, state)
        print(
            json.dumps(
                {"job": name, "status": row["status"], "seconds": row["seconds"]}
            ),
            flush=True,
        )
        if result.returncode:
            raise RuntimeError(f"Job failed; inspect {log_path}")


if __name__ == "__main__":
    main()
