"""Audit and render the complete frozen financial suite without editing papers."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root, data, output = [p.resolve() for p in [args.root, args.data, args.output]]
    if output.exists():
        raise FileExistsError(output)
    plans = {
        family: json.loads((root / f"{family}_plan.json").read_text())
        for family in ["corrflow", "diffeocfm_evaluation", "ccorrgan"]
    }
    last_status = None
    while True:
        status = {}
        for family, plan in plans.items():
            ledger = json.loads((root / f"{family}_ledger.json").read_text())
            jobs = ledger["jobs"]
            names = {job["name"] for job in plan["jobs"]}
            if set(jobs) - names:
                raise ValueError(f"Unexpected ledger jobs: {family}")
            states = [jobs.get(name, {}).get("status") for name in names]
            if any(state in {"failed", "missing_artifact"} for state in states):
                raise RuntimeError(f"Frozen {family} job failed; inspect its log")
            status[family] = {
                "complete": states.count("complete"),
                "planned": len(names),
            }
        if status != last_status:
            print(json.dumps({"stage": "waiting", "queues": status}), flush=True)
            last_status = status
        if all(r["complete"] == r["planned"] for r in status.values()):
            break
        if not args.wait:
            raise RuntimeError(
                "Suite incomplete; use --wait to await all declared jobs"
            )
        time.sleep(30)
    scripts = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    commands = [
        [
            sys.executable,
            str(scripts / "audit_matched_taiwan.py"),
            "--root",
            str(root),
            "--data",
            str(data),
            "--output",
            str(output / "audit"),
            "--require-corrgan",
        ],
        [
            sys.executable,
            str(scripts / "summarize_matched_taiwan.py"),
            "--audit",
            str(output / "audit/audit.json"),
            "--output",
            str(output / "rendered"),
            "--table",
            str(output / "matched_categorical.tex"),
        ],
    ]
    for index, command in enumerate(commands):
        with (output / f"stage_{index}.log").open("w") as handle:
            subprocess.run(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
            )
    report = {
        "stage": "complete",
        "queues": status,
        "commands": commands,
        "scope": "Local complete-suite audit and figures; "
        "papers and releases untouched",
    }
    (output / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
