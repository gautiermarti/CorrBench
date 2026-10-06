import hashlib
import json
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "research" / "run_frozen_jobs.py"


def test_frozen_runner_resumes_only_unchanged_completed_artifacts(tmp_path):
    source = tmp_path / "source"
    source.write_text("frozen")
    output = tmp_path / "output"
    plan = tmp_path / "plan.json"
    ledger = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "hashes": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                },
                "jobs": [
                    {
                        "name": "smoke",
                        "cwd": str(tmp_path),
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; "
                            "Path('output').write_text('result')",
                        ],
                        "artifacts": [str(output)],
                    }
                ],
            }
        )
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--plan",
        str(plan),
        "--ledger",
        str(ledger),
    ]
    subprocess.run(command, check=True, capture_output=True)
    original = ledger.read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert ledger.read_bytes() == original
    output.write_text("changed")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Completed artifact changed" in result.stderr


def test_frozen_runner_records_failed_job_without_automatic_retry(tmp_path):
    plan = tmp_path / "plan.json"
    ledger = tmp_path / "ledger.json"
    plan.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "failure",
                        "cwd": str(tmp_path),
                        "command": [
                            sys.executable,
                            "-c",
                            "raise RuntimeError('intentional')",
                        ],
                        "artifacts": [],
                    }
                ]
            }
        )
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--plan",
        str(plan),
        "--ledger",
        str(ledger),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(ledger.read_text())["jobs"]["failure"]["status"] == "failed"
    result = subprocess.run(command, capture_output=True, text=True)
    assert "Explicit recovery required" in result.stderr
