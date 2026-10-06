"""Execute the packaged starter and metric notebooks in an isolated kernel."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import nbformat
import numpy as np
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", default="corrbench-readiness")
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    for name in ["train.csv", "test.csv"]:
        shutil.copy2(fixture / "public" / name, output / name)
    starter_path, metric_path = [
        fixture / "public" / f"{name}.ipynb" for name in ["starter", "metric"]
    ]
    starter = nbformat.read(starter_path, as_version=4)
    nbformat.validate(starter)
    NotebookClient(
        starter,
        kernel_name=args.kernel,
        timeout=180,
        resources={"metadata": {"path": str(output)}},
    ).execute()
    nbformat.write(starter, output / "starter_executed.ipynb")
    metric = nbformat.read(metric_path, as_version=4)
    nbformat.validate(metric)
    solution_path = fixture / "organizer_only/solution.csv"
    test_source = f"""
import json
import platform
solution = pd.read_csv({str(solution_path)!r})
submission = pd.read_csv("submission.csv")
scores = {{}}
for usage in ["Public", "Private"]:
    mask = solution["Usage"] == usage
    scores[usage] = score(solution[mask], submission[mask], "id")
with open("kernel_scores.json", "w") as handle:
    json.dump({{"scores": scores, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__}}, handle)
scores
"""
    metric.cells.append(nbformat.v4.new_code_cell(test_source))
    NotebookClient(
        metric,
        kernel_name=args.kernel,
        timeout=180,
        resources={"metadata": {"path": str(output)}},
    ).execute()
    nbformat.write(metric, output / "metric_executed_organizer_only.ipynb")
    result = json.loads((output / "kernel_scores.json").read_text())
    expected = json.loads((fixture / "audit.json").read_text())["scores"][
        "conditional_replay"
    ]
    for split in ["Public", "Private"]:
        np.testing.assert_allclose(result["scores"][split], expected[split], atol=1e-12)
    record = {
        "status": "passed",
        "scope": "Local isolated Jupyter execution, not the hosted Kaggle sandbox",
        "starter_reads_public_files_only": True,
        "starter_and_metric_schema_valid": True,
        "csv_scores_match_standalone_audit": True,
        "kernel": result,
        "notebook_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [starter_path, metric_path]
        },
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
