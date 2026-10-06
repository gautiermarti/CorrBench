"""Build all three packages and audit an isolated wheel/sdist installation.

Run with a disposable CPU environment that already has the declared scientific
and test dependencies, build and setuptools. This never changes the environment
used by running scientific jobs. It does not verify GitLab CI or CUDA training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.models_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    environment.pop("PYTHONPATH", None)
    python = str(Path(sys.executable).absolute())
    names = ["corrbench", "corrflow", "corrgan"]
    logs = {}

    def run(name, command, cwd=output):
        path = output / f"{name}.log"
        with path.open("w") as handle:
            subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        logs[path.name] = sha(path)
        print(name + " passed", flush=True)

    for name in names:
        run(
            "build_" + name,
            [python, "-m", "build", "--outdir", str(output / "dist" / name)],
            root / name,
        )
    wheels = [next((output / "dist" / name).glob("*.whl")) for name in names]
    run(
        "install",
        [
            str(args.uv),
            "pip",
            "install",
            "--python",
            python,
            "--reinstall",
            "--no-deps",
            *map(str, wheels),
        ],
    )
    run("dependencies", [str(args.uv), "pip", "check", "--python", python])
    archives = output / "source_archives"
    archives.mkdir()
    # A release may be staged inside a research checkout whose pytest config
    # injects its own src/ onto sys.path. Freeze the test configuration explicitly
    # so the wheel audit cannot import a parent checkout instead of the wheel.
    pytest_config = output / "pytest.ini"
    pytest_config.write_text("[pytest]\naddopts = -ra\n")
    packaged_images = {}
    for name in names:
        archive = next((output / "dist" / name).glob("*.tar.gz"))
        with tarfile.open(archive) as tar:
            tar.extractall(archives, filter="data")
        extracted = next(archives.glob(name + "-*"))
        images = re.findall(
            r"!\[[^\]]*\]\(([^)]+)\)", (extracted / "README.md").read_text()
        )
        packaged_images[name] = {}
        for relative in images:
            if "://" in relative:
                continue
            source = root / name / relative
            packaged = extracted / relative
            assert packaged.is_file(), (
                f"README image missing from source archive: {packaged}"
            )
            assert sha(source) == sha(packaged), str(packaged)
            packaged_images[name][relative] = sha(packaged)
    for variant, testpaths in [
        ("installed_wheels", [root / name / "tests" for name in names]),
        (
            "extracted_source_archives",
            [next(archives.glob(name + "-*")) / "tests" for name in names],
        ),
    ]:
        if variant == "extracted_source_archives":
            sdists = [next((output / "dist" / name).glob("*.tar.gz")) for name in names]
            run(
                "install_source_archives",
                [
                    str(args.uv),
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--reinstall",
                    "--no-deps",
                    "--no-build-isolation",
                    *map(str, sdists),
                ],
            )
            run(
                "source_dependencies",
                [str(args.uv), "pip", "check", "--python", python],
            )
        run(
            variant,
            [
                python,
                "-m",
                "pytest",
                "-c",
                str(pytest_config),
                "--import-mode=importlib",
                "-q",
                "--junitxml",
                str(output / f"{variant}.xml"),
                *map(str, testpaths),
            ],
        )
    console_commands = {}
    for name in names:
        commands = sorted(
            point.name
            for point in importlib.metadata.distribution(name).entry_points
            if point.group == "console_scripts"
        )
        console_commands[name] = commands
        for command in commands:
            run(
                "entrypoint_" + command,
                [str(Path(sys.prefix) / "bin" / command), "--help"],
            )
    for name, module in [
        ("weather", "corrbench.cli.build_weather_dataset"),
        ("domain_export", "corrbench.cli.export_domain_reference"),
        ("leaderboard", "corrbench.cli.validate_leaderboard"),
        ("unconditional_training", "corrgan.cli.train_unconditional"),
        ("unconditional_sampling", "corrgan.cli.sample_unconditional"),
        ("diffeocfm", "corrflow.cli.train_diffeocfm"),
        ("node_training", "corrflow.cli.train_neuro_flow"),
        ("node_sampling", "corrflow.cli.sample_neuro_augmentation"),
    ]:
        run("cli_" + name, [python, "-m", module, "--help"])
    run(
        "leaderboard_bundle",
        [
            python,
            "-m",
            "corrbench.cli.validate_leaderboard",
            "--directory",
            str(root / "corrbench/leaderboard"),
        ],
    )
    # Resolve imports in a fresh interpreter after installation; the validator
    # itself has no package imports and cannot hide a stale imported module.
    command = (
        "import importlib,json; from pathlib import Path; "
        "print(json.dumps({n:str(Path(importlib.import_module(n).__file__).parent) "
        "for n in ['corrbench','corrflow','corrgan']}))"
    )
    installed = json.loads(
        subprocess.check_output(
            [python, "-c", command], cwd=output, env=environment, text=True
        )
    )
    packages = {}
    for name in names:
        destination = Path(installed[name])
        assert destination.is_relative_to(Path(sys.prefix)), destination
        hashes = {}
        for source in (root / name / "src" / name).rglob("*.py"):
            relative = source.relative_to(root / name / "src" / name)
            assert sha(source) == sha(destination / relative), str(relative)
            extracted_source = (
                next(archives.glob(name + "-*")) / "src" / name / relative
            )
            assert sha(extracted_source) == sha(destination / relative), str(relative)
            hashes[str(relative)] = sha(source)
        packages[name] = {
            "version": importlib.metadata.version(name),
            "installed_source_matches_checkout": True,
            "source_sha256": hashes,
            "checkout_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root / name, text=True
            ).strip(),
            "checkout_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root / name, text=True
                )
            ),
            "distributions": {
                p.name: sha(p) for p in (output / "dist" / name).iterdir()
            },
        }
    import xml.etree.ElementTree as et

    tests = {}
    for variant in ["installed_wheels", "extracted_source_archives"]:
        suites = et.parse(output / f"{variant}.xml").getroot().findall("testsuite")
        assert all(
            int(s.get("failures", 0)) + int(s.get("errors", 0)) == 0 for s in suites
        )
        assert all(int(s.get("skipped", 0)) == 0 for s in suites)
        tests[variant] = sum(int(s.get("tests", 0)) for s in suites)
    dependencies = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scipy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "torch",
            "torchdiffeq",
            "netCDF4",
            "nibabel",
            "pytest",
        ]
    }
    result = {
        "scope": "Isolated CPU installed-wheel and source-archive tests; "
        "not remote CI or clean-machine CUDA retraining",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tests": tests,
        "source_archives_installed_before_second_test_pass": True,
        "console_commands_checked": console_commands,
        "dependencies": dependencies,
        "packages": packages,
        "readme_images_verified_in_source_archives": packaged_images,
        "test_log_sha256": logs,
        "validation_code_sha256": sha(Path(__file__)),
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "tests": tests}), flush=True)


if __name__ == "__main__":
    main()
