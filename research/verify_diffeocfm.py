"""Smoke-test pinned upstream DiffeoCFM correlation code on synthetic data.

The upstream FastDataloader class is extracted unchanged to avoid importing
unrelated download/EEG dependencies. No dataset loader or restricted SPD code
is executed. This is an integration check, not a matched benchmark result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from corrbench.geometry import (
    corr_to_normalized_cholesky,
    normalized_cholesky_to_corr,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(Path(__file__).with_name("diffeocfm_source.json").read_text())
    root = args.upstream.resolve()
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream file changed: {name}")
    if any(
        name in sys.modules for name in ["data", "diffeo", "fm", "gaussian", "model"]
    ):
        raise RuntimeError("Run in a fresh process to avoid upstream module collisions")
    tree = ast.parse((root / "data.py").read_text())
    node = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "FastDataloader"
    )
    module = types.ModuleType("data")
    module.torch = torch
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), str(root / "data.py"), "exec"
        ),
        module.__dict__,
    )
    sys.modules["data"] = module
    sys.path.insert(0, str(root))
    from diffeo import DiffeoTransform
    from fm import DiffeoCFM
    from gaussian import DiffeoGauss

    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    rng = np.random.default_rng(20261005)
    observations = rng.normal(size=(120, 32, 8))
    matrices = np.stack([np.corrcoef(x.T) for x in observations])
    labels = np.tile(np.arange(3), 40)
    transform = DiffeoTransform("corrcholesky")
    upstream_coordinates = transform.fit_transform(matrices)
    local_coordinates = corr_to_normalized_cholesky(matrices)
    np.testing.assert_allclose(local_coordinates, upstream_coordinates, atol=1e-12)
    restored = transform.inverse_transform(upstream_coordinates)
    local_restored = normalized_cholesky_to_corr(local_coordinates)
    np.testing.assert_allclose(restored, matrices, atol=1e-12)
    np.testing.assert_allclose(local_restored, restored, atol=1e-12)
    config = {
        "DIFFEO": "corrcholesky",
        "FM_TYPE": "classic",
        "DEVICE": "cpu",
        "EPOCHS": 2,
        "WARMUP_EPOCHS": 1,
        "FACTOR_LR": 1,
        "LR": 0.001,
        "HIDDEN_DIM": [32],
        "BATCH_SIZE": 16,
        "PRINT_EVERY": 1,
        "T_GRID": torch.linspace(0, 1, 3, dtype=torch.float64),
    }
    result = {
        "scope": "synthetic 8x8 integration test; not financial or fMRI fidelity",
        "source": manifest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transform_max_difference": float(
            np.max(np.abs(local_coordinates - upstream_coordinates))
        ),
        "inverse_max_error": float(np.max(np.abs(restored - matrices))),
        "versions": {
            p: version(p)
            for p in ["torch", "torchcfm", "torchdiffeq", "pyriemann", "POT"]
        },
        "models": {},
    }
    for cls in [DiffeoGauss, DiffeoCFM]:
        model = cls({**config, "RNG": np.random.RandomState(20261005)})
        history = model.fit(matrices, labels)
        with torch.no_grad():
            samples = model.sample(np.tile(np.arange(3), 4))[-1]
        minimum = float(np.linalg.eigvalsh(samples).min())
        error = float(np.max(np.abs(np.diagonal(samples, axis1=-2, axis2=-1) - 1)))
        if not np.isfinite(samples).all() or minimum <= 0 or error > 1e-12:
            raise AssertionError("Invalid output")
        row = {
            "samples": len(samples),
            "minimum_eigenvalue": minimum,
            "diagonal_error": error,
        }
        if history is not None:
            row["training_loss"] = history["train_loss"].tolist()
            row["validation_loss"] = history["val_loss"].tolist()
        result["models"][cls.__name__] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
