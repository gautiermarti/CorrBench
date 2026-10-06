"""Audit complete five-seed categorical suites before producing a comparison table."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from corrbench.geometry import nearest_correlation_torch
from corrbench.metrics import sliced_wasserstein, structural_features
from corrbench.reference import sha256


def read(path):
    return json.loads(path.read_text())


def selected_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
            for c in range(3)
        ]
    )


def get_score(row):
    return row.get(
        "validation_feature_swd", row.get("structural", {}).get("feature_swd")
    )


def get_metrics(report):
    pooled = report["pooled"]
    return {
        "feature_swd": pooled["feature_sliced_wasserstein"],
        "feature_energy": pooled["feature_energy_distance"],
        "gft_swd": pooled["gft_sliced_wasserstein"],
        "orbit_gft_swd": pooled["orbit_gft_sliced_wasserstein"],
        "portfolio_swd": pooled["portfolio_sliced_wasserstein"],
        "precision": pooled["feature_manifold"]["precision"],
        "recall": pooled["feature_manifold"]["recall"],
        "classifier_auc": pooled["feature_classifier_two_sample"]["auc"],
        "nearest_train_feature_exact_fraction": pooled["nearest_train"][
            "generated_exact_match_fraction"
        ],
        "valid_fraction": pooled["validity"]["valid_fraction"],
        **{
            f"class_{c}_feature_swd": report["by_class"][str(c)][
                "feature_sliced_wasserstein"
            ]
            for c in range(3)
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-corrgan", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    plans, ledgers, hashes, states = {}, {}, {}, {}
    for family in ["corrflow", "diffeocfm_evaluation", "ccorrgan"]:
        plan_path = args.root / f"{family}_plan.json"
        ledger_path = args.root / f"{family}_ledger.json"
        plan, ledger = read(plan_path), read(ledger_path)
        assert sha256(plan_path) == ledger["plan_sha256"], family
        plans[family], ledgers[family] = plan, ledger
        jobs = ledger["jobs"]
        complete = len(jobs) == len(plan["jobs"]) and all(
            job["status"] == "complete" for job in jobs.values()
        )
        if not complete and (family != "ccorrgan" or args.require_corrgan):
            raise ValueError(f"Required suite incomplete: {family}")
        states[family] = {
            "complete": complete,
            "completed_jobs": sum(j["status"] == "complete" for j in jobs.values()),
            "planned_jobs": len(plan["jobs"]),
            "plan_sha256": sha256(plan_path),
            "ledger_sha256": sha256(ledger_path),
        }
        for path, expected in plan["hashes"].items():
            if path in hashes and hashes[path] != expected:
                raise ValueError(f"Inconsistent frozen source hash: {path}")
            hashes[path] = expected
        for job in jobs.values():
            if job["status"] == "complete":
                for path, expected in job["artifacts"].items():
                    assert sha256(Path(path)) == expected, path
    for path, expected in hashes.items():
        assert sha256(Path(path)) == expected, path
    arrays = {}
    for split in ["train", "validation", "test"]:
        with np.load(args.data / f"{split}.npz") as a:
            arrays[split] = {k: a[k] for k in ["matrices", "labels"]}
    reference_indices = {
        "validation": selected_indices(arrays["validation"]["labels"], 100, 44301),
        "test": selected_indices(arrays["test"]["labels"], 500, 2027),
    }
    features, _ = structural_features(arrays["train"]["matrices"])
    feature_scale = np.maximum(features.std(axis=0, ddof=1), 1e-6)
    validation_features, _ = structural_features(
        arrays["validation"]["matrices"][reference_indices["validation"]]
    )
    families = ["corrflow", "diffeocfm"]
    if states["ccorrgan"]["complete"]:
        families.append("ccorrgan")
    rows, selection_checks = [], []

    def audit_sample(stem, split):
        source = args.root / f"{stem}.npz"
        report_path = args.root / f"{stem}_metrics.json"
        scored = args.root / f"{stem}_metrics.npz"
        report = read(report_path)
        with np.load(source) as sample, np.load(scored) as evaluated:
            for a in [sample, evaluated]:
                np.testing.assert_array_equal(
                    a["selected_test_indices"], reference_indices[split]
                )
                np.testing.assert_array_equal(
                    a["labels"], arrays[split]["labels"][reference_indices[split]]
                )
            assert str(sample["split"].item()) == split
            assert len(sample["matrices"]) == (300 if split == "validation" else 1500)
            assert np.isfinite(sample["matrices"]).all()
            assert report["samples"] == len(sample["matrices"])
            if "replay_train_indices" in sample:
                replay_indices = sample["replay_train_indices"]
                np.testing.assert_array_equal(
                    sample["matrices"], arrays["train"]["matrices"][replay_indices]
                )
                np.testing.assert_array_equal(
                    sample["labels"], arrays["train"]["labels"][replay_indices]
                )
            if report["repair"] is not None:
                assert report["repair"]["all_converged"]
            return report, {
                "sample_sha256": sha256(source),
                "score_sha256": sha256(report_path),
                "scored_arrays_sha256": sha256(scored),
                "epoch": int(sample["checkpoint_epoch"])
                if "checkpoint_epoch" in sample
                else 0,
            }

    for family in families:
        for seed in range(1701, 1706):
            run = args.root / f"{family}_seed{seed}"
            if family == "diffeocfm" and seed == 1701:
                run = args.root.parent / "diffeocfm_taiwan_200_seed1701_20261005"
            history = read(run / "history.json")
            evaluated = [r for r in history if get_score(r) is not None]
            chosen = min(evaluated, key=get_score)
            checkpoint = torch.load(
                run / "best_structural.pt", map_location="cpu", weights_only=False
            )
            assert checkpoint["epoch"] == chosen["epoch"]
            expected_epochs = 20 if family == "corrflow" else 200
            assert history[-1]["epoch"] == expected_epochs
            stem = f"{family}_seed{seed}_best_structural_validation"
            common, metadata = audit_sample(stem, "validation")
            with np.load(args.root / f"{stem}.npz") as a:
                native = a["matrices"]
            if family == "ccorrgan":
                upper = np.triu(native, 1)
                prepared = upper + upper.swapaxes(-1, -2) + np.eye(80, dtype=np.float32)
                repaired = []
                for start in range(0, len(prepared), 32):
                    valid, info = nearest_correlation_torch(
                        torch.from_numpy(prepared[start : start + 32]).double(),
                        max_iterations=1000,
                        tolerance=1e-6,
                        return_info=True,
                    )
                    assert bool(info["converged"].all())
                    repaired.append(valid.numpy())
                native = np.concatenate(repaired)
            native_features, _ = structural_features(native)
            native_score = sliced_wasserstein(
                validation_features,
                native_features,
                scale=feature_scale,
                seed=44301,
                device="cpu",
            )
            assert abs(native_score - get_score(chosen)) < 2e-5, (family, seed)
            assert metadata["epoch"] == chosen["epoch"]
            selection_checks.append(
                {
                    "method": family,
                    "training_seed": seed,
                    "epoch": chosen["epoch"],
                    "training_validation_swd": get_score(chosen),
                    "reproduced_native_validation_swd": native_score,
                    "common_float32_validation_swd": common["pooled"][
                        "feature_sliced_wasserstein"
                    ],
                    "precision_note": "Native selection differs from float32 scoring",
                    "checkpoint_sha256": sha256(run / "best_structural.pt"),
                }
            )
            for weights in ["best_structural", "last"]:
                stem = f"{family}_seed{seed}_{weights}_test"
                report, artifact = audit_sample(stem, "test")
                assert artifact["epoch"] == (
                    expected_epochs if weights == "last" else chosen["epoch"]
                )
                rows.append(
                    {
                        "method": family,
                        "weights": weights,
                        "training_seed": seed,
                        "generation_seed": 2027 + seed - 1701,
                        **artifact,
                        **get_metrics(report),
                        "sampling_seconds_concurrent": report["sampling_seconds"],
                        "repair_seconds_concurrent": (
                            report["repair"]["seconds"] if report["repair"] else 0
                        ),
                    }
                )
    for offset, seed in enumerate(range(1701, 1706)):
        for method, stem in [
            ("gaussian", f"diffeocfm_seed{seed}_gaussian_test"),
            ("replay", f"categorical_replay_seed{2027 + offset}_test"),
        ]:
            report, artifact = audit_sample(stem, "test")
            rows.append(
                {
                    "method": method,
                    "weights": "baseline",
                    "training_seed": None,
                    "generation_seed": 2027 + offset,
                    **artifact,
                    **get_metrics(report),
                }
            )
    summary = []
    metrics = list(get_metrics(report))
    for method, weights in sorted({(r["method"], r["weights"]) for r in rows}):
        selected = [
            r for r in rows if r["method"] == method and r["weights"] == weights
        ]
        assert len(selected) == 5
        summary.append(
            {
                "method": method,
                "weights": weights,
                "replicates": 5,
                "uncertainty": (
                    "training and paired generation variability"
                    if method in families
                    else "generation variability; same fitted training data"
                ),
                "metrics": {
                    k: {
                        "mean": float(np.mean([r[k] for r in selected])),
                        "sd": float(np.std([r[k] for r in selected], ddof=1)),
                    }
                    for k in metrics
                },
            }
        )
    result = {
        "utc": datetime.now(UTC).isoformat(),
        "protocol": "matched_taiwan_categorical_20261005",
        "scope": "Retrospective, previously inspected Taiwan periods; categorical-only",
        "training_budgets": {"corrflow": 20, "diffeocfm": 200, "ccorrgan": 200},
        "equal_compute": False,
        "case_selection": "identical 1500 test rows; 500 per regime; seed 2027",
        "timing_limit": "Concurrent execution; do not use these times to rank speed",
        "seed_uncertainty_limit": "Not uncertainty over independent market episodes",
        "states": states,
        "frozen_sources_verified": hashes,
        "source_data_sha256": {s: sha256(args.data / f"{s}.npz") for s in arrays},
        "test_indices": reference_indices["test"].tolist(),
        "validation_selection_checks": selection_checks,
        "rows": rows,
        "summary": summary,
    }
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
