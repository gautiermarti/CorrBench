"""Build a static, evidence-linked research preview from audited local reports."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from corrbench.leaderboard import validate_leaderboard
from corrbench.reference import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("leaderboard"))
    p.add_argument("--matched-audit", type=Path)
    args = p.parse_args()
    root = Path(__file__).parents[1]
    if args.matched_audit is None:
        saved = root / "docs/matched_taiwan_results.json"
        if saved.exists():
            args.matched_audit = saved
    destination = args.output
    for name in ["evidence", "protocols", "datasets"]:
        (destination / name).mkdir(parents=True, exist_ok=True)
    for name in [
        "abide_atlas_audit_20261006.md",
        "abide_atlas_audit_20261006.json",
        "abide_release_review.md",
        "abide_named110_results_20261006.md",
        "abide_named110_results_20261006.json",
    ]:
        shutil.copy2(root / "docs" / name, destination / "protocols" / name)
    shutil.copy2(
        root / "docs/abide_atlas_audit_20261006.json",
        destination / "evidence/abide_atlas_audit_20261006.json",
    )
    for name in [
        "fmri_pilot_protocol",
        "weather_pilot_protocol",
        "weather_generation_protocol",
        "matched_taiwan_protocol",
    ]:
        shutil.copy2(root / f"docs/{name}.md", destination / f"protocols/{name}.md")
    fmri_path = root / "docs/fmri_pilot_results.json"
    fmri = json.loads(fmri_path.read_text())
    weather_path = root / "docs/weather_pilot_results.json"
    weather = json.loads(weather_path.read_text())
    for path in [fmri_path, weather_path]:
        shutil.copy2(path, destination / "evidence" / path.name)
    records = []
    protocols = [
        {
            "id": "abide_utility_20261005",
            "domain": "Brain connectivity",
            "title": "ABIDE · augmentation at a held-out site",
            "description": (
                "111 source labels (110 named regions and one unnamed label). "
                "Train: 349 subjects; validation: 92 Michigan "
                "subjects; test: 160 NYU subjects. Generator selection and classifier "
                "tuning use validation only."
            ),
            "status": "Local pilot complete",
            "finding": "No established augmentation gain over real-only training.",
            "limitations": (
                "One generator training seed; five generation seeds. Subject-bootstrap "
                "intervals condition on one site and fitted models. No clinical claim."
                " Post-pilot atlas audit: unnamed two-voxel label 3455 alone caused "
                "161 of 179 exclusions. Original pilot preserved."
            ),
            "metric": "auc",
            "direction": "higher",
            "metric_label": "Test ROC-AUC ↑",
            "columns": [
                ["auc", "Test AUC ↑"],
                ["generation_sd", "Generation SD"],
                ["balanced_accuracy", "Balanced accuracy ↑"],
                ["structure_swd", "Structure SWD ↓"],
                ["roi_swd", "Named-ROI SWD ↓"],
            ],
            "protocol_file": "protocols/fmri_pilot_protocol.md",
            "evidence_file": "evidence/fmri_pilot_results.json",
            "dataset_hash": json.loads(
                (root / "docs/fmri_reconstruction_audit.json").read_text()
            )["splits"]["test"]["sha256"],
            "metric_version": "abide-logistic-utility-v1",
            "split": "test",
        },
    ]
    base = {
        "protocol": protocols[0]["id"],
        "status": "locally_audited",
        "evidence_sha256": sha256(fmri_path),
        "samples": 160,
    }
    records.append(
        {
            **base,
            "id": "abide-real-only",
            "method": "Real only",
            "training_seeds": 0,
            "generation_seeds": 0,
            "metrics": {
                "auc": fmri["reference"]["roc_auc"],
                "balanced_accuracy": fmri["reference"]["balanced_accuracy"],
            },
            "note": "Reference classifier; no synthetic augmentation.",
        }
    )
    for row in fmri["methods"]:
        records.append(
            {
                **base,
                "id": "abide-" + row["method"],
                "method": row["label"],
                "training_seeds": row["training_seeds"],
                "generation_seeds": 5,
                "metrics": {
                    "auc": row["auc_mean"],
                    "generation_sd": row["auc_generation_sd"],
                    "balanced_accuracy": row["balanced_accuracy_mean"],
                    "structure_swd": row["structure_swd_mean"],
                    "roi_swd": row["roi_swd_mean"],
                },
                "note": (
                    "Mean AUC difference from real only: "
                    f"{row['auc_difference']:+.4f}; paired subject 95% interval "
                    f"{row['auc_difference_paired_subject_95pct']}. "
                    "Unadjusted for multiplicity."
                ),
            }
        )
    replication_path = root / "docs/fmri_replication_results.json"
    if replication_path.exists():
        replication = json.loads(replication_path.read_text())
        assert len(replication["methods"]) == 4
        assert all(r["training_seeds"] == 5 for r in replication["methods"])
        shutil.copy2(replication_path, destination / "evidence" / replication_path.name)
        shutil.copy2(
            root / "docs/fmri_replication_protocol.md",
            destination / "protocols/fmri_replication_protocol.md",
        )
        protocol = {
            "id": "abide_training_replication_20261005",
            "domain": "Brain connectivity",
            "title": "ABIDE · five training replications",
            "description": "Same 111 source labels and subject/site splits. "
            "Each of five learned-model fits averages five generation draws.",
            "status": "Local exploratory replication complete",
            "finding": "Inspect training variability and paired utility differences "
            "alongside average AUC.",
            "limitations": replication["uncertainty"] + " The initial NYU results "
            "were known before expanding training seeds. No new site holdout. "
            "Post-pilot atlas audit: unnamed two-voxel label 3455 alone caused "
            "161 of 179 exclusions; frozen population preserved.",
            "metric": "auc",
            "direction": "higher",
            "metric_label": "Test ROC-AUC ↑",
            "columns": [
                ["auc", "Test AUC ↑"],
                ["training_sd", "Training SD"],
                ["auc_difference", "Δ real-only AUC"],
                ["balanced_accuracy", "Balanced accuracy ↑"],
            ],
            "protocol_file": "protocols/fmri_replication_protocol.md",
            "evidence_file": "evidence/fmri_replication_results.json",
            "dataset_hash": replication["dataset_sha256"],
            "metric_version": "abide-logistic-utility-five-training-v1",
            "split": "test",
        }
        protocols.append(protocol)
        base = {
            "protocol": protocol["id"],
            "status": "locally_audited",
            "evidence_sha256": sha256(replication_path),
            "samples": replication["test_subjects"],
        }
        reference = replication["reference"]
        records.append(
            {
                **base,
                "id": "abide-replication-real-only",
                "method": "Real only",
                "training_seeds": 0,
                "generation_seeds": 0,
                "metrics": {
                    "auc": reference["roc_auc"],
                    "auc_difference": 0.0,
                    "balanced_accuracy": reference["balanced_accuracy"],
                },
                "note": "Identical reference probabilities checked across all queues.",
            }
        )
        for row in replication["controls"] + replication["methods"]:
            repeated = "auc_training_sd" in row
            metrics = {
                "auc": row["auc_mean"],
                "auc_difference": row["auc_difference"],
                "balanced_accuracy": row["balanced_accuracy_mean"],
            }
            if repeated:
                metrics["training_sd"] = row["auc_training_sd"]
                note = "Five training fits × five generation seeds. Joint paired "
                note += "95% AUC-difference range "
                note += str(row["training_and_paired_subject_95pct"])
                note += "; one inspected site, unadjusted for multiplicity."
            else:
                note = "Fixed training-data control from the original pilot, "
                note += "five generation seeds; not five independent learned fits."
            records.append(
                {
                    **base,
                    "id": "abide-replication-" + row["method"],
                    "method": row["label"],
                    "training_seeds": row["training_seeds"],
                    "generation_seeds": 5,
                    "metrics": metrics,
                    "note": note,
                }
            )
    for split, title, description in [
        (
            "test",
            "Weather · future years in three regions",
            "64 cells per patch. Train 1981–2010; validate 2011–2015; test 2016–2024.",
        ),
        (
            "spatial_transfer",
            "Weather · calibrated South America transfer",
            "Unseen region, 2016–2024; local 1981–2010 climatology calibration.",
        ),
    ]:
        protocol_id = "weather_" + split + "_20261005"
        protocols.append(
            {
                "id": protocol_id,
                "domain": "Weather",
                "title": title,
                "description": description,
                "status": "Baseline pilot complete",
                "finding": (
                    "Historical replay is a strong reference for aggregate dependence."
                    if split == "test"
                    else "Geographical transfer is harder in this pilot."
                ),
                "limitations": (
                    "Nine year groups; regions and quarters are dependent. Unit local "
                    "variance and reanalysis estimates; no forecast claim. "
                    "Bootstrap ranges are descriptive and can be biased."
                ),
                "metric": "aggregate_w1",
                "direction": "lower",
                "metric_label": "Normalized aggregate W1 ↓",
                "columns": [
                    ["aggregate_w1", "Aggregate W1 ↓"],
                    ["year_groups", "Year groups"],
                    ["contexts", "Region × quarter"],
                ],
                "protocol_file": "protocols/weather_pilot_protocol.md",
                "evidence_file": "evidence/weather_pilot_results.json",
                "dataset_hash": weather["source_sha256"][split],
                "metric_version": "weather-aggregate-variance-w1-v1",
                "split": split,
            }
        )
        for method, row in weather["summary"][split].items():
            records.append(
                {
                    "id": f"weather-{split}-{method}",
                    "protocol": protocol_id,
                    "method": method.replace("_", " ").capitalize(),
                    "status": "locally_audited",
                    "evidence_sha256": sha256(weather_path),
                    "samples": weather["dataset"][split]["matrices"],
                    "training_seeds": 0,
                    "generation_seeds": 1 if method == "conditional_replay" else 0,
                    "metrics": {
                        "aggregate_w1": row["normalized_aggregate_w1"],
                        "year_groups": row["year_groups"],
                        "contexts": row["contexts"],
                    },
                    "note": "Joint year-bootstrap percentile range: "
                    + str(row["year_bootstrap_95pct"])
                    + "; not calibrated confidence limits.",
                }
            )
    generation_path = root / "docs/weather_generation_results.json"
    if generation_path.exists():
        generation = json.loads(generation_path.read_text())
        shutil.copy2(generation_path, destination / "evidence" / generation_path.name)
        protocol = {
            "id": "weather_generation_20261005",
            "domain": "Weather / energy",
            "title": "Temperature · exploratory learned generation",
            "description": "64 cells; three regions and four quarters. "
            "Train 1981–2010, validation 2011–2015, test 2016–2024.",
            "status": "Local exploratory study complete",
            "finding": generation["finding"],
            "limitations": generation["limits"],
            "metric": "aggregate_w1",
            "direction": "lower",
            "metric_label": "Aggregate-variance W1 ↓",
            "columns": [
                ["aggregate_w1", "Aggregate W1 ↓"],
                ["generation_sd", "Generation SD"],
                ["structure_swd", "Structure SWD ↓"],
            ],
            "protocol_file": "protocols/weather_generation_protocol.md",
            "evidence_file": "evidence/weather_generation_results.json",
            "dataset_hash": generation["source_sha256"]["test"],
            "metric_version": "weather-variance-w1-generated200-v1",
            "split": "test",
        }
        protocols.append(protocol)
        for row in generation["rows"]:
            records.append(
                {
                    "id": "weather-generation-" + row["method"],
                    "protocol": protocol["id"],
                    "method": row["label"],
                    "status": "locally_audited",
                    "evidence_sha256": sha256(generation_path),
                    "samples": 108,
                    "training_seeds": int(row["method"].startswith("diffeocfm")),
                    "generation_seeds": 5,
                    "metrics": {
                        "aggregate_w1": row["aggregate_w1_mean"],
                        "generation_sd": row["generation_sd"],
                        "structure_swd": row["structure_swd_mean"],
                    },
                    "note": "200 generated matrices per context; one model fit. "
                    "Final weights are sensitivity only; "
                    "epoch 190 selected on validation.",
                }
            )
        for key, value in generation["deterministic_controls"].items():
            records.append(
                {
                    "id": "weather-generation-" + key,
                    "protocol": protocol["id"],
                    "method": {
                        "conditional_mean": "Region / quarter mean",
                        "pooled_grid_mean": "Pooled matrix mean",
                        "spatial_kernel": "Spatial kernel",
                    }[key],
                    "status": "locally_audited",
                    "evidence_sha256": sha256(generation_path),
                    "samples": 108,
                    "training_seeds": 0,
                    "generation_seeds": 0,
                    "metrics": {"aggregate_w1": value},
                    "note": "Deterministic train-fitted control; exact agreement "
                    "with the original baseline report was checked.",
                }
            )
    unconditional_path = root / "docs/unconditional_validation_results.json"
    if unconditional_path.exists():
        unconditional = json.loads(unconditional_path.read_text())
        shutil.copy2(
            unconditional_path, destination / "evidence" / unconditional_path.name
        )
        shutil.copy2(
            root / "docs/unconditional_validation_protocol.md",
            destination / "protocols/unconditional_validation_protocol.md",
        )
        protocol = {
            "id": "legacy_unconditional_validation_20261005",
            "domain": "Finance",
            "title": "Legacy archive · unconditional validation diagnostic",
            "description": "80D precomputed matrix archive without dates or asset "
            "identities. One full 50-epoch reference reconstruction; 300 fixed "
            "validation cases, no conditioning.",
            "status": "Local validation diagnostic complete",
            "finding": unconditional["finding"],
            "limitations": unconditional["limits"],
            "metric": "feature_swd",
            "direction": "lower",
            "metric_label": "Validation structural SWD ↓",
            "columns": [
                ["feature_swd", "Validation SWD ↓"],
                ["generation_sd", "Generation SD"],
                ["precision", "Precision ↑"],
                ["recall", "Recall ↑"],
                ["portfolio_swd", "Portfolio SWD ↓"],
            ],
            "protocol_file": "protocols/unconditional_validation_protocol.md",
            "evidence_file": "evidence/unconditional_validation_results.json",
            "dataset_hash": unconditional["source_sha256"]["validation"],
            "metric_version": "legacy-unconditional-validation-300-v1",
            "split": "validation",
        }
        protocols.append(protocol)
        for row in unconditional["summary"]:
            metrics = row["metrics"]
            records.append(
                {
                    "id": "legacy-unconditional-" + row["weights"],
                    "protocol": protocol["id"],
                    "method": {
                        "selected": "CorrGAN reference (selected)",
                        "final": "CorrGAN reference (final)",
                        "replay": "Training replay",
                    }[row["weights"]],
                    "status": "locally_audited",
                    "evidence_sha256": sha256(unconditional_path),
                    "samples": 300,
                    "training_seeds": int(row["weights"] != "replay"),
                    "generation_seeds": 5,
                    "metrics": {
                        **{
                            key: metrics[key]["mean"]
                            for key in [
                                "feature_swd",
                                "precision",
                                "recall",
                                "portfolio_swd",
                            ]
                        },
                        "generation_sd": metrics["feature_swd"]["generation_sd"],
                    },
                    "note": "Validation only; epoch 45 selected, final epoch 50. "
                    "Projection seed 2027; native selection seed 44301. "
                    "Original training recipe remains unverified.",
                }
            )
    financial = {
        "id": "taiwan_categorical_20261005",
        "domain": "Finance",
        "title": "Taiwan · matched categorical generation",
        "description": "80 assets, identical 1,500 test cases, five seeds.",
        "status": "Audited result pending",
        "finding": "Full five-seed runs are being audited.",
        "limitations": (
            "Previously inspected periods; method-specific training budgets. "
            "Concurrent timing does not establish a speed ranking."
        ),
        "metric": "feature_swd",
        "direction": "lower",
        "metric_label": "Structural SWD ↓",
        "columns": [
            ["feature_swd", "Structural SWD ↓"],
            ["feature_sd", "Seed SD"],
            ["recall", "Coverage recall ↑"],
            ["portfolio_swd", "Portfolio SWD ↓"],
            ["classifier_auc", "C2ST AUC · .5 ideal"],
        ],
        "protocol_file": "protocols/matched_taiwan_protocol.md",
        "evidence_file": None,
        "dataset_hash": None,
        "metric_version": "corrbench-21feature-float32-v1",
        "split": "test",
    }
    if args.matched_audit:
        full = json.loads(args.matched_audit.read_text())
        assert full["states"]["corrflow"]["complete"]
        assert full["states"]["diffeocfm_evaluation"]["complete"]
        sanitized = {
            k: full[k]
            for k in [
                "protocol",
                "scope",
                "training_budgets",
                "equal_compute",
                "case_selection",
                "timing_limit",
                "seed_uncertainty_limit",
                "states",
                "source_data_sha256",
                "validation_selection_checks",
                "rows",
                "summary",
            ]
        }
        evidence = destination / "evidence/matched_taiwan.json"
        evidence.write_text(json.dumps(sanitized, indent=2) + "\n")
        financial.update(
            status="Complete included suites audited",
            finding="Read fidelity, coverage and portfolio diagnostics together.",
            evidence_file="evidence/matched_taiwan.json",
            dataset_hash=full["source_data_sha256"]["test"],
        )
        for row in full["summary"]:
            if row["weights"] == "last":
                continue
            metrics = row["metrics"]
            records.append(
                {
                    "id": "taiwan-" + row["method"],
                    "protocol": financial["id"],
                    "method": {
                        "corrflow": "CorrFlow",
                        "diffeocfm": "DiffeoCFM",
                        "gaussian": "Transformed Gaussian",
                        "replay": "Training replay",
                        "ccorrgan": "cCorrGAN reimplementation",
                    }[row["method"]],
                    "status": "locally_audited",
                    "evidence_sha256": sha256(evidence),
                    "samples": 1500,
                    "training_seeds": 0 if row["weights"] == "baseline" else 5,
                    "generation_seeds": 5,
                    "metrics": {
                        **{
                            k: metrics[k]["mean"]
                            for k in [
                                "feature_swd",
                                "recall",
                                "portfolio_swd",
                                "classifier_auc",
                            ]
                        },
                        "feature_sd": metrics["feature_swd"]["sd"],
                    },
                    "note": row["uncertainty"]
                    + ". Selected checkpoints; final weights retained in evidence.",
                }
            )
        if not full["states"]["ccorrgan"]["complete"]:
            financial["limitations"] += (
                " cCorrGAN's full suite is still running and excluded."
            )
    protocols.insert(0, financial)
    named_path = root / "docs/abide_named110_results_20261006.json"
    named = json.loads(named_path.read_text())
    if not named["input_selection_audit"]["all_native_validation_scores_exact"]:
        raise ValueError("Named-region controls lack a complete selection audit")
    shutil.copy2(named_path, destination / "evidence" / named_path.name)
    named_protocol = "abide_named110_protocol_20261006.md"
    shutil.copy2(
        root / "docs" / named_protocol, destination / "protocols" / named_protocol
    )
    for cohort, outcome in named["cohorts"].items():
        protocol_id = f"abide_named110_{cohort}_20261006"
        splits = outcome["splits"]
        count = splits["test"]["subjects"]
        protocols.append(
            {
                "id": protocol_id,
                "domain": "Brain connectivity",
                "title": f"ABIDE · 110 named regions · {cohort}",
                "description": (
                    "Re-estimated OAS targets. "
                    f"{splits['train']['subjects']} training, "
                    f"{splits['validation']['subjects']} Michigan validation and "
                    f"{count} NYU test subjects. Simple controls; no learned fits."
                ),
                "status": "Post-pilot sensitivity complete",
                "finding": "Replay and Gaussian augmentation establish no gain.",
                "limitations": named["uncertainty"] + ". Previously inspected sites. "
                "Compare cohorts separately; common-subject results are in evidence.",
                "metric": "auc",
                "direction": "higher",
                "metric_label": "Test ROC-AUC ↑",
                "columns": [
                    ["auc", "Test AUC ↑"],
                    ["generation_sd", "Generation SD"],
                    ["balanced_accuracy", "Balanced accuracy ↑"],
                    ["auc_difference", "Δ real-only AUC"],
                ],
                "protocol_file": f"protocols/{named_protocol}",
                "evidence_file": f"evidence/{named_path.name}",
                "dataset_hash": splits["test"]["sha256"],
                "metric_version": "abide-logistic-utility-named110-v1",
                "split": "test",
            }
        )
        for row in outcome["full_test"]:
            records.append(
                {
                    "id": f"named110-{cohort}-{row['method']}",
                    "protocol": protocol_id,
                    "method": {
                        "real_only": "Real only",
                        "replay": "Training replay",
                        "gaussian": "Transformed Gaussian",
                    }[row["method"]],
                    "status": "locally_audited",
                    "evidence_sha256": sha256(named_path),
                    "samples": count,
                    "training_seeds": 0,
                    "generation_seeds": row["generation_seeds"],
                    "metrics": {
                        "auc": row["auc_mean"],
                        "generation_sd": row["auc_generation_sd"],
                        "balanced_accuracy": row["balanced_accuracy_mean"],
                        "auc_difference": row["auc_difference_from_real_only"],
                    },
                    "note": "Paired subject 95% interval, difference from real-only: "
                    f"{row['auc_difference_paired_subject_95pct']}. "
                    "Conditional on fitted models and this site; unadjusted.",
                }
            )
    for protocol in protocols:
        protocol["protocol_sha256"] = sha256(destination / protocol["protocol_file"])
        for record in records:
            if record["protocol"] == protocol["id"]:
                record.update(
                    dataset_hash=protocol["dataset_hash"],
                    split=protocol["split"],
                    metric_version=protocol["metric_version"],
                    protocol_sha256=protocol["protocol_sha256"],
                )
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate result IDs")
    payload = {
        "schema_version": "corrbench.leaderboard.v1",
        "public_release": False,
        "updated": "2026-10-06",
        "protocols": protocols,
        "records": records,
        "review_scope": "Local provenance audits; no external review claimed.",
    }
    (destination / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    validate_leaderboard(destination)
    print(f"Built {len(records)} records across {len(protocols)} separate protocols")


if __name__ == "__main__":
    main()
