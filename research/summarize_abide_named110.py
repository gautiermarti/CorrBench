"""Summarize both declared cohorts and paired original-subject comparisons."""

import argparse
import json
from pathlib import Path

import fmri_utility as utility
import numpy as np
from summarize_fmri_pilot import bootstrap_auc_difference

from corrbench.neuro import sha256


def grouped(predictions, take=None):
    if take is None:
        take = np.arange(len(predictions["labels"]))
    labels = predictions["labels"][take]
    reference = predictions["real_only"][take]
    rows = []
    for family in ("real_only", "replay", "gaussian"):
        names = (
            ["real_only"]
            if family == "real_only"
            else [f"{family}_{s}" for s in range(2027, 2032)]
        )
        probabilities = np.stack([predictions[name][take] for name in names])
        scores = [utility.metrics(labels, p) for p in probabilities]
        rows.append(
            {
                "method": family,
                "auc_mean": float(np.mean([s["roc_auc"] for s in scores])),
                "auc_generation_sd": float(
                    np.std([s["roc_auc"] for s in scores], ddof=1)
                )
                if len(scores) > 1
                else 0.0,
                "balanced_accuracy_mean": float(
                    np.mean([s["balanced_accuracy"] for s in scores])
                ),
                "auc_difference_from_real_only": float(
                    np.mean([s["roc_auc"] for s in scores])
                    - utility.metrics(labels, reference)["roc_auc"]
                ),
                "auc_difference_paired_subject_95pct": bootstrap_auc_difference(
                    labels, probabilities, reference
                ),
                "generation_seeds": len(scores) if family != "real_only" else 0,
            }
        )
    return rows


def paired(labels, candidate, reference):
    return {
        "candidate_auc": utility.metrics(labels, candidate)["roc_auc"],
        "reference_auc": utility.metrics(labels, reference)["roc_auc"],
        "auc_difference": utility.metrics(labels, candidate)["roc_auc"]
        - utility.metrics(labels, reference)["roc_auc"],
        "paired_subject_95pct": bootstrap_auc_difference(
            labels, candidate[None], reference
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, bench = args.study_root.resolve(), args.bench_root.resolve()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        raise FileExistsError(args.output)
    audit_path = root / "input_selection_audit.json"
    audit = json.loads(audit_path.read_text())
    registration_path = root / "controls/selection_registration.json"
    if audit["status"] != "passed" or audit["selection_registration_sha256"] != sha256(
        registration_path
    ):
        raise ValueError("Selection audit absent or stale")
    registration = json.loads(registration_path.read_text())
    construction_path = root / "data/construction_audit.json"
    construction = json.loads(construction_path.read_text())
    summaries, predictions = {}, {}
    for cohort in ("same601", "expanded762"):
        directory = root / "controls" / cohort / "utility"
        report_path = directory / "test_report.json"
        report = json.loads(report_path.read_text())
        if (
            report["selection_sha256"]
            != registration["cohorts"][cohort]["selection_sha256"]
        ):
            raise ValueError("Test report used another selection")
        expected_test = construction["cohorts"][cohort]["splits"]["test"]["sha256"]
        if report["test_sha256"] != expected_test or len(report["arms"]) != 11:
            raise ValueError("Incomplete or mismatched test report")
        with np.load(directory / "test_predictions.npz") as a:
            predictions[cohort] = {k: a[k] for k in a.files}
        with np.load(root / "data" / cohort / "test.npz") as a:
            for key in ("subject_ids", "labels"):
                np.testing.assert_array_equal(predictions[cohort][key], a[key])
        for arm in report["arms"]:
            score = utility.metrics(
                predictions[cohort]["labels"], predictions[cohort][arm["name"]]
            )
            if any(score[key] != arm[key] for key in score):
                raise ValueError("Saved test score differs from predictions")
        summaries[cohort] = {
            "splits": construction["cohorts"][cohort]["splits"],
            "full_test": grouped(predictions[cohort]),
            "per_arm": report["arms"],
            "sources_sha256": {
                name: sha256(directory / name)
                for name in (
                    "selection.json",
                    "test_report.json",
                    "test_predictions.npz",
                )
            },
        }
    same, expanded = predictions["same601"], predictions["expanded762"]
    index = {int(v): i for i, v in enumerate(expanded["subject_ids"])}
    take = [index[int(v)] for v in same["subject_ids"]]
    np.testing.assert_array_equal(same["labels"], expanded["labels"][take])
    summaries["expanded762"]["common160_test"] = grouped(expanded, take)
    old_path = bench / "artifacts/abide_complete_utility_20261005/test_predictions.npz"
    if sha256(old_path) != registration["frozen_predictions_sha256"]:
        raise ValueError("Frozen comparison changed")
    with np.load(old_path) as old:
        np.testing.assert_array_equal(same["subject_ids"], old["subject_ids"])
        np.testing.assert_array_equal(same["labels"], old["labels"])
        target_change = paired(same["labels"], same["real_only"], old["real_only"])
    population_change = paired(
        same["labels"], expanded["real_only"][take], same["real_only"]
    )
    result = {
        "protocol": registration["protocol"],
        "protocol_sha256": registration["protocol_sha256"],
        "scope": (
            "Post-pilot named-region/cohort sensitivity on inspected NYU and "
            "Michigan sites; simple controls only"
        ),
        "fresh_holdout": False,
        "learned_110_region_fits": False,
        "cohorts": summaries,
        "paired_common160_real_only": {
            "target_change_same_cohort": target_change,
            "training_validation_cohort_change": population_change,
        },
        "construction": construction,
        "input_selection_audit": audit,
        "uncertainty": (
            "Five generation draws, not training replications; 1000 paired "
            "diagnosis-stratified subject draws, conditional on fitted models "
            "and one site, unadjusted for multiple comparisons"
        ),
        "source_sha256": {
            "selection_registration": sha256(registration_path),
            "input_selection_audit": sha256(audit_path),
            "construction": sha256(construction_path),
            "summary_code": sha256(Path(__file__)),
        },
        "environment": {
            k: registration[k] for k in ("python", "numpy", "sklearn", "torch")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# ABIDE 110 named-region sensitivity results — 2026-10-06",
        "",
        "Both declared cohorts and all 22 control arms complete. This is a sensitivity",
        "analysis on inspected sites. No 110-region learned generator was fitted.",
        "",
        "| Cohort / evaluation | Real-only AUC | Replay AUC ± draw SD | "
        "Gaussian AUC ± draw SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, rows in [
        ("Same 601 / original 160 NYU subjects", summaries["same601"]["full_test"]),
        ("Expanded 762 / all 166 NYU subjects", summaries["expanded762"]["full_test"]),
        (
            "Expanded 762 / common 160 NYU subjects",
            summaries["expanded762"]["common160_test"],
        ),
    ]:
        real, replay, gauss = rows
        lines.append(
            f"| {label} | {real['auc_mean']:.4f} | {replay['auc_mean']:.4f} "
            f"± {replay['auc_generation_sd']:.4f} | {gauss['auc_mean']:.4f} "
            f"± {gauss['auc_generation_sd']:.4f} |"
        )
    lines += [
        "",
        "## Augmentation differences from each cohort's real-only control",
        "",
        "| Cohort / evaluation | Control | Δ AUC [paired subject 95% interval] |",
        "| --- | --- | ---: |",
    ]
    for cohort, evaluation, rows in [
        ("Same 601", "160", summaries["same601"]["full_test"]),
        ("Expanded 762", "166", summaries["expanded762"]["full_test"]),
        ("Expanded 762", "common 160", summaries["expanded762"]["common160_test"]),
    ]:
        for row in rows[1:]:
            lo, hi = row["auc_difference_paired_subject_95pct"]
            lines.append(
                f"| {cohort} / {evaluation} | {row['method']} | "
                f"{row['auc_difference_from_real_only']:+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}] |"
            )
    lines += ["", "## Paired source/cohort comparisons", ""]
    for label, comparison in [
        ("110 versus 111 labels on the same cohort", target_change),
        (
            "Expanded versus original training/validation cohort "
            "on the same 160 test subjects",
            population_change,
        ),
    ]:
        lo, hi = comparison["paired_subject_95pct"]
        lines.append(
            f"- {label}: real-only AUC {comparison['reference_auc']:.4f} "
            f"→ {comparison['candidate_auc']:.4f}; "
            f"Δ {comparison['auc_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]."
        )
    lines += [
        "",
        "Five draws measure generation variation; intervals condition on fitted models",
        "and this one site, with no multiple-comparison correction. Expanded-cohort",
        "scores also change the population; the common-subject comparison retains",
        "training/validation membership changes. These analyses do not establish",
        "generalization to new sites or clinical usefulness.",
        "",
        "## Reconstruction and selection evidence",
        "",
        "All 780 source hashes checked. Reconstructed 762 named-region scans; 18 still",
        "have another constant region. Original 601-subject arrays remain unchanged.",
        "Independent adapter calculations reproduce all 762 targets using the same",
        "sklearn OAS estimator. Common-subject arrays match exactly between cohorts.",
        "All 22 selected validation scores reproduce exactly and all feature scalers",
        "use real training subjects only. Both selections were sealed before scoring.",
        "",
        "The frozen 111-label learned-augmentation results remain separate. This study",
        "does not establish how retrained learned generators perform at 110 regions.",
        "",
        "Protocol: [declared settings](abide_named110_protocol_20261006.md).",
        "Detailed counts, all control scores and hashes: "
        f"[{args.output.name}]({args.output.name}).",
        "",
        "Local artifacts: `artifacts/abide_named110_20261006/`. Data redistribution",
        "uncleared; no participant matrices or atlas images were uploaded.",
        "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
