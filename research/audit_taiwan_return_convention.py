"""Verify the frozen Taiwan return column's arithmetic without changing data.

This checks curated-column provenance, not independent corporate-action events,
cash-dividend total returns, delisting proceeds or upstream acquisition rights.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--builder", type=Path, required=True)
    p.add_argument("--dataset-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.dataset_manifest.read_text())
    panel_sha = sha(args.panel)
    assert panel_sha == manifest["source_sha256"]
    fields = [
        "instrument_id",
        "date",
        "close",
        "spread",
        "previous_close",
        "reference_price",
        "reference_return",
        "raw_close_return",
        "uses_exchange_reference",
        "daily_total_return",
        "unresolved_return_flag",
    ]
    data = pd.read_parquet(args.panel, columns=fields)
    ids = data["instrument_id"].to_numpy()
    dates = data["date"].to_numpy()
    close = data["close"].to_numpy()
    same = ids[1:] == ids[:-1]
    assert np.all(dates[1:][same] > dates[:-1][same])
    previous = np.r_[np.nan, close[:-1]]
    previous[np.r_[True, ~same]] = np.nan
    np.testing.assert_allclose(
        previous, data["previous_close"], rtol=0, atol=0, equal_nan=True
    )
    reference = close - data["spread"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        reference_return = data["spread"].to_numpy() / reference
        raw = close / previous - 1
    use_reference = (
        (reference > 0)
        & np.isfinite(reference_return)
        & (np.abs(reference_return) <= 0.35)
    )
    candidate = np.where(use_reference, reference_return, raw)
    expected = np.where(np.abs(candidate) <= 0.35, candidate, np.nan)
    for actual, rebuilt in [
        (data["reference_price"], reference),
        (data["reference_return"], reference_return),
        (data["raw_close_return"], raw),
        (data["daily_total_return"], expected),
    ]:
        np.testing.assert_allclose(actual, rebuilt, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(data["uses_exchange_reference"], use_reference)
    np.testing.assert_array_equal(
        data["unresolved_return_flag"], np.abs(candidate) > 0.35
    )
    finite = np.isfinite(expected)
    difference = np.abs(expected - raw)
    observed = np.isfinite(difference)
    # An algebraic illustration only, not an observed dividend event.
    old_price, cash_distribution, new_price = 100.0, 10.0, 95.0
    ref_example = new_price / (old_price - cash_distribution) - 1
    cash_example = (new_price + cash_distribution) / old_price - 1
    result = {
        "scope": "Complete arithmetic/provenance audit of the frozen curated column",
        "source_column": "daily_total_return",
        "precise_description": "Returns relative to exchange reference prices, "
        "with a raw-close fallback and a 35% unresolved-return screen",
        "panel_sha256": panel_sha,
        "builder_sha256": sha(args.builder),
        "dataset_manifest_sha256": sha(args.dataset_manifest),
        "audit_code_sha256": sha(Path(__file__)),
        "rows": len(data),
        "finite_returns": int(finite.sum()),
        "reference_branch_rows": int(use_reference.sum()),
        "finite_raw_close_fallback_rows": int((finite & ~use_reference).sum()),
        "null_returns": int((~finite).sum()),
        "unresolved_over_35pct": int((np.abs(candidate) > 0.35).sum()),
        "reference_vs_raw_difference_over_1bp": int(
            (observed & (difference > 1e-4)).sum()
        ),
        "maximum_absolute_arithmetic_error": 0.0,
        "data_modified": False,
        "formulas": {
            "reference_price": "close - exchange_reported_spread",
            "reference_return": "spread / reference_price",
            "raw_close_return": "close / previous_observed_close - 1",
            "choice": "Use reference if reference_price>0 and "
            "abs(reference_return)<=.35; otherwise raw. "
            "If abs(chosen_return)>.35 or missing, save null.",
        },
        "algebraic_example_not_an_observed_event": {
            "previous_close": old_price,
            "cash_distribution": cash_distribution,
            "close": new_price,
            "reference_return": ref_example,
            "cash_distribution_total_return": cash_example,
        },
        "limits": "The internal column name is preserved for byte-identical "
        "reproduction. It is not proof of an independently reconciled "
        "cash-dividend total-return series. A reference-price adjustment is "
        "not generally equal to (close+cash_distribution)/previous_close-1. "
        "No event-level adjustment, delisting-proceeds, upstream-download "
        "or legal-permission audit is established by this calculation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
