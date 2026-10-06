# Financial window-overlap audit — 2026-10-05

The paper's split endpoints are chronological, but their full return windows
are not disjoint at adjacent boundaries. The audit maps every endpoint to the
actual source-session index and includes exactly 252 observations. It verifies
the source, manifest and original split hashes before computing any view.
The [machine-readable report](market_window_overlap_audit.json) preserves
hashes, date extents, counts and row-index artifact fingerprints.

| Dataset | Train/validation shared sessions | Validation/test shared sessions | Train/test shared sessions |
|---|---:|---:|---:|
| Taiwan paper | 251 | 251 | 0 |
| Japan paper | 251 | 251 | 9 |

These are dates present in the union of observation windows, not a count of
shared date/asset pairs. Date disjointness is a conservative stronger condition
than excluding only identical returns for the same stock.

## Two diagnostic views

**Boundary purging:** keep the original training rows. In each later split,
retain only windows whose first return session is strictly after the last
original window endpoint of the preceding split. Within-split windows remain
dense and overlapping. Boundaries, lookback, source matrices and labels are
unchanged; no model is fitted and the old results do not apply to this view.

| Dataset | Train matrices | Validation matrices | Test matrices |
|---|---:|---:|---:|
| Taiwan, original | 13,600 | 3,920 | 6,888 |
| Taiwan, boundary purged | 13,600 | 1,912 | 4,880 |
| Japan, original | 3,848 | 1,936 | 2,888 |
| Japan, boundary purged | 3,848 | 0 | 880 |

Japan's 2020 validation split has only 242 endpoint sessions. With a 252-session
lookback, none of those windows starts after the 2019 training endpoint.
Its validation period must be lengthened or its complete protocol redesigned;
silently keeping overlapping rows would not satisfy the stated requirement.

**Disjoint parent blocks:** within the same boundary-purged candidates, retain
the earliest eligible endpoint, then the next whose first session is after the
previous retained endpoint. Keep every original asset-subset row at each chosen
endpoint. This deterministic greedy rule has no dependence on model outcomes.

| Dataset | Train parent windows (matrices) | Validation parent windows (matrices) | Test parent windows (matrices) |
|---|---:|---:|---:|
| Taiwan | 7 (56) | 1 (8) | 3 (24) |
| Japan | 2 (16) | 0 (0) | 1 (8) |

This illustrates the cost of demanding non-overlapping 252-session windows
everywhere. Eight asset subsets of one block remain eight views of the same
episode, not eight independent years. Even non-overlapping blocks can be
dependent through economic regimes. No effective sample size is inferred.

## Reproduction and intended use

```bash
python research/audit_market_windows.py --data /path/to/paper-splits \
  --source /path/to/curated-returns.parquet \
  --output artifacts/window-audit
```

The command writes original, boundary-purged and disjoint-block row indices plus
the audit. It never overwrites a dataset. Tests cover calendar gaps, inclusive
window endpoints, multiple views of one parent, an empty validation split and
invalid provenance. Both real source panels have been audited.

These are construction diagnostics on previously inspected periods. They do
not create an untouched evaluation set, reproduce the model table on a new
split, or settle data redistribution. A future financial CorrBench track should
declare its observation-disjoint boundaries and sufficient validation history
before fitting models. A competition additionally needs a protected reference
that cannot be reconstructed from released returns.
