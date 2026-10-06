# Reference export v1

`corrbench-export-hf` stages immutable dated-market datasets locally as Parquet,
a Hub-compatible dataset card, a sanitized manifest and integrity hashes. It has
no upload or authentication code. Existing NPZ files are inputs, never modified.

```bash
corrbench-export-hf --data data/taiwan --metadata release-metadata.yaml \
  --output artifacts/reference/taiwan-v1
```

Required YAML fields:

```yaml
config_name: finance_taiwan_n80
dataset_version: paper-v1-staging
domain: finance
protocol_id: taiwan-paper-sample-holdout-v1
source_reference: Describe the audited source and public attribution here
observation_definition: State the exact return/adjustment convention and lookback
builder_reference: Exact builder revision or source hashes
split_description: State the dates, prior inspection and overlapping-window limitations
matrix_space: real_correlation_pd
rights_status: pending
```

Use `rights_status: cleared` only with `license` and `rights_evidence` describing
the actual source-specific review. This metadata records evidence; a string value
does not perform a legal review or authorize publication. Pending rights produce
a marked staging bundle with no asserted data license. The final source attribution
and paper/code citation links must be reviewed before public release.
Use `rights_status: restricted` with nonempty `rights_evidence` when the recorded
source terms expressly limit the intended use. Both pending and restricted
statuses set `distribution_blocked: true`, omit an open-license assertion and
display the evidence on the card. The Japan competition source currently has
this restricted status; see the [rules review](jpx_rules_review_20261006.md).

The frozen construction manifest supplies matrix dimension, window length, regime
definition, source hash and per-split hashes. The exporter checks the latter
against every NPZ it reads. Public metadata excludes machine-specific source
paths and unrecognized top-level construction fields.

| Record field | Meaning |
| --- | --- |
| `sample_id` | Configuration, split and original row identity |
| `source_row_index` | Index in the frozen source NPZ |
| `matrix` | All n*n entries, flattened in C/row-major order; original float precision |
| `variable_ids` | Ordered identifiers corresponding to rows/columns |
| `condition` | Original integer class label |
| `window_end` | Original observation-window endpoint |
| `n_observations` | Source-session lookback, not an effective-independent-sample count |
| `original_labels`, `sharpes`, `minimum_eigenvalues` | Optional dated-market diagnostics, unchanged |

The manifest records original NumPy dtypes and per-sample shapes. The reader
`corrbench.reference.read_reference_split` restores the original arrays and
verifies manifest and Parquet integrity. Full matrices are retained instead of
discarding one triangle, preserving even floating-point asymmetries within the
declared 1e-6 tolerance. Eigenvalues are checked on the symmetrized float64 matrix.
PD requires strictly positive eigenvalues; PSD allows a minimum of -1e-6. No
matrix repair occurs, and numeric checks do not establish statistical realism.

This dated-market adapter accepts only its declared fields. Nested parent IDs,
legacy undated data and variable-size matrices require explicit new adapters,
not silent dropping of provenance. The implemented brain and weather adapters
use their own domain contracts; see [domain reference exports](domain_reference_format.md).

The Taiwan column's historical name daily_total_return is retained for exact
reproduction, but its verified meaning is returns relative to exchange reference
prices with a raw-close fallback. See the [arithmetic audit](taiwan_return_convention.md)
and [source-specific release review](financial_release_review.md).

`--purpose competition_training` exports train and validation only and never
reads test NPZ files or copies their split reports. It does not create a new
holdout or make already-public sources secret. Never publish a competition's
protected reference data through a paper-reproduction export.

Output directories must not exist. A failed export removes its temporary staging
files instead of leaving a partial release. Hashes support integrity/reproduction;
they are not digital signatures or evidence of source-data redistribution rights.
