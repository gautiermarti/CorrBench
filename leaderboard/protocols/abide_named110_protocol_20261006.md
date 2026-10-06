# ABIDE 110 named-region sensitivity protocol — 2026-10-06

Declared after the [source-atlas diagnostic](abide_atlas_audit_20261006.md),
before reconstructing 110-region matrices, fitting these controls, or examining
their scores. This is a post-pilot sensitivity analysis on inspected sites,
not a new holdout or an amendment to the frozen 111-label study. The original
601-subject arrays, checkpoints, results and leaderboard entries remain intact.

## Source, target and cohorts

Use the same 780 hashed PCP `cpac/filt_noglobal/rois_ho` series, metadata quality
rules and subject/site partitions as the original protocol. The pinned official
label CSV (SHA-256
`b719f40d03414a61e8cfc12f9fd377a493521e3624bdd00ee088e7f613f31214`)
names exactly 110 of the 111 source IDs. Retain those IDs in original header
order, excluding only unnamed 3455. Do not infer that this proves an anatomical
error or globally optimal parcellation. Reject any mismatch with the audited
header, any changed source hash, nonfinite observations, fewer than 50 time
points, or any remaining constant region (sample SD <= 1e-12).

Remove 3455 from the raw regional series before per-scan standardization and
OAS estimation. Use the existing `scan_correlations` estimator unchanged.
Dropping a row/column from cached 111-dimensional targets is not this estimator:
OAS shrinkage must be re-estimated. Preserve empirical Pearson, ranks, shrinkage,
minimum eigenvalues, source hashes, ordered node IDs and original subject metadata.

Build two explicitly versioned local datasets:

- `same601`: exactly the original subject IDs and their order in each partition
  (349 train, 92 Michigan validation, 160 NYU test), with reconstructed targets.
- `expanded762`: all accepted named-region scans (expected 488/108/166 from the
  source audit). All remaining exclusions must match the 18 scans with another
  constant region. Unexpected differences stop the study for diagnosis.

Require subject/site disjointness, positive definite targets and identical
arrays for common subjects across these two reconstructions. Report differences
between reconstructed targets and row/column deletion, including shrinkage.
The common cohort isolates the target/feature change; the expanded cohort also
changes training, validation and test populations. Neither isolates all possible
preprocessing or site confounds. Data remain local under the original source
terms; redistribution and complete atlas/site attribution are separate tasks.

## Fixed controls and selection

Reuse the original utility implementation unchanged: ordered upper-triangle
entrywise Fisher features clipped at +/- (1-1e-7), StandardScaler fitted on real
training subjects only, balanced L2 logistic regression with liblinear,
max_iter 2000 and threshold .5. Select C from {.001, .01, .1, 1, 10} using maximum
validation-site ROC-AUC, taking the smallest C on ties. Fit each cohort separately.
Treat convergence warnings as failures. Save all validation scores, selected
models, code/data hashes and a study-specific registration before test evaluation.
The reused helper retains its historical protocol field; the enclosing study
registration explicitly identifies this new protocol and cohort.

For each cohort, evaluate real-only; real plus class-balanced training replay;
and real plus the full class-wise OAS Gaussian in train-standardized normalized
Cholesky coordinates, using the existing CorrFlow `CorrelationGaussian` code.
Fit each Gaussian from that cohort's training matrices only. Use generation
seeds 2027--2031 and add ceil(N_train/2) matrices per class, exactly as before.
Keep full covariance and float64; CPU, four threads, no GPU or learned fits.
Check every generated matrix for finiteness, symmetry, unit diagonal and PD.
Preserve each generated draw set and hash. Do not replace unsuccessful controls.

Report AUC and balanced accuracy per arm and mean/sample SD across five replay
or Gaussian seeds. Use the existing 1,000 diagnosis-stratified paired subject
bootstrap, seed 7201, for mean per-seed AUC difference from real-only (not AUC
of averaged predictions). Intervals condition on this cohort, fitted models
and one site; they are unadjusted for multiple comparisons and do not represent
training- or site-level uncertainty. No clinical interpretation is justified.

## Paired comparisons and stopping point

On the exact common 160 NYU subjects, report each cohort's controls as well as:

1. `same601` real-only versus the frozen 111-label real-only predictions, with
   a paired subject-bootstrap interval. This changes the matrix target/features.
2. `expanded762` real-only versus `same601` real-only on those same 160 subjects,
   with the same interval. This changes training and validation membership while
   keeping evaluation subjects fixed. Also report full expanded-test scores.

Record class/site composition and all excluded subjects locally; use aggregated
counts in release-facing reports. Do not optimize using either comparison.
The added six NYU subjects do not make a fresh site-level holdout. Stop after
these complete controls and the source/selection audits. Learned 110-region
fits, alternative atlas/QC choices and genuinely fresh sites need separate
protocols; they are not silently added based on these scores.

The original financial archival paper is unaffected. Any extended-paper update
must distinguish this sensitivity analysis from the original negative learned
augmentation result and retain both.
