# Node identities are part of the task

Post-pilot controlled diagnostic, 2026-10-05. Both test datasets were already
inspected. This is a deliberately corrupted-input check, not a trained-generator
comparison or a newly untouched holdout. Ten fixed permutation seeds 2027–2036
are all retained; no fitting or selection occurs.

Simultaneous row/column permutation preserves eigenvalues and the multiset of
off-diagonal correlations. Consequently, every one of the 21 shared structural
summaries is mathematically invariant. A score built only from those summaries
cannot distinguish a correct matrix from the same entries assigned to the wrong
named regions or locations.

| Check | Original assignment | Scrambled matrix, unchanged identities |
| --- | --- | --- |
| Brain: fixed real-only classifier test AUC | 0.7654 | Mean 0.5511; range 0.4337–0.6683 |
| Weather: mean relative quadrant-variance error against the same OAS matrices | 0 | 29.50% |
| Maximum standardized structural-feature difference, brain | 0 | 4.11e-13 |
| Maximum standardized structural-feature difference, weather | 0 | 3.40e-13 |

The brain classifier and scaler are the original validation-selected real-only
artifacts, verified by hash before use. Predictions after restoring the original
ROI order match exactly. In weather, co-permuting the aggregation weights and
matrices restores variance to within 5.11e-15. This is the expected consistency
of a true relabeling; assigning a matrix to unchanged, incorrect labels is a
different operation.

These checks support named-node diagnostics and utility tests in CorrBench.
They do not invalidate permutation-invariant metrics for explicitly anonymous
financial tasks. They also do not establish a new method ranking or uncertainty
over future sites. The weather calculation isolates dependence under unit
marginal scales; the reported error is against the original matrix functionals,
not a new forecasting target.

Reproduce with research/check_node_identity.py. The
[machine-readable report](node_identity_diagnostic.json) retains every
permutation, source/model hashes and the exact code hash. Individual prediction
arrays remain local. No test-driven model change followed this diagnostic.
