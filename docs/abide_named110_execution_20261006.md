# Named-region control execution record

The [declared protocol](abide_named110_protocol_20261006.md) was committed and
pushed as `394f855` before reconstruction or control fitting. The complete
[results](abide_named110_results_20261006.md) and executable study source are
preserved at `ff9bd2183fa7cfe67f674318f528d6e54f4ce7d5`.

Both cohorts and all 22 control arms completed. Reconstruction checks independently
calculate all 762 targets from original series using the same sklearn OAS library;
maximum entry error is 2.220446049250313e-16 and shrinkage agrees exactly.
All 601 common subjects' arrays agree exactly between cohorts. All 22 native
validation selections reproduce exactly, with real-training-only scalers.
Seven targeted construction/utility tests pass in the isolated CPU environment.
No GPU was used and no learned 110-region generator was fitted.

The source-aligned execution sequence, from a checkout of that revision with
the previously acquired local source artifacts, was:

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=src:../corrflow/src
artifacts/clean-env/bin/python research/build_abide_named110.py \
  --output artifacts/abide_named110_20261006/data
artifacts/clean-env/bin/python research/run_abide_named110_controls.py select \
  --study-root artifacts/abide_named110_20261006
artifacts/clean-env/bin/python research/audit_abide_named110.py \
  --study-root artifacts/abide_named110_20261006
artifacts/clean-env/bin/python research/run_abide_named110_controls.py evaluate \
  --study-root artifacts/abide_named110_20261006
artifacts/clean-env/bin/python research/summarize_abide_named110.py \
  --study-root artifacts/abide_named110_20261006 \
  --output docs/abide_named110_results_20261006.json
```

Outputs must be absent; use a separate destination when reproducing. Exact
environment versions, source/data/code hashes, both sealed selections and every
score are recorded in the result JSON and local artifacts. The maintained runner
subsequently received import-order formatting only; its pre-format execution
bytes remain in the pinned commit. Use that revision when auditing the existing
sealed run, whose guards correctly reject any changed runner bytes. Fresh runs
using the maintained runner record their own hashes.

The GPU DDPM queue is independent. Original financial, 111-label brain and weather
results remain intact. Original brain arrays and all 780 source file hashes were
checked. The original 37 leaderboard records are preserved; the follow-up adds
six controls under two separate cohort-specific views, for 43 records in nine
views. The secondary common-160 comparison remains in the evidence record.

No source time series, participant matrices, fitted brain classifier or atlas image
was uploaded. Data-rights review remains separate. The intervals are conditional
on inspected sites and fitted controls; the real-only improvements' paired
intervals include zero, and no augmentation benefit is established.
