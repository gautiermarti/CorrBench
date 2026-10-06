# Exploratory weather generation result — 2026-10-05

DiffeoCFM improves on its Gaussian prior but does not beat replay or the conditional mean on aggregate-variance W1.

The frozen 200-epoch run selected epoch 190 using validation structural SWD. All generated matrices pass validity checks.

| Method | Aggregate-variance W1 ↓ | Generation SD | Structural SWD ↓ |
|---|---:|---:|---:|
| Transformed Gaussian | 0.107577 | 0.000167 | 2.003368 |
| DiffeoCFM (selected) | 0.089409 | 0.000898 | 1.787219 |
| DiffeoCFM (final) | 0.087124 | 0.000712 | 1.780329 |
| Region / quarter replay | 0.048258 | 0.000747 | 0.387261 |

Conditional mean W1: 0.082379.

One fitted model seed, five generation seeds, 200 matrices per region/quarter context. The final checkpoint is a declared sensitivity; it does not replace validation selection. Replay was recomputed at the same draw count (the initial baseline study used 1,000).

This extension was declared after the baseline test results were known. It uses the same nine test years, with 12 dependent region/quarter contexts. Generation SD is Monte Carlo variability conditional on the fit, not uncertainty over training or future years. Structural SWD subsamples only nine matrices per condition; the utility calculation uses all 200 generated draws. No forecasting, extreme-event, energy-production or spatial-transfer benefit follows.

The summary validates every frozen source and output hash, complete-job status, generated artifact hash, and numerical aggregation. Full per-context values and sampling diagnostics remain in the hashed original report.
