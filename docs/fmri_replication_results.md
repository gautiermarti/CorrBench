# ABIDE five-training-seed replication — 2026-10-05

Five fixed training replications; previously inspected NYU site.

Real-only AUC: 0.7654.

| Method | AUC mean ± training SD | Δ AUC | Joint bootstrap 95% range |
| --- | --- | --- | --- |
| DiffeoCFM selected | 0.7452 ± 0.0056 | -0.0202 | [-0.0404, -0.0016] |
| DiffeoCFM final | 0.7396 ± 0.0036 | -0.0258 | [-0.0463, -0.0065] |
| Node-aware CorrFlow | 0.7365 ± 0.0137 | -0.0289 | [-0.0614, +0.0032] |
| Anonymous CorrFlow | 0.7314 ± 0.0184 | -0.0340 | [-0.0747, +0.0044] |

Average AUC over five draws within each training seed, then mean/sample SD across five training seeds. Paired subject and joint training/subject bootstraps are conditional on this site, exploratory and unadjusted for multiplicity. Five generation draws are not independent training fits.

The initial one-seed report remains preserved. No test-driven configuration change was made for this expansion.
All 100 learned augmentation outcomes and all per-training-seed results are retained in the machine-readable report.
Source audits verify common data, validation selection, complete budgets, prior parity and ROI ordering.

## DiffeoCFM selected: per-training-seed results

| Training seed | Mean AUC | Generation SD |
| --- | --- | --- |
| 1701 | 0.7441 | 0.0141 |
| 1702 | 0.7519 | 0.0122 |
| 1703 | 0.7490 | 0.0121 |
| 1704 | 0.7435 | 0.0116 |
| 1705 | 0.7374 | 0.0133 |

## DiffeoCFM final: per-training-seed results

| Training seed | Mean AUC | Generation SD |
| --- | --- | --- |
| 1701 | 0.7397 | 0.0073 |
| 1702 | 0.7431 | 0.0206 |
| 1703 | 0.7423 | 0.0172 |
| 1704 | 0.7339 | 0.0083 |
| 1705 | 0.7390 | 0.0123 |

## Node-aware CorrFlow: per-training-seed results

| Training seed | Mean AUC | Generation SD |
| --- | --- | --- |
| 1701 | 0.7573 | 0.0178 |
| 1702 | 0.7365 | 0.0221 |
| 1703 | 0.7319 | 0.0194 |
| 1704 | 0.7194 | 0.0213 |
| 1705 | 0.7373 | 0.0214 |

## Anonymous CorrFlow: per-training-seed results

| Training seed | Mean AUC | Generation SD |
| --- | --- | --- |
| 1701 | 0.7628 | 0.0167 |
| 1702 | 0.7214 | 0.0098 |
| 1703 | 0.7216 | 0.0091 |
| 1704 | 0.7324 | 0.0083 |
| 1705 | 0.7186 | 0.0122 |

