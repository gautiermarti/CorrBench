# ABIDE pilot results — 2026-10-05

Completed local research pilot; no clinical validation or public release.
One training seed per generator; five generation seeds; one test site.
Real-only ROC-AUC: 0.7654; balanced accuracy: 0.7016.

| Augmentation | AUC mean ± generation SD | Δ AUC, subject-bootstrap 95% CI | Structure SWD ↓ | Named-ROI SWD ↓ |
| --- | --- | --- | --- | --- |
| Training replay | 0.7472 ± 0.0054 | -0.0182 [-0.0316, -0.0052] | 0.467 | 0.343 |
| Transformed Gaussian | 0.7539 ± 0.0076 | -0.0115 [-0.0314, +0.0077] | 1.174 | 0.380 |
| DiffeoCFM selected | 0.7441 ± 0.0141 | -0.0213 [-0.0422, -0.0008] | 1.219 | 0.353 |
| DiffeoCFM final | 0.7397 ± 0.0073 | -0.0257 [-0.0477, -0.0035] | 1.349 | 0.333 |
| Node-aware CorrFlow pilot | 0.7573 ± 0.0178 | -0.0081 [-0.0435, +0.0268] | 0.966 | 0.393 |
| Anonymous CorrFlow pilot | 0.7628 ± 0.0167 | -0.0026 [-0.0388, +0.0328] | 1.008 | 0.752 |

Every generated matrix is PD. All bootstrap intervals are conditional on the
fitted models and NYU test site, and are unadjusted for multiple comparisons.
Generation seeds are not independent training replications. Structural and
named-ROI SWD have different representations; their scales differ.
Fidelity uses 70 test subjects per diagnosis; utility uses all 160.

No method establishes improved augmentation utility in this pilot. Better
anonymous structural fidelity is insufficient for named-ROI task fidelity.
The node-aware architecture improves its named-ROI distance relative to the
anonymous ablation, but the small feasibility configuration is not yet a
validated neuroscience generator. Retain the simple Gaussian baseline.

Synthetic-only diagnostics (one generation seed):

| Generator | ROC-AUC | Balanced accuracy |
| --- | --- | --- |
| gaussian_synthetic_only_2027 | 0.7432 | 0.6238 |
| diffeocfm_synthetic_only_2027 | 0.6973 | 0.6222 |
| nodeflow_synthetic_only_2027 | 0.5519 | 0.5667 |
| anonymous_flow_synthetic_only_2027 | 0.4770 | 0.5357 |

Protocol: docs/fmri_pilot_protocol.md. Dataset: 349 training subjects,
92 validation subjects from UM_1/UM_2, and 160 NYU test subjects; 111 ROIs.
Inputs are official ABIDE PCP preprocessed regional BOLD series. Numerical
exclusions, OAS estimator, site-dependent observation lengths, and source
permissions accompany interpretation. No test-driven retraining followed.
Descriptive fidelity diagnostics were added after utility evaluation began;
they are not retrospective checkpoint selection criteria.
