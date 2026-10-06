# ABIDE 110 named-region sensitivity results — 2026-10-06

Both declared cohorts and all 22 control arms complete. This is a sensitivity
analysis on inspected sites. No 110-region learned generator was fitted.

| Cohort / evaluation | Real-only AUC | Replay AUC ± draw SD | Gaussian AUC ± draw SD |
| --- | ---: | ---: | ---: |
| Same 601 / original 160 NYU subjects | 0.7725 | 0.7543 ± 0.0059 | 0.7724 ± 0.0049 |
| Expanded 762 / all 166 NYU subjects | 0.8013 | 0.7814 ± 0.0101 | 0.7682 ± 0.0122 |
| Expanded 762 / common 160 NYU subjects | 0.8059 | 0.7822 ± 0.0103 | 0.7707 ± 0.0123 |

## Augmentation differences from each cohort's real-only control

| Cohort / evaluation | Control | Δ AUC [paired subject 95% interval] |
| --- | --- | ---: |
| Same 601 / 160 | replay | -0.0182 [-0.0313, -0.0050] |
| Same 601 / 160 | gaussian | -0.0002 [-0.0190, +0.0187] |
| Expanded 762 / 166 | replay | -0.0199 [-0.0395, -0.0030] |
| Expanded 762 / 166 | gaussian | -0.0330 [-0.0590, -0.0095] |
| Expanded 762 / common 160 | replay | -0.0237 [-0.0424, -0.0050] |
| Expanded 762 / common 160 | gaussian | -0.0352 [-0.0611, -0.0110] |

## Paired source/cohort comparisons

- 110 versus 111 labels on the same cohort: real-only AUC 0.7654 → 0.7725; Δ +0.0071 [-0.0078, +0.0216].
- Expanded versus original training/validation cohort on the same 160 test subjects: real-only AUC 0.7725 → 0.8059; Δ +0.0333 [-0.0008, +0.0681].

Five draws measure generation variation; intervals condition on fitted models
and this one site, with no multiple-comparison correction. Expanded-cohort
scores also change the population; the common-subject comparison retains
training/validation membership changes. These analyses do not establish
generalization to new sites or clinical usefulness.

## Reconstruction and selection evidence

All 780 source hashes checked. Reconstructed 762 named-region scans; 18 still
have another constant region. Original 601-subject arrays remain unchanged.
Independent adapter calculations reproduce all 762 targets using the same
sklearn OAS estimator. Common-subject arrays match exactly between cohorts.
All 22 selected validation scores reproduce exactly and all feature scalers
use real training subjects only. Both selections were sealed before scoring.

The frozen 111-label learned-augmentation results remain separate. This study
does not establish how retrained learned generators perform at 110 regions.

Protocol: [declared settings](abide_named110_protocol_20261006.md).
Detailed counts, all control scores and hashes: [abide_named110_results_20261006.json](abide_named110_results_20261006.json).

Local artifacts: `artifacts/abide_named110_20261006/`. Data redistribution
uncleared; no participant matrices or atlas images were uploaded.
