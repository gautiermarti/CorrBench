# Exploratory weather generation extension — 2026-10-05

Declared before launching this model fit, after the initial spatial-baseline
study and its test outcomes were inspected. This is an exploratory extension
on the same historical periods, not an untouched confirmation.

Use the existing 360 OAS training matrices, 60 validation matrices and 108
chronological test matrices. Twelve categorical conditions encode the three
known regions crossed with four quarters. Matrix order is the declared grid-cell
order. No South America model is fitted or evaluated in this extension.

The bounded learned comparison is the pinned DiffeoCFM correlation adapter,
unchanged 512-wide float64 field, full class-wise OAS Gaussian prior, 200 epochs,
learning rate .001 and batch 64. One training seed, 1701. Select the minimum
structural validation SWD among the usual evaluated epochs (every ten, plus the
trainer's initial evaluation if present), using all five validation years per
condition and seed 44301. Retain final weights as sensitivity. This is not a
matched CorrFlow comparison or an equal-compute claim.

Generate 200 matrices per condition with seeds 2027–2031 for the Gaussian,
selected DiffeoCFM and final DiffeoCFM. Replay uses the same count and seeds.
Condition means, pooled means and the original train-fitted spatial kernel are
deterministic controls. Preserve every result. The initial replay table used
1,000 draws per context; this new study recomputes replay at the matched count
and remains separately versioned.

Keep the existing aggregate-variance utility definition: whole-patch and four
quadrant area-weighted variances under unit local scales; 1-Wasserstein against
empirical quarterly targets; train-mean variance normalization; equal averaging
across functions and region/quarter contexts. Store all per-context values.
Report structural fidelity separately, retaining node ordering in the utility
calculation. Five generation seeds do not replicate model training.

Audit code, data and checkpoint hashes, complete epochs and selection before
generation/evaluation. No tuning follows test utility. Only nine test-year
groups are available; generator SD does not describe uncertainty over future
years. No claim about forecasting temperatures, extremes or energy production
follows from this dependence-only proxy.
