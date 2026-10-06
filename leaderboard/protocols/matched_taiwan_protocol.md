# Matched categorical Taiwan comparison — 2026-10-05

This protocol extends the validated single-seed DiffeoCFM workflow to CorrFlow
and the categorical cCorrGAN reimplementation. It answers a method comparison
on the frozen historical case study. Previously inspected dates remain
retrospective evidence, and the existing continuous-conditioned paper table
remains a separate experiment.

## Frozen information and selection

- Inputs: audited return-derived Taiwan matrices, n=80, T=252, existing frozen
  chronological splits. All models use the same Ward-ordered matrices. Scalar
  GFT normalization preserves CorrFlow equivariance, so supplying this order
  does not introduce an architectural dependence on asset labels.
- Model inputs: categorical regime only. Ignore continuous performance scores;
  do not refit regime thresholds, data construction, splits or matrix ordering.
- Training seeds 1701–1705. Fit every transform, prior and feature scale on
  the 13,600 training rows. Validation has 3,920 rows.
- Selection subset: 100 validation rows per class, NumPy default_rng(44301).
  Fixed model-specific latent draws, sampling seed 44301, projection seed 44301.
  Minimize the existing full 21-feature SWD, retaining the first tie. Preserve
  raw final weights and selected weights separately. EMA is not selected here.
- CorrFlow: existing small paper architecture (48 channels, three blocks),
  AdamW .0003, batch 64, 20 epochs, bfloat16 forward; validate every epoch.
- DiffeoCFM: upstream-style 512-wide float64 SELU MLP and full class-wise OAS
  Gaussian, AdamW .001, batch 64, 200 epochs; validate epoch 1 and every 10.
- cCorrGAN: native categorical architecture (latent 80, channels 128), Adam
  .0002, batch 64, 200 epochs; validate epoch 1 and every 10. This continues the
  audited reimplementation; original discriminator/training fidelity remains
  unestablished and is not implied by a longer training budget.

These are method-specific established budgets, not equal compute or exhaustive
tuning. The number of checkpoint comparisons is approximately matched (20/21).
The full-dimension pilots exercise execution and measure cost. Their extra final
epochs must not be candidates in the full training run's selection schedule.
Do not reduce long GAN budgets based on observed unfavorable fidelity.

## Sampling and evaluation

CorrFlow uses 25 Heun steps (50 actual field calls), batch 32 and a converged
float64 GFT inverse, tolerance 1e-8, cap 1,000. DiffeoCFM uses adaptive dopri5,
atol=rtol=1e-5, batch 64 and its float64 normalized-Cholesky inverse. cCorrGAN uses
batch 32, upper-triangle symmetry/unit diagonal and converged float64 nearest
correlation repair, tolerance 1e-6, cap 1,000, strict-PD floor 1e-6. Count inverse
and repair work in full sampling cost. Record actual NFE; equal sample counts
do not imply equal model evaluations or parameter counts.

Reproduce validation selection scores before considering the comparison ready.
After configurations and implementations pass checks and are frozen, evaluate
the existing test split on 500 rows per class, selection seed 2027, with all
methods receiving those same 1,500 regimes. This is a historically inspected
test period, not a new temporal holdout. Test metrics are not used to change
hyperparameters, select epochs or choose seeds. Score selected and final models.

Use five generation seeds 2027–2031 paired with training seeds 1701–1705 for
test draws, while keeping test-row selection fixed at 2027. Simple priors/replay
receive those same five generation seeds and fixed cases. Sampling tools need
separate selection and generation seeds for this evaluation; validate that the
selected row arrays are identical across all candidates before scoring.

Use CorrBench's common float32 scored representation for the main comparison;
retain native float64 validity and convergence reports. Record full structural
and energy distances, GFT/orbit metrics, entry/spectral subsets, per-regime
behavior, coverage, classifier separation, nearest-train distances and portfolio
functionals. Report training-seed sample standard deviations as training
variability, not uncertainty over independent market episodes. Preserve negative
and mixed results. Portfolio functionals assume unit volatility; no backtest or
augmentation utility is implied.

## Evidence and operational record

Store source/data/configuration hashes, selected indices, checkpoints, learning
curves, samples, timing and runtime versions. Keep all five runs, even poor ones.
Scripts refuse overwrites. Concurrent GPU runs may improve throughput but their
wall times are not isolated speed benchmarks; record concurrency and collect
warmed single-process timing separately before ranking speed.

The DiffeoCFM seed-1701 run already exists and is reused without alteration.
Remaining artifacts are under the CorrFlow repo's
`artifacts/matched_taiwan_20261005/`. Configuration files are
`corrflow/configs/flow_taiwan_categorical.yaml`,
`corrflow/configs/diffeocfm_taiwan_categorical.yaml`, and
`corrgan/configs/ccorrgan_taiwan_categorical.yaml` in the respective checkouts.
Evidence and implementation updates remain private until the coordinated release.
