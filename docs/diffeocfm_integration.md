# DiffeoCFM integration checkpoint — 2026-10-05

Update: the CorrFlow release repo now contains the explicit-validation adapter,
component parity audit, full-dimension pilot and a complete 200-epoch Taiwan
seed-1701 diagnostic. See `corrflow/docs/diffeocfm_financial_findings.md` in the
sibling checkout. The matched multi-model/multi-seed comparison remains pending.
The source observations below motivated that adapter and remain applicable.

Pinned [official repository](https://github.com/antoinecollas/DiffeoCFM) at
`902282acd13a67784e2096d09ee93477b268c079`. File hashes are recorded in
`research/diffeocfm_source.json`; source is cloned under ignored artifacts.

The source license permits the correlation path under MIT. `spd.py` is separately
CC-BY-NC-4.0; the correlation audit imports neither it nor `spd_fm.py`.
The official normalized-Cholesky transform agrees with CorrBench's implementation
to numerical precision. Official DiffeoGauss and DiffeoCFM both completed fitting
and sampling in the synthetic 8-by-8 smoke test, producing positive-definite,
unit-diagonal matrices. Exact tolerances/versions are in
[diffeocfm_smoke_audit.json](diffeocfm_smoke_audit.json).

This is source compatibility and an integration test, not financial/fMRI fidelity
or a clean full-upstream environment reproduction. The check extracts the unchanged
FastDataloader class from `data.py` to avoid unrelated EEG/download imports; all
loaded source hashes are checked before execution. It uses a small MLP and two
epochs solely to exercise the flow, prior, ODE and inverse-transform path.

## Details that affect a fair comparison

- The official correlation model standardizes normalized-Cholesky coordinates
  using training statistics and fits a separate **full OAS covariance Gaussian
  per class** as its starting distribution. Include DiffeoGauss as a serious
  baseline; an isotropic or diagonal-Gaussian control is not this implementation.
- Its vector field is a float64 SELU MLP with categorical one-hot labels. The
  fMRI default has one 512-wide hidden layer, 200 epochs, batch size 64, AdamW
  with zero weight decay and learning rate .001. Continuous performance scores
  are not supported by the unmodified model.
- The current `fit` creates an internal random 90/10 split. The outer fMRI script
  groups by subject, but the inner random split does not preserve those groups.
  A benchmark adapter must accept an explicit validation set and keep all fitted
  transforms/prior statistics restricted to training. Record this adaptation.
- The original flow retains last-epoch weights; it does not select the best
  structural validation checkpoint. Preregister both the faithful original
  setting and any common selection-rule adaptation rather than conflating them.
- Sampling uses adaptive dopri5 with atol/rtol 1e-5. Six requested output times
  are **not six vector-field evaluations**. Instrument actual function calls,
  endpoint validity, sampling time and inverse-transform time.

## Next matched financial run

1. Use the frozen Taiwan return-derived train/validation data and canonical Ward
   ordering for order-sensitive image/vector models. Preserve the original matrix
   and IDs; report ordering costs and sensitivity. No newly claimed test holdout.
2. First compare categorical information only across DiffeoCFM, cCorrGAN and
   CorrFlow. Keep the existing categorical-plus-continuous Taiwan paper study
   separate; an extended conditional DiffeoCFM must be explicitly named.
3. Include class-conditional replay and the exact transformed OAS Gaussian.
   Record full structural, entry, spectral, coverage and downstream diagnostics.
4. Add explicit validation and checkpoint selection with a fixed independent RNG;
   measure one small full-dimension run before declaring the multi-seed budget.
5. Freeze configurations/seeds and compatible package revisions before the full
   comparison. Keep unfavorable outcomes and existing paper results intact.

## Reproduce the completed integration check

Install CorrBench plus the computational dependencies `pyriemann`, `torchcfm`,
`torchdiffeq` and `POT` in a research environment; exact versions used appear in
the audit. Clone the pinned upstream repository, then run:

```bash
python research/verify_diffeocfm.py --upstream /path/to/DiffeoCFM \
  --output /new/path/diffeocfm_smoke.json
```

No financial or brain data are downloaded or loaded by this command.
