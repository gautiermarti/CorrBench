# ABIDE training-seed replication — 2026-10-05

Post-pilot atlas audit (October 6): the unchanged 111-label construction includes
unnamed two-voxel label 3455, which alone caused 161 of 179 subject exclusions.
The original results remain a frozen-cohort replication. See
[the audit](abide_atlas_audit_20261006.md); removing this label requires a new
dataset/estimator study and does not create an uninspected site.

Declared at 11:25 UTC, after the seed-1701 pilot and its NYU outcomes were known,
before launching the additional seeds. This is replication of the unchanged
pilot on the **same previously inspected test site**, not a fresh holdout.
The original pilot artifacts and report remain immutable.

## Fixed expansion

Run training seeds 1702–1705 for all three original families: node-identity
CorrFlow, anonymous CorrFlow and DiffeoCFM. Combine them with seed 1701, so each
learned family has five training replications. Use exactly the original data,
configurations, 20/20/200 epochs, CPU precision, class conditions, validation
subset, structural selection rule and sampling settings. No architecture,
hyperparameter or estimator change is permitted in this expansion.

For every training seed, retain generation seeds 2027–2031 and 175 draws per
diagnosis. Evaluate selected weights for all models and final DiffeoCFM weights
as the existing sensitivity arm. Preserve every result. Do not choose or omit
training seeds after reading their test outcomes.

Repeat the original validation-only classifier selection with the same feature
map, training-only scaler, C grid, balanced class weights and augmentation count.
Test each saved selection once. Reuse the original real-only, five replay and
five fitted-Gaussian arms as fixed controls; these are not five independent
generator fits. Add synthetic-only classification at generation seed 2027 for
each new trained model and final DiffeoCFM arm, following the original diagnostic.

Before scoring, check frozen code/data/config hashes, completed epochs, the
minimum-validation checkpoint, ROI identity order and generation provenance.
Persist argument-array jobs, hashes, logs and resumable ledgers. CPU queues are
separate from the frozen financial GPU experiments.

## Reporting

Report the average over five latent draws within each training seed, then the
mean and sample SD across training seeds. Report each training seed separately.
Generation seeds are nested draws, not extra independent training replications.
Use joint subject resampling for paired task comparisons. Any interval combining
training-seed and subject resampling conditions on this single site and the
limited set of fitted generators; it is not uncertainty over unseen sites.
Mark comparisons as exploratory, unadjusted and based on an already inspected
test population. Keep source/site/observation-length and exclusion limitations.

No early-stopping by wall-clock results is allowed. If the six-hour work window
ends before all declared jobs complete, report completed versus running jobs
without presenting a partial seed set as the final five-seed result.
