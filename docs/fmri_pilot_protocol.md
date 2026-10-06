# ABIDE PCP pilot protocol — 2026-10-05

**Post-pilot source note, October 6:** the frozen 111-label mask contains 110
officially named regions and unnamed two-voxel label 3455. That label alone
caused 161 of 179 constant-series exclusions. All source hashes and exclusion
membership reproduce. The original protocol and results remain unchanged;
see [the atlas audit](abide_atlas_audit_20261006.md) for the limitation and separate
110-region follow-up. This was not part of the original prespecification.

Status: source and metadata audited; protocol declared before retrieving ROI
series or fitting models. This is a local non-commercial research pilot. It is
not yet an HF release, a clinical validation, or a reproduction of DiffeoCFM's
exact fMRI dataset. The financial archival paper remains independently scoped.

## Source and permissions

Use the official [ABIDE Preprocessed Connectomes Project repository](https://github.com/preprocessed-connectomes-project/abide)
and its public FCP-INDI S3 location. Its
[download guide](https://github.com/preprocessed-connectomes-project/abide/blob/master/download_abide_preproc_guide.txt)
explicitly provides public access to derivatives. This differs from the original
NITRC download path, which requires registration. Do not bypass NITRC access
controls or download the mixed ABIDE/ADNI/OASIS archive from another paper.

ABIDE's [usage agreement](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
permits non-commercial research and links Attribution-NonCommercial-ShareAlike
terms; dataset/site credits and funding acknowledgments must accompany a release.
Record the exact applicable license version and derivative/atlas requirements
before redistribution. Public accessibility alone does not establish permission
for commercial reuse, an unrestricted HF mirror, or a Kaggle competition.

Input starts at preprocessed regional BOLD time series, not raw scanner images.
Select `cpac/filt_noglobal/rois_ho`: Harvard-Oxford ROI means, band-pass filtering,
without global signal regression. Preserve pipeline/strategy/atlas identifiers,
source file hashes and the exact header ordering. Do not mix preprocessing
strategies as independent samples. Original ABIDE and PCP citations are required.

## Independent units and predeclared quality filters

One complete scan per public subject ID; never split windows from the same
subject across partitions. The audited metadata contains 1,112 unique IDs.
Apply the Nilearn documented quality rule: rater 1 OK; anatomical and functional
rater 2 each OK or maybe; anatomical and functional rater 3 each OK. Require a
real FILE_ID and mean framewise displacement < .2, as in the PCP download guide.
This yields 780 eligible metadata rows before file availability/numerical checks.

Reject missing/unreadable files, inconsistent ROI headers, nonfinite values,
fewer than 50 time points, or any constant ROI series. Record every excluded
subject and reason. Do not choose/drop brain regions based on test correlations.
Keep the full common Harvard-Oxford header if feasible; a different atlas or
node set would require a new protocol rather than silent feature selection.

## Splits and matrices

Hold NYU out for test (168 eligible metadata rows), UM_1 and UM_2 together for
validation (108), all remaining sites for training (504). These site choices
precede matrix/model inspection. Group physical sites across numbered acquisition
cohorts. They test transfer to particular sites, not uncertainty over all future
sites. Preserve diagnosis, age, sex, motion and site as metadata. Diagnosis is
the only generator condition in the first comparison: control=0, ASD=1.

Center and standardize each ROI within its own scan. Save empirical Pearson
correlation as a PSD diagnostic target, with rank bounded by T-1. The primary
PD target uses per-scan Oracle Approximating Shrinkage covariance followed by
diagonal normalization; save shrinkage and minimum eigenvalues. This explicitly
models the distribution of a regularized estimator, not latent neural coupling.
OAS is a deterministic per-observation transform and is not fitted across test
subjects. All dataset-level scales, priors, models and tuning use train/validation.

## First utility test and structural comparisons

Use a prespecified regularized logistic classifier on the ordered upper triangle
of Fisher-transformed entries. Standardization and classifier C are fitted using
training data and selected using the held-out validation site. Assess real-only,
real plus class-balanced training replay, real plus transformed-Gaussian draws,
and real plus learned draws; include synthetic-only as a diagnostic, not a
substitute for improvement over real-only. Use the same augmentation count and
seed policy across generators. Report ROC-AUC and balanced accuracy, with paired
subject-bootstrap uncertainty conditional on the one held-out site. No clinical
diagnostic claims follow from this research proxy.

For the first execution, fix classifier C to a validation-selected value from
{.001, .01, .1, 1, 10}, breaking ties toward the smaller C. Use L2 logistic
regression, balanced class weights, liblinear, max_iter=2,000 and threshold .5.
Clip correlations to +/- (1-1e-7) before the entry-wise Fisher transform; fit
feature means/scales on real training subjects only, shared by all arms.
Each augmentation adds ceil(N_train/2) matrices per diagnosis, seeds 2027--2031.
Select C separately per arm on validation ROC-AUC, save selections before opening
test observations, and use 1,000 paired, diagnosis-stratified subject bootstrap
draws for the test difference from real-only. The bootstrap is conditional on
the trained model and this site; it is not uncertainty over sites or training.

Also evaluate class-conditional matrix distances, spectra, validity, support
coverage and nearest-training distances. Preserve named ROI connectivity:
permutation-invariant structural fidelity alone is insufficient for this task.

The current CorrFlow model has global conditions but no ROI identity/attributes.
With exchangeable noise, its equivariant vector field induces an exchangeable
matrix law. It therefore cannot generally reproduce a distribution tied to fixed
named brain regions. Record this architectural limitation explicitly. Compare
fixed-order DiffeoCFM/Gaussian first; any node-conditioned CorrFlow extension is
a new model requiring simultaneous matrix/node-attribute permutation tests.
Do not silently Ward-sort brain regions or interpret permutation symmetry as
scientific interchangeability of regions.

Initial DiffeoCFM pilot: unchanged adapter/512-wide float64 architecture, 200
epochs, AdamW .001, batch 64, seed 1701. Select on 30 fixed validation cases per
class, seed 44301, full structural SWD at epoch 1/every 10. This small declared
pilot checks the fixed-ROI generation/utility workflow; it is not five-seed
evidence or an exhaustive configuration search. Include its fitted OAS Gaussian
prior with the same augmentation counts and five generation seeds.

An explicitly experimental CorrFlow extension supplies one-hot atlas ROI identity
to the initial node states. This preserves simultaneous matrix/attribute
permutation equivariance while permitting a law tied to named ROIs. Compare
with the same network with no node identity. Both use 32 hidden channels, two
blocks, AdamW .0003/weight decay .0001, batch 16, 20 epochs, seed 1701, scalar
training GFT normalization and structural validation every epoch on the same
30/class cases. Sample with 25 Heun steps, float32 field and converged float64
inverse. This is a small CPU feasibility/utility ablation, not the financial
paper architecture, equal-compute comparison, or unseen-atlas generalization.
Freeze these settings before examining test utility.

## Pilot acceptance

Require a reproducible downloaded-source manifest, disjoint subject/site splits,
documented numerical exclusions, PD/rank audits, a complete real/replay utility
baseline, and a measured generation pilot. A valid matrix export alone does not
complete the non-financial milestone. Public packaging and multi-site replication
are later work; retain unfavorable utility results.

## Construction result

All 780 eligible public ROI files downloaded with hashes. The predeclared
constant-ROI rule excluded 179 subjects; 601 remain (349 train, 92 validation,
160 test). Every accepted scan retains the same 111 atlas IDs. All regularized
targets are PD; the smallest eigenvalue across splits is .0132165. Empirical
correlations are numerically rank deficient for 345 training and all 160 test
subjects, versus none on validation. Time-point counts differ substantially by
site (validation 296, test 176); filtering further reduces effective temporal
degrees of freedom. Site transfer therefore also shifts the observation process.
Do not equate these empirical ranks with a count of independent neural signals.

The exclusions change site representation and require reporting, rather than
silent cleanup. Raw-series acquisition, hashes, exclusions, ROI IDs, matrix
estimator diagnostics and split hashes are saved locally under
`artifacts/abide_source_20261005/` and `artifacts/abide_ho_20261005/`. These are
local research artifacts with the source-specific restrictions above.
