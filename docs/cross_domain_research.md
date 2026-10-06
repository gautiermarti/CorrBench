# CorrBench beyond finance: research assessment

Prepared 2026-10-05 in response to the author's questions about a standalone
benchmark paper, non-overlapping evaluation and applications outside finance.
This is a proposal and literature assessment, not completed cross-domain evidence.

## Recommendation

Design the CorrBench interface for general real correlation matrices. Retain the
financial reproduction track and pilot brain functional connectivity first,
then spatial weather/renewable-energy dependence. Add other domains only after
identifying a useful scientific task, enough independent groups, data rights,
and an evaluation that tests more than visually plausible matrices.

Potential eventual title: **CorrBench: A Multi-Domain Benchmark for Generative
Correlation Models**. The benchmark paper would contribute data construction,
evaluation, domain transfer and utility findings. CorrFlow's paper contributes
the generative method, geometry and architectural analysis. Shared code and a
clearly attributed financial subset are appropriate; substantially repeating the
same experiments and conclusions would not create an independent contribution.

## What transfers, and what does not

For measured variables X_1,...,X_n, R_ij = Corr(X_i,X_j) has the same real,
symmetric, unit-diagonal PSD constraints across domains. The current GFT/inverse,
geometry checks, model interfaces, condition handling and many evaluation tools
can be reused for positive-definite inputs. Domain-trained weights, financial
regime labels, preprocessing and downstream metrics do not transfer automatically.

Preserve node identities and attributes. Reordering brain regions or geographical
locations must reorder their anatomical or spatial information as well.
Permutation equivariance is consistency under relabeling; it is not an assumption
that every brain region or location is scientifically interchangeable. Evaluate
both structure-only generation and generation conditioned on meaningful nodes.

For T observations, an unregularized sample correlation matrix has rank at most
min(n,T-1). Consequently n >= T introduces singular matrices, common in omics
and high-resolution imaging. The present positive-definite coordinate models
cannot simply ingest these. A benchmark must explicitly separate empirical PSD
targets from shrinkage-regularized PD targets, record the shrinkage, and measure
its effects. Larger dimension alone is not evidence of richer information.

Correlation generation models only part of dependence. It does not specify
marginal distributions, all nonlinear/tail dependence, temporal trajectories or
causal effects. Scenario generation needs additional, explicitly evaluated
observation/marginal/time models. This does not require broadening the current
CorrFlow archival manuscript to covariance modeling.

## Candidate domains

The dimensions below are proposed pilot targets, not counts already collected.

| Domain | Matrix and proposed pilot | Useful evaluation | Main issue |
| --- | --- | --- | --- |
| fMRI | Correlations of regional BOLD time series; start with the reference atlas, then 100–400 regions | Synthetic-to-real classification, augmentation, population coverage, held-out sites | Subject/site grouping, preprocessing and atlas identity, access terms |
| EEG | Electrode-signal correlations; 64-channel PhysioNet pilot | Motor-imagery utility on unseen subjects/sessions | Correlation removes power information; compare with covariance references honestly |
| Weather | Correlations of deseasonalized temperature/wind anomalies across 100–500 sites | Spatial structure, aggregation uncertainty, held-out years/regions | Seasonality, spatial dependence and shortage of independent episodes |
| Energy | Correlations of wind/solar generation or load across sites | Diversity of joint scenarios and regional aggregation errors | Marginals and time dynamics must accompany matrices for operational claims |
| Genomics | Gene coexpression or latent copula correlations in selected panels | Gene-module preservation and utility of simulated expression | Rank deficiency, donor/batch dependence, normalization and limited independent matrices |
| Wireless | Normalized spatial channel covariance across antenna elements | Beamforming robustness and channel simulation | Usually complex Hermitian; needs an extension beyond the current real geometry |

A large variable count is insufficient: we need a meaningful population of
matrices, for example across independent subjects, recording sessions, regions
or environmental episodes. Random submatrices from one giant matrix are related
views, not independent biological or environmental replicates. Establish the
independent sampling unit before choosing a generator or data split.

## DiffeoCFM: direct precedent and required baseline

Collas, Ju, Salvy and Thirion, **Riemannian Flow Matching for Brain Connectivity
Matrices via Pullback Geometry**, NeurIPS 2025:
[paper](https://arxiv.org/abs/2505.18193),
[official code](https://github.com/antoinecollas/DiffeoCFM).

The paper learns distributions of fMRI correlation matrices and EEG covariance
matrices. Its correlation variant uses normalized Cholesky coordinates; the SPD
variant uses a matrix logarithm. Flow matching is carried out in transformed
coordinates and samples are mapped back to valid matrices. These are different
matrix spaces, so its EEG covariance results must not be relabeled correlation
results.

Its experiments include three fMRI cohorts and two EEG motor-imagery datasets.
It measures fidelity/coverage and classification utility: train a classifier on
synthetic samples, then evaluate it on real held-out examples. The fMRI task is
patient/control classification, with subject-grouped splits; EEG evaluates
motor-imagery classes. The result is evidence about retained task information,
not a general demonstration that adding synthetic data beats real-only training.

For our benchmark, additionally compare real-only, real plus synthetic, bootstrap
augmentation, and simple transformed-Gaussian generation at matched budgets.
Report site/subject holdout and performance uncertainty. Generated matrices do
not constitute additional independent people or establish a causal disease model.

## Other primary literature and source candidates

1. Schefzik, Thorarinsdottir and Gneiting (2013), **Uncertainty Quantification in
   Complex Simulation Models Using Ensemble Copula Coupling**, Statistical
   Science. [Paper](https://arxiv.org/abs/1302.7149).
   Establishes the role of multivariate spatial/temporal dependence in weather
   ensembles. ECC preserves an empirical copula; it is an application precedent,
   not a learned generator of correlation matrices.
2. Chen, Wang, Kirschen and Zhang, **Model-Free Renewable Scenario Generation
   Using Generative Adversarial Networks**, IEEE Transactions on Power Systems,
   preprint 2017 / revised 2018.
   [Paper](https://arxiv.org/abs/1707.09676).
   Generates wind/solar profiles and evaluates spatial/temporal patterns using
   NREL data. Supports a power-systems use case; outputs are trajectories rather
   than correlation matrices.
3. Song et al., **scDesign3 generates realistic in silico data for multimodal
   single-cell and spatial omics**, Nature Biotechnology, online 2023 / volume
   42 (2024). [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182337/).
   Uses Gaussian/vine copulas and gene-dependence estimates in expression
   simulation. Strong motivation for dependence modeling, with a distinct target
   from learning a population distribution of empirical correlation matrices.
4. Price et al., **Probabilistic weather forecasting with machine learning**,
   Nature 637 (2025). [GenCast paper](https://doi.org/10.1038/s41586-024-08252-9).
   Generates weather fields and evaluates applications requiring joint spatial
   and temporal structure. Relevant downstream context, not a direct CorrFlow
   or correlation-matrix baseline.

Candidate data:

- [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/):
  109 volunteers, 64 electrodes, 160 Hz. The page lists Open Data Commons
  Attribution v1.0. A practical raw-signal pilot, with subject/session grouping.
- [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html):
  direct connection to DiffeoCFM; the source specifies non-commercial research
  terms and registration. Audit source/derivative terms before redistribution.
- [ERA5 single-level reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview):
  hourly global fields from 1940; the current source page lists CC-BY. Freeze
  final data versions, temporal aggregation, detrending and anomaly estimation.
  Train-derived preprocessing must be reused on held-out periods.
- [DeepMIMO](https://www.deepmimo.net/docs/quickstart.html):
  ray-traced channel data and configurable antenna systems. Defer this track
  until complex-valued matrix support is scientifically justified.

This targeted search establishes relevant precedents. It is not an exhaustive
novelty audit, and supports no claim that CorrBench is the first such benchmark.

## Two meanings of non-overlap

**Publications:** distinguish research questions, new evidence and conclusions.
CorrFlow evaluates a method; CorrBench studies reliable comparison across methods
and domains. Cite reused artifacts and identify shared experiments explicitly.

**Data:** split at the independent group level before constructing windows.
Purge any correlation window whose underlying observations cross a split boundary.
Use subject/family/site grouping in neuroimaging, subject/session grouping in EEG,
and date/event/region blocks for weather. Related atlas resolutions and nested
submatrices follow their parent into the same split. Fit preprocessing and select
hyperparameters on training/validation only.

Fully disjoint windows within every split are possible using a stride of at least
the window length; for 252-day finance windows this drastically reduces temporal
episodes. A more data-efficient main protocol can allow within-training overlap,
prohibit cross-split observation reuse, and estimate uncertainty by appropriate
blocks. Non-overlap does not itself remove temporal/spatial dependence. Include a
fully disjoint sensitivity analysis instead of calling overlapping windows IID.

## Proposed pilot sequence

1. Preserve the financial paper's exact reproduction configuration.
2. Reproduce DiffeoCFM on one fMRI cohort, including its simple Gaussian baseline,
   then compare CorrFlow under the same conditions and grouped splits.
3. Run a small open EEG raw-signal pilot to exercise a non-financial data adapter.
4. Design an ERA5 anomaly-correlation pilot with meaningful spatial context and
   enough independent year/region groups; establish utility before scaling it.
5. Freeze shared validity/coverage/conditioning/size tests and domain-specific
   utility protocols. Keep separate leaderboard tables for each domain/task.
6. Publish versioned HF configurations and the benchmark paper when reproducible
   cross-domain findings exist. Choose a single validated track for the first
   Kaggle challenge; a broad research suite need not be one giant competition.

## Implementation status

These notes and the community release contract are planning artifacts. The
existing private repository contains financial builders and common evaluation.
A local HF-compatible financial exporter is now implemented and audited; see
[the community release plan](community_release.md) and
[verification](reference_export_audit.json).
The leaderboard UI, hosted Kaggle scoring, non-financial adapters, and cross-domain
experiments remain implementation work. No data or competition has been published.
