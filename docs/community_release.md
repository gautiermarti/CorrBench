# CorrBench: reference dataset, leaderboard, and community challenge

Decision recorded 2026-10-05. Develop privately on GitLab, then coordinate the
public GitHub, Hugging Face, paper, and tutorial releases. Hugging Face dataset
hosting, a maintained leaderboard, and Kaggle Community Competition support are
now explicit deliverables. No public competition or dataset has been launched.
The author also requested a suite of domain-specific sub-datasets. The
[application survey](applications_survey.md) covers 18 fields; inclusion of each
candidate depends on a concrete dataset, scientific task and reproducible evidence.

## One benchmark, distinct artifacts

| Artifact | Responsibility |
| --- | --- |
| CorrBench Git repository | Return-to-matrix builders, versioned protocols, metrics, split manifests, export tools and result records |
| Hugging Face dataset | Downloadable, versioned matrices and provenance; return panels only where redistribution is permitted |
| Hugging Face leaderboard Space | Display reviewed result records grouped by exact dataset, task, split and metric version |
| Kaggle Community Competition | A dated challenge using a frozen CorrBench task and an independently protected evaluation set |
| CorrGAN and CorrFlow repositories | Model training, checkpoints, sampling and reproducible benchmark submissions |

The paper narrative stays: adjusted returns -> matrix estimation and conditions
-> fixed splits -> generation -> common evaluation. Precomputed matrices are a
convenience download, with their construction traceable to returns. Legacy S&P
matrices stay a separate historical configuration because dates, assets and
underlying returns are unavailable.

## Dataset contract

Start with Taiwan and Japan at n=80. Publish named configurations for markets and
dimensions. Paired n=20/40/80/160/200 draws require persistent parent identifiers;
nested matrices and overlapping windows do not count as independent observations.
Keep the original paper splits immutable and label their prior inspection and
window overlap. Improved purged chronological splits belong to a new protocol.

Each release contains a dataset card, data rights assessment, source references,
return adjustment convention, estimator, lookback, regime definition, asset
ordering, split definitions, source and output hashes, and exact builder revision.
Each matrix retains its asset IDs, window end date, condition, and source-row ID.
The source calendar plus lookback determines the window start; do not infer it
by subtracting calendar days. No machine-specific source paths are exported.

The implemented `corrbench-export-hf` command stages Parquet files and a card
locally, without uploading. It verifies NPZ checksums and preserves all matrix
entries and provenance. Pending rights produce an explicitly blocked staging
bundle. See [format](reference_format.md) and [audit](reference_export_audit.json).
Software's MIT license does not establish rights to financial data.
Competition-training exports allow only train/validation and omit test manifest
information; this is a packaging safeguard, not proof against reconstructing
hidden correlations from publicly obtainable return data.

## Leaderboard contract

The research leaderboard is an enduring, open reproducibility resource. Kaggle's
leaderboard is specific to a competition and its private evaluation protocol.
Do not merge their rankings. The planned initial static Space will be generated
from reviewed JSON records, without executing arbitrary participant code.

Partition every ranking by protocol ID, dataset manifest hash, market/size,
task, split, and metric version. A record includes model revision, evaluation
revision, generated artifact hash, seeds, sample count, score and uncertainty,
validity, coverage, repair/timing details, hardware and evidence links. Label
author-reported and independently reproduced results separately. A maintained
admission review is required: a field reading 'verified' is not verification.
Keep raw per-seed reports; never rank rows computed with incompatible settings.

## First Kaggle task

Working title: **CorrBench: Generate Realistic Correlation Matrices**.

Begin with fixed-size n=80 generation under three market-regime conditions.
Participants receive return-derived training matrices, conditions, a builder,
local evaluation, and replay/PCA/model examples. Inputs request independent draws
under a specified condition. This evaluates conditional distributions rather
than forecasts for named dates or a one-to-one match to hidden matrices.

One CSV row represents a complete generated matrix: an opaque draw ID and the
3,160 strict lower-triangle values in NumPy row-major `tril_indices(80, -1)` order.
The diagonal is one; symmetry follows from unpacking. Reject nonfinite values,
incorrect IDs, wrong columns and matrices outside the declared PSD tolerance.
Methods may use repair themselves; report its cost and magnitude. Do not silently
repair submissions in the scoring function. The solution owns conditions and
public/private membership; participants cannot relabel their requested draws.

The scalar ranking score must compare distributions within conditions, with
predeclared aggregation and training-only normalization. Per-row MSE is unsuitable
because a generated draw has no unique matching real target. Existing structural
feature distances are candidates, not an approved competition score. Before
freezing the metric, test whether it rewards replay, collapse, excessive noise,
or matching summaries while destroying joint structure. Compare structural,
spectral and portfolio diagnostics and held-out discrimination. Do not select a
score to favor CorrFlow. Include validity, coverage and sampling cost alongside
the scalar rank on the research leaderboard.

Use intact date/time groups for public/private reference splits and purge
overlapping return windows. The paper's inspected Taiwan/Japan periods are not
an untouched final challenge holdout. Publishing their raw returns allows hidden
matrices to be reconstructed. A credible blind contest needs genuinely protected
data or frozen generators evaluated on future data. Merely hiding dates or
withholding a CSV is insufficient when the source returns are public.

A CSV leaderboard evaluates submitted samples, not the reusable stochastic
generator. Finalist reproduction should require code/checkpoints and organizer
reruns with fresh seeds under predeclared rules. Native Community code-rerun
support must be confirmed before promising it; current CLI documentation marks
private rerun bundle updates as requiring Kaggle admin access.

Before launch, exercise the complete custom metric in an unpublished Kaggle
sandbox: full and public/private-subset scoring, condition coverage, CSV size,
CPU time, memory, error messages and deterministic results. Offline validation
does not establish that the hosted scoring path works.

## Deliverables and order

1. Freeze the paper's reproduction release and complete model fidelity checks.
2. Stage the HF dataset, resolve source-specific rights, verify a fresh-machine
   download/build/evaluate path, and archive immutable release identifiers.
3. Populate the reviewed leaderboard from reproducible reports, including replay,
   statistical baselines, CorrGAN, cCorrGAN, CorrFlow and matched diffusion.
4. Pilot the competition adapter and candidate scores privately. Establish a new
   holdout and rules for external data, resources, submissions and reproduction.
5. Coordinate GitHub/HF publication with paper links and tutorial notebooks.
6. Launch the Kaggle challenge once its separate readiness criteria are met.

The archival paper does not depend on completing a public competition. Cite
released software/data accurately; describe a future challenge as planned.

## Could CorrBench support its own paper?

Yes, as a potential dataset-and-benchmark contribution with independent scientific
claims. Current Taiwan/Japan n=80 data comprise 24,408 + 8,672 matrices, rebuilt
from stored return panels. That is a useful foundation, not evidence of 33,080
independent market episodes. Packaging alone does not establish paper readiness.

The author subsequently asked about a broader benchmark across scientific
domains. The direction and literature are recorded in
[the cross-domain assessment](cross_domain_research.md). Finance, an ABIDE brain
connectivity pilot and a NOAA weather pilot now have implemented adapters,
reconstruction audits and task results. The latter two remain limited pilot
studies; neither has established a learned-generator utility gain. Their initial
findings already appear in the extended CorrFlow draft and cannot constitute the
same new contribution again in a benchmark paper. A third financial market is
useful but is not an arbitrary acceptance requirement.

The paper should answer: which evaluation choices distinguish realistic,
generalizable correlation generators from methods that merely match selected
summaries? Study temporal/market shift, conditioning, dimension transfer, nested
consistency, and metric failure cases. Use uncertainty estimates that respect
date-block dependence. Separate these contributions from CorrFlow's geometry and
architecture. The benchmark should remain useful if a competitor wins.

Before committing to a standalone submission, require:

- A prior-art comparison establishing a specific unmet benchmark need.
- Accessible, documented data with resolved redistribution rights and permanent
  versions, including machine-readable metadata required by the selected venue.
- Strong, faithfully reproduced baselines with matched data/budgets and multiple
  seeds; existing GAN fidelity issues must be resolved or accurately delimited.
- A fresh evaluation protocol, analysis of metric reliability, and at least one
  substantive scientific finding beyond a leaderboard ordering.
- Independent reproduction of a full result and a maintenance/submission policy.

NeurIPS renamed its relevant track Evaluations & Datasets in 2026; its scope
explicitly includes benchmark design, evaluation audits and negative findings.
Its 2026 reviewer guidelines require reviewer-accessible datasets and Croissant
metadata for dataset contributions. These are useful design standards, not a
promise of acceptance or a claim that a future call/deadline is announced.

## Official platform and publication references

Checked 2026-10-05:

- [HF data configurations](https://huggingface.co/docs/hub/datasets-data-files-configuration)
- [HF dataset cards](https://huggingface.co/docs/hub/datasets-cards)
- [HF static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)
- [Kaggle Community setup and custom metrics](https://www.kaggle.com/docs/competitions-setup)
- [Kaggle CLI creation and rerun constraints](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competition_creation.md)
- [NeurIPS 2026 evaluation scope](https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/)
- [NeurIPS 2026 reviewing guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines)
