# Scientific workflows and evidence

Run the commands below from the repository root. Real inputs require the
source-specific permissions described in the linked reviews.

## Reproduce from returns

Supply an audited Parquet panel with date, instrument ID, return, and liquidity
columns configured in YAML. Taiwan uses returns relative to exchange reference
prices with a raw-close fallback ([exact convention](taiwan_return_convention.md)); Japan uses
split-adjusted returns and does not include cash dividends. Source conventions
are part of the benchmark, not interchangeable preprocessing choices.

```bash
corrbench-build --config configs/taiwan_final.yaml \
  --source /path/to/taiwan_returns.parquet --output data/taiwan
corrbench-order --input data/taiwan --output data/taiwan_ward
for split in train validation test; do
  corrbench-cache-gft --input data/taiwan/${split}.npz \
    --output data/taiwan/gft/${split}.npz --device cpu
done
```

The builder records source/output hashes, dates, constituents, and training-only
regime thresholds. Ward ordering retains sample correspondence. Image models
use the ordered copy; CorrFlow supports either ordering with the corresponding
GFT cache. Each comparison freezes its representation and uses the same
underlying return windows and evaluation cases.

Chronological endpoints do not imply disjoint observation windows. The
[actual-session audit](market_window_overlap.md) records adjacent-boundary
overlap, nine shared train/test sessions in the historical Japan diagnostic,
and the cost of purged or fully disjoint views. These views do not turn inspected
periods into a new holdout.

`corrbench-build-jpx` constructs the Japanese return panel from a separately
obtained JPX archive. `corrbench-build-legacy` imports the 18,000 precomputed
cCorrGAN matrices for historical reproduction. That archive lacks dates and
asset IDs, so its random split is not a chronological return-derived benchmark.

## Evaluate

`corrbench-evaluate` measures generated matrices against a frozen dataset;
`corrbench-controls` fits replay and Gaussian/PCA controls. Each command has
`--help`; further audits are available as `python -m corrbench.cli.<module>`.
Image evaluation must include raw validity, converged repair, repair magnitude,
and sampling/repair time. Report fidelity, coverage, detectability, and novelty
together. Exact validity does not establish realism or forecasting skill.

The completed [five-seed categorical comparison](matched_taiwan_results.md)
uses identical cases and audited validation selection. CorrFlow has lower
structural and portfolio distances than the adapted DiffeoCFM under the tested
budgets; coverage and final-checkpoint rankings remain mixed. The completed five-fit cCorrGAN reconstruction has structural SWD 4.0390 ± .1537
and zero precision after repair; the full negative result is retained.

## Data and releases

Real prices, return panels, generated samples, and checkpoints are not committed.
`python examples/make_synthetic_panel.py --output data/synthetic` provides a small
synthetic return panel and configuration for exercising the construction path.
The model repositories pin this package to a Git revision. Git owns builders,
protocols, and metrics; Hugging Face will distribute cleared, versioned data and
checkpoints with cards and hashes. See [release criteria](release_contract.md).

Stage a reference dataset locally:

```bash
corrbench-export-hf --data data/taiwan --metadata configs/releases/taiwan_paper.yaml \
  --output artifacts/taiwan-reference
```

This writes Parquet, a dataset card and integrity/provenance manifests without
uploading. See [the format and rights status](reference_format.md). All
33,080 frozen Taiwan/Japan matrices and associated arrays passed exact round-trip
checks, and both staged datasets loaded through the HF client in offline mode;
[verification](reference_export_audit.json) records the scope and hashes.

The [community release plan](community_release.md) defines dataset hosting,
leaderboard records and competition requirements. The
[cross-domain assessment](cross_domain_research.md) examines a standalone
benchmark paper, brain connectivity, weather/energy and other applications.
The [application survey](applications_survey.md) explains concrete uses in
18 fields and distinguishes established workflows from proposed matrix-generation
extensions. CorrBench is planned as an umbrella of domain-specific sub-datasets
with shared interfaces and separate scientific protocols.
The first non-financial adapter now constructs an ABIDE PCP fMRI pilot from
regional time series, with subject/site-disjoint splits, fixed atlas IDs and
explicit OAS regularization. See the [source, split and utility protocol](fmri_pilot_protocol.md).
Its local 601-subject dataset is a research pilot. The
[held-out-site utility study](fmri_pilot_results.md) found no established
augmentation gain with the initial single-training-seed models. The completed
[five-training-seed replication](fmri_replication_results.md) retains that
negative finding: real-only AUC .7654, selected DiffeoCFM .7452, node-aware
CorrFlow .7365 and anonymous CorrFlow .7314. It uses the same inspected site.
Source-specific redistribution review remains pending.

A later [source-atlas audit](abide_atlas_audit_20261006.md) identifies an
unnamed two-voxel label responsible for 161 of the 179 exclusions. The
[separately declared 110-region controls](abide_named110_results_20261006.md)
reconstruct both the same 601 subjects and an expanded 762-subject cohort.
Neither replay nor Gaussian augmentation establishes a gain. Original results
remain preserved, and learned 110-region generators have not been fitted.

```bash
python research/fetch_abide_roi.py --metadata /path/to/pinned_phenotypic.csv \
  --output artifacts/abide_roi_series
corrbench-build-abide --metadata /path/to/pinned_phenotypic.csv \
  --series artifacts/abide_roi_series --output artifacts/abide_ho
```

## Spatial weather pilot

The [weather protocol](weather_pilot_protocol.md) builds 564 matrices from
44 years of NOAA NCEP–NCAR daily 2 m temperatures. It uses training-fitted seasonal
adjustment, explicit grid coordinates, non-overlapping quarterly windows and
chronological year groups. A South America patch tests spatial transfer.
The [baseline study](weather_pilot_results.md) evaluates spatial aggregate
variances. A separate [exploratory learned comparison](weather_generation_results.md)
finds that DiffeoCFM improves on its Gaussian prior but does not beat replay or
conditional means on this utility measure. It uses one training seed on inspected
periods; no forecasting gain is claimed.

~~~bash
python -m pip install -e '.[weather]'
python research/fetch_ncep_temperature.py --help
corrbench-build-weather --source artifacts/verified_noaa_source \
  --output artifacts/weather_ncep
~~~

An isolated CPU environment reconstructed every brain and weather dataset file
exactly, including the weather climatology and manifests. These are environment
reproductions from the same downloaded sources, not external-person replications.

## Reference exports, leaderboard and competition

The separate corrbench-export-domain command preserves the brain and weather
schemas. All 1,165 rows and every source array/dtype pass exact Parquet round-trip
checks, and the Hugging Face client loads all configurations and splits offline.
See [domain formats and usage](domain_reference_format.md).

The [local leaderboard](../leaderboard/README.md) provides separate protocols,
sortable results, evidence links and downloadable records. It is prepared for a
static Hugging Face Space and has passed browser checks at desktop and mobile
widths. A private Space candidate is staged and its files hash-verified; public launch
and hosted browser delivery remain separate.

The [competition prototype](../competition/README.md) includes a standalone candidate
distribution metric, replay starter and generated notebooks. An entirely
synthetic 80-dimensional fixture passes CSV, public/private subset, invariance
and malformed-submission checks. A protected real holdout, final metric and
unpublished hosted Kaggle test remain requirements before launch.

The [earlier development audit](release_validation_20261006.json) passed 104
tests for the pinned `0.1.0.dev0` candidate. The coordinated `0.1.0rc1` candidate
includes later DDPM tooling and named-region controls; its distribution audit
also installs the source archives before testing them and checks every console
entry point. These release checks remain separate from remote CI and GPU training
reproduction. See [candidate notes](release_candidate_rc1.md).
