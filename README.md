# CorrBench

Build, validate and compare generated correlation matrices with explicit data
provenance and shared evaluation protocols. Research software · **0.1.0rc2**.

| Repository | Responsibility |
| --- | --- |
| **CorrBench** | Observation-to-matrix builders, geometry, validation, metrics and reference formats |
| [CorrFlow](https://github.com/gautiermarti/CorrFlow) | Geometric flow matching, matched diffusion controls and runnable lessons |
| [CorrGAN](https://github.com/gautiermarti/CorrGAN) | Unconditional/conditional GAN implementations, legacy importers and repair-aware evaluation |

Geometry, distributional realism and downstream usefulness are different claims.
The tools keep raw validity, repair, coverage, replay baselines and negative
outcomes visible. Finance, brain and weather each have their own scientific protocol.

## Start on CPU

Use Python 3.12 or 3.13. These commands require no account, GPU or real dataset
once the coordinated repositories are public:

```bash
git clone https://github.com/gautiermarti/CorrBench.git corrbench
cd corrbench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
python examples/geometry.py
```

Expected: `GFT round trip valid`, with reconstruction error below `1e-7`.
On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. For GPU
training, use the PyTorch installation appropriate to your CUDA environment.
The source checkout contains configurations and examples; the wheel provides
importable APIs and console commands. No PyPI publication is currently claimed.

## Build your first dataset

This example creates entirely synthetic returns and 36 matrix records:

```bash
python examples/make_synthetic_panel.py --output data/synthetic
corrbench-build --config data/synthetic/config.yaml --output data/synthetic_matrices
corrbench-order --input data/synthetic_matrices --output data/synthetic_ordered --workers 1
corrbench-cache-gft --input data/synthetic_matrices/train.npz \
  --output data/synthetic_matrices/gft/train.npz --device cpu
```

The builder writes `train.npz`, `validation.npz`, `test.npz` and `manifest.json`.
It records dates, stock IDs, source hashes and training-derived thresholds. The
ordered copy retains row correspondence. Use a **new output path** for each run;
these commands refuse to replace existing artifacts. Synthetic examples exercise
the code and do not reproduce a financial result.

## Find the right workflow

| Task | Guide |
| --- | --- |
| Real-data construction and evaluation | [Scientific workflows and evidence](docs/scientific_workflows.md) |
| Financial return convention and permissions | [Return audit](docs/taiwan_return_convention.md), [source review](docs/financial_release_review.md) |
| Brain and weather matrix contracts | [Domain reference formats](docs/domain_reference_format.md) |
| Exact financial Parquet staging | [Reference format](docs/reference_format.md) |
| Model comparisons and limitations | [Matched financial study](docs/matched_taiwan_results.md), [brain](docs/fmri_replication_results.md), [weather](docs/weather_generation_results.md) |
| Research leaderboard | [Protocols, records and admission](leaderboard/README.md) |
| Competition tooling | [Synthetic scorer and starter](competition/README.md) |
| Contributing and troubleshooting | [Contribution guide](CONTRIBUTING.md) |

Every console command accepts `--help`. Install `.[weather]` for NetCDF,
`.[atlas]` for atlas inspection, `.[plots]` for figures, `.[notebooks]` for
notebook execution, or `.[dev,weather,atlas]` for the test suite.

## Data, versions and citation

This repository distributes **MIT-licensed software**, not financial prices,
participant records or trained research models. Japan's competition source is
restricted; Taiwan historical-feed coverage and numerical atlas-derivative terms
remain under review. NOAA has a separate source card and attribution. Packaging
and hash checks do not establish source-data permissions.

Use the exact compatible CorrBench revision pinned by each model repository.
[Release notes](CHANGELOG.md) describe changes; [CITATION.cff](CITATION.cff) gives
software citation metadata. Use a paper identifier only once it is actually
published. See [security and checkpoint handling](SECURITY.md) and the
[release contract](docs/release_contract.md).
