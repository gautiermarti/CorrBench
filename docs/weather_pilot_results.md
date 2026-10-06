# Weather pilot results — 2026-10-05

The local adapter and baseline aggregate-variance study are complete. No learned
weather-generator result or operational forecast benefit is claimed.

The primary target is per-quarter OAS correlation of seasonally adjusted daily
2 m temperatures on 64 fixed grid cells. The utility proxy compares distributions
of regional and quadrant average variances against empirical quarter estimates,
under unit local variance. Smaller normalized 1-Wasserstein is better.

| Baseline | 2016–2024, three known regions | South America transfer |
| --- | --- | --- |
| identity | 0.9076 | 1.0397 |
| spatial kernel | 0.1433 | 0.2550 |
| pooled grid mean | 0.1030 | 0.1831 |
| conditional mean | 0.0824 | Not applicable |
| conditional replay | 0.0477 | Not applicable |

Historical replay is a strong reference. Independence badly misrepresents spatial
aggregation. The simple exponential kernel does not beat the empirical means on
this proxy; distance alone omits regional anisotropy and geography. A learned
generator must add value beyond these baselines before claiming utility.

The baseline settings were fixed before matrices were built. Kernel length/nugget
were fitted on training years only. All validation, test and transfer outcomes are
retained. There are only nine test year groups; patches/quarters within a year are
dependent. The report includes joint year-bootstrap percentiles. They are
descriptive resampling ranges, not calibrated confidence limits: Wasserstein
resampling is visibly upward biased at this sample size. In particular, the replay
point estimate can lie below its resampled percentile range.

| Partition | Matrices | Year groups | Smallest regularized eigenvalue |
| --- | --- | --- | --- |
| train | 360 | 30 | 0.029449 |
| validation | 60 | 5 | 0.029443 |
| test | 108 | 9 | 0.030017 |
| spatial_transfer | 36 | 9 | 0.028942 |

All 44 annual source files are hashed and readable. Initial truncated HTTP
downloads were excluded; verified byte ranges/content lengths reconstruct the
complete official NetCDF files. Data sources: [NOAA PSL](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html),
with [usage and attribution policy](https://psl.noaa.gov/disclaimer/).
Cite Kalnay et al., BAMS 77, 437–470 (1996). CorrBench derivatives do not imply
NOAA endorsement. Reanalysis observations are not independent station records.

Splits contain no shared dates in the chronological task. The transfer task
allows a local 1981–2010 South America climatology calibration but no regional
matrix-generator or kernel fitting. It shares 2016–2024 with the primary test and
must not be counted as an additional independent temporal sample.

See [protocol](weather_pilot_protocol.md), source/estimator tests, and the
machine-readable companion [results](weather_pilot_results.json).
