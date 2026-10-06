# Spatial temperature pilot — 2026-10-05

Protocol declared before downloading temperature arrays or evaluating matrices.
This is an initial CorrBench adapter and utility study, not an operational
weather forecast, energy-production model, or extreme-event risk model.

## Source and permission

Use NOAA PSL's NCEP/NCAR Reanalysis 1 daily mean 2 m temperature on the
94-by-192 Gaussian grid, 1981 through 2024 inclusive. Retrieve immutable local
copies of the annual NetCDF files from
https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface_gauss/
and record URLs, byte sizes, SHA-256 hashes, time coordinates and variable units.
This starts from a documented reanalysis product, not independent raw stations.
The download hostname returned HTTP 403 from this workspace; the same official
public dataset is accessible through NOAA PSL's THREDDS fileServer endpoint.
Use that documented public service and retain its exact URL in each record.

The [official dataset page](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)
states no usage restrictions. The [PSL policy](https://psl.noaa.gov/disclaimer/)
allows use of unannotated government material and requires accurate attribution;
do not imply NOAA endorsement or describe our derivatives as official products.
Cite Kalnay et al., *The NCEP/NCAR 40-year reanalysis project*, Bulletin of the
American Meteorological Society 77, 437–470 (1996), and acknowledge NOAA PSL,
Boulder, Colorado, as the provider. Preserve any more specific file notices.
Public export remains coordinated with the rest of CorrBench.

The source ended production in March 2026 according to the
[current project notice](https://psl.noaa.gov/data/reanalysis/).
Our fixed 1981–2024 interval does not depend on continuing updates.

## Sampling units and splits

Use four fixed 8-by-8 patches centered on the nearest grid point to:
western/central Europe (45 N, 0 E), central North America (40 N, 105 W),
East Asia (35 N, 120 E), and subtropical South America (25 S, 60 W).
Take index offsets -3 through +4 on each grid axis; wrap longitude. Retain the
actual latitude, longitude, global grid-cell IDs and local row/column ordering.
No land mask or data-dependent location selection is used; patches may include
water. The target is regional near-surface temperature, not land-only exposure.

Each matrix uses one region and one non-overlapping calendar quarter of a year
(January–March, April–June, July–September, October–December). Preserve all daily
observations, including leap days; record the exact count and date interval.
Different regions within a year are dependent views of the same global climate.
Use years, not matrices or individual days, as the uncertainty/resampling unit.

For the first three regions, train on 1981–2010, validate on 2011–2015, and test
on 2016–2024. No date occurs in two chronological splits. South America is an
additional spatial-transfer target in 2016–2024; it never contributes to fitting
matrix generators or spatial-kernel parameters. Its 1981–2010 local climatology
is allowed as an explicitly documented preprocessing calibration. This is
transfer to a calibrated new region, not a claim of zero observations there.
Its other years remain unused in the initial model comparison.

## Observation process and matrix targets

At each grid cell, fit a deterministic annual-cycle model using only 1981–2010:
intercept plus sine/cosine harmonics 1 through 4 of fractional calendar year.
No trend term and no later-year refitting. Subtract the fitted seasonal mean.
Within each quarter, center and standardize each cell's anomalies to form the
empirical Pearson target, retaining its rank and eigenvalue diagnostics.

The primary PD target is per-quarter OAS shrinkage of these standardized
anomalies followed by diagonal normalization. Save empirical and regularized
correlations, shrinkage, sample standard deviations, observation count, region,
quarter, year and node coordinates. OAS's independent-observation assumptions
are imperfect for serially correlated weather; this is a declared regularized
estimator target, not a claim of optimal shrinkage or latent physical truth.
Reject missing/nonfinite data, duplicate days, incomplete quarters or constant
cells, with explicit reasons; never fill them using another split.

## First baselines and downstream proxy

Evaluate region/quarter-conditioned training replay, a deterministic training
mean matrix, identity, and a spatial exponential kernel with nugget:
R_ij = (1 - eta) exp(-chord_distance_ij / length), i != j; R_ii = 1.
Use Earth-centered unit vectors times radius 6371 km to define chord distance;
the Euclidean-distance exponential kernel is PD for distinct locations.
Fit length from {100, 250, 500, 1000, 2000, 4000} km and eta from
{.01, .05, .1, .2, .4}, minimizing training mean squared off-diagonal error.
Fit per calendar quarter, pooling the three training regions, so the same
kernel can be evaluated on South America's unseen coordinates.
The training mean and replay require a known region and do not receive a
fictional unseen-region category. For transfer, additionally report a
quarter-wise pooled local-grid mean, labeled explicitly as a weak transfer
baseline whose local axes are aligned but absolute coordinates differ.

The first utility proxy is the distribution of equal-area spatial-average
variance of unit-variance local anomalies, w' R w with w proportional to cos(lat),
and four quadrant contrasts/averages. Compare predicted distributions with
empirical held-out quarter estimates, reporting normalized errors and
year-block bootstrap intervals. Keep regularized-matrix fidelity and empirical
aggregate variance separate: an estimator's shrinkage changes the latter.
This isolates dependence under unit marginal variance; it does not forecast
temperature magnitudes, attach realistic marginal tails, or model trajectories.

Use 1,000 reproducible replay draws per region/quarter, seed 2027. Report all
baselines, including cases where independence or a spatial kernel performs
poorly. No learned-generator superiority is claimed until a separate frozen
training/selection protocol and comparison have actually been executed.

## Acceptance and limitations

Require hashed source retrieval, exact date-disjoint chronological construction,
train-only climatology verification, retained node identities, PD/rank audits,
and measured aggregate-variance baselines. A pilot with a few dozen independent
test year/region units is not a large independent sample merely because it
contains many daily values or matrix entries. Reanalysis assimilation, regional
dependence, seasonal nonstationarity and serial correlation remain limitations.
Future extensions should compare stronger spatial/factor and direct scenario
models before claiming scientific or engineering decision benefit.
