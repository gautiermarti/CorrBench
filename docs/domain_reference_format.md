# Brain and weather reference exports

The domain exporter complements the dated-market exporter. It never uploads.
Export directories contain a Hugging Face-compatible dataset card, Parquet
splits, source/field metadata and a sanitized provenance manifest.

~~~bash
corrbench-export-domain --data artifacts/abide_ho_20261005 \
  --adapter abide_pcp \
  --metadata configs/releases/abide_pilot.yaml \
  --output artifacts/abide_reference
corrbench-export-domain --data artifacts/weather_ncep_20261005 \
  --adapter ncep_temperature \
  --metadata configs/releases/weather_pilot.yaml \
  --output artifacts/weather_reference
~~~

Each row contains one matrix. Fixed-size multidimensional arrays are flattened
with their original shape and dtype recorded; static atlas IDs are repeated per
row and collapsed back to the original array by the reference reader. All source
fields are explicit. Unknown fields fail rather than disappearing. Validation
checks matrix constraints, identities, dimensions, source hashes and counts
before atomic export.

The brain configuration preserves regularized/empirical matrices, diagnosis,
subject/site identifiers, atlas IDs, observation counts and numerical estimation
diagnostics. The weather configuration preserves both matrix targets, region,
quarter/year, window dates, grid IDs/coordinates, anomaly scales and diagnostics.
The weather spatial_transfer split remains separate from the chronological test.

Use the common reader to restore the original arrays:

~~~python
from pathlib import Path
from corrbench.reference import read_reference_split

arrays = read_reference_split(Path("artifacts/weather_reference"), "train")
print(arrays["matrices"].shape)
~~~

Load Parquet through Hugging Face datasets using the data_files mappings in the
generated card. There is no published Hub identifier yet. The isolated-client
audit uses the actual datasets library with offline loading, checks all splits
and compares first/middle/last records field by field. A separate complete
round-trip checks every array and dtype for all 1,165 domain records.

With --purpose competition_training the exporter writes only train and validation.
This is a staging safeguard, not a protected-holdout design: public upstream data
can still reconstruct a supposedly hidden evaluation set.

## Evidence and source rights

- [Complete array round-trip](domain_reference_audit.json).
- [Independent HF client](domain_hf_client_audit.json).
- [Brain reconstruction](fmri_reconstruction_audit.json).
- [Weather reconstruction](weather_reconstruction_audit.json).

Brain-derivative distribution is blocked pending source-version, atlas and
acknowledgment review. Financial rights remain separately unresolved. NOAA PSL
states no usage restrictions for NCEP–NCAR Reanalysis 1; its attribution and
non-endorsement policy accompanies the weather card. The package MIT license
does not replace any source-data terms. See the source links in each release
configuration. All exports are still local.

The [metadata refresh audit](domain_reference_metadata_audit.json) records updated
cards with byte-identical Parquet data for all 1,165 records. The current brain
card identifies the official ABIDE CC BY-NC-SA 3.0 link while retaining the
derivative redistribution block. Current weather cards use the publisher's
Kalnay et al. page range, 437–472, and DOI. The original NOAA listing and frozen
pilot protocol retain their earlier 437–470 citation; no scientific inputs or
protocol hashes were changed for this bibliographic correction.

NOAA's [current data notice](https://psl.noaa.gov/data/getpage/) says Reanalysis 1
production ended on March 17, 2026. The completed 1981–2024 pilot is unaffected.
Future expansion must account for this endpoint; do not promise ongoing future
holdouts from this product. A different reanalysis needs its own versioned
construction and comparability study.
