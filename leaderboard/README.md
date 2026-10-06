---
title: CorrBench
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
license: mit
short_description: Correlation generation, measured across distinct scientific tasks
---

# CorrBench research leaderboard preview

Private development artifact. No Space has been uploaded or published.
The MIT license covers the application code; it does not license source datasets.

Build records from audited reports with:

~~~bash
python research/build_leaderboard.py
corrbench-validate-leaderboard --directory leaderboard
python -m http.server 8765 --bind 127.0.0.1 --directory leaderboard
~~~

Supply the completed matched-suite audit with --matched-audit when it exists.
The app never executes participant code. It loads curated local JSON, keeps
protocols separate, shows uncertainty/limits, and links its evidence.
No row is described as independently verified merely because a JSON flag says so.

Before a public export, review every result, replace release-pending copy with
actual coordinated repository/dataset versions, and preserve source-specific
data rights. Hosting preparation follows the official
[static Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-static).

The builder uses committed compact audit reports. Integrity validation rejects
incompatible dataset/split/metric/protocol fingerprints, changed evidence,
nonfinite scores and unsupported verification claims. These checks do not
substitute for scientific review or independent reproduction.
