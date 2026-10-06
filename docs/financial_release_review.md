# Financial reference release: source review

Updated 2026-10-06. Taiwan remains pending; Japan is now explicitly restricted
under the retrieved competition rules. Both exports remain local. This note records source evidence and the
remaining scope checks; it is not a grant of redistribution permission.

## Taiwan

The curated-panel lineage identifies official TWSE MI_INDEX and TPEx
afterTrading/dailyQuotes data, acquired through July 30, 2026. The matrix source
hash is d13a5fa92abe105d069eda1f6669111e84f13b40d637d4b78378a8dacaa43cfd.
The exact [return convention](taiwan_return_convention.md) is now audited.

There is a concrete open-data route to investigate. The government's
[listed-stock daily transaction dataset](https://data.gov.tw/dataset/11549)
and [OTC stock quote dataset](https://data.gov.tw/en/datasets/11370) identify the
Open Government Data License 1.0. The former lists date, security code/name,
volume/value, OHLC, price change and trade count. Both point to exchange data
services. Their listed license permits reuse and derivatives under its
attribution requirements. See the [license text](https://data.gov.tw/license).

TWSE's [website terms](https://wwwc.twse.com.tw/en/terms/use.html) separately
describe restrictions and an exception for materials authorized through the
Government Open Data Platform. The current website OpenAPI MI_INDEX listing
concerns market/index statistics, whereas the historical acquisition extracted
individual-stock tables from the website's MI_INDEX report. A shared endpoint
name does not by itself establish that the same historical fields, dates and
distribution are covered by a particular open-data listing.

Next establish the exact correspondence between the preserved historical
TWSE/TPEx records and the licensed datasets, including pre-listing history and
the attribution required for both providers. Record applicable versions and
whether the intended release contains raw reports, curated returns or derived
matrices. Where the documented license applies, a redundant permission request
is unnecessary. Where scope remains unresolved, obtain a specific clarification
or release the builder for separately obtained input instead of assuming a
blanket exchange-wide license. No rights-holder contact has been sent.

## Japan

The preserved source is the
[JPX Tokyo Stock Exchange Prediction competition archive](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data),
SHA-256 aa144566f63e62a5af9ac5a7851e42ded46554fbfe650f851223b01920210c85.
The builder combines the declared training and supplemental primary/secondary
price files and excludes overlapping example-test files. Returns are
split-adjusted close-to-close, excluding cash dividends.

The October 5 text-only check was inconclusive. A normal logged-out Chromium
visit on October 6 retrieves the official rules: section 7.A limits data use
to competition participation and Kaggle forums; section 7.B prohibits making
the competition data available to nonparticipants. An open-source winner-code
license does not grant a general data license. See the [dated rules review](jpx_rules_review_20261006.md)
for retrieval evidence and the distinction between explicit source restrictions
and the unresolved scope of derived matrices and aggregate publication.

Separate permission for this off-platform research/publication has not been
established. Preserve the results for author review, hold the Japan data release
and further Japan experiments, and resolve permission before public submission.
An author question is pending; the [specific permission draft](jpx_permission_question_draft.md)
has not been sent. Do not delete or silently relabel the negative Japan result.

The [Taiwan field/date crosswalk](taiwan_source_scope_20261006.md) now identifies
the exact feeds and historical panel coverage. The panel hash is reverified;
the two portal publication dates are not treated as legal coverage cutoffs.

## Release preparation

The package can document sources, builders, protocols and result audits without
asserting an open license for these dataset files. A cleared dataset export
needs its source-specific evidence and attribution recorded in the release
configuration. The current paper-reproduction periods are already inspected
and publicly reconstructible; source rights clearance would not turn them
into protected competition targets.
