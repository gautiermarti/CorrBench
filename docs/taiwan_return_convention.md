# Taiwan return convention: source-column audit

Verified 2026-10-05. The frozen source column is called daily_total_return, but
the implemented convention is more precisely **returns relative to exchange
reference prices, with a raw-close fallback**. Keep the original field name,
panel, matrix files and all model results unchanged for reproduction. Do not
describe the column as an independently reconciled cash-dividend total return.

For closing price P and the exchange-reported signed change Δ, the builder sets
the implied reference price to B = P − Δ and computes Δ/B. It uses that result
when B > 0 and its magnitude is at most .35. Otherwise it tries the raw return
P/P_previous − 1, where P_previous is the previous observed close for that
instrument. A missing candidate or magnitude above .35 produces null.

The [complete arithmetic audit](taiwan_return_convention_audit.json) checks all
5,534,321 rows, the preceding-close links, both candidate-return formulas, branch
flags and the final values. It verifies the source-panel hash against the frozen
matrix dataset manifest. Every value reproduces exactly, including missingness.

| Quantity | Rows |
| --- | ---: |
| Exchange-reference branch | 5,526,284 |
| Finite raw-close fallback | 7,877 |
| Finite returns in total | 5,534,161 |
| Null returns | 160 |
| Flagged unresolved magnitude above 35% | 13 |
| Finite chosen/raw-close differences above one basis point | 47,859 |

These are panel-row counts, not counts of verified corporate actions. The
160 missing returns include the 13 flagged large candidates; they are different
quantities. The correlation builder's window completeness rules still apply.

The [TWSE ex-right/ex-dividend reference formula](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
and [TPEx explanation](https://www.tpex.org.tw/en-us/about/company/faq.html)
describe how distributions and capital events enter exchange reference prices.
That does not make every reference-price return identical to an investor's
cash-distribution return. As a purely algebraic illustration, a previous close
of 100, cash distribution of 10 and new close of 95 gives 95/90 − 1 = 5.56%
relative to the ex-dividend reference, versus (95 + 10)/100 − 1 = 5.00% with
cash included against the previous investment value. This is not an observed
event or an estimate of the benchmark's error.

The audit establishes the arithmetic actually used. It does not independently
reconcile every event, establish terminal delisting proceeds, re-download
upstream prices or quantify how alternative return definitions would change
the benchmark. Such a study needs a new version and explicit comparison, not
an in-place correction of frozen paper data.

Source: official TWSE MI_INDEX and TPEx afterTrading/dailyQuotes reports,
curated by the separately preserved foundation builder. Its hash and the
source-panel hash are recorded in the audit. The [financial source review](financial_release_review.md)
tracks the separate attribution and redistribution questions.
