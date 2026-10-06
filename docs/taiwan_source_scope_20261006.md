# Taiwan historical feed and open-data scope

Checked 2026-10-06. Status remains **pending**. The [machine-readable crosswalk](taiwan_source_scope_20261006.json)
records hashes and cached report headers. No new market data were acquired.

The current curated panel rehashes to
`d13a5fa92abe105d069eda1f6669111e84f13b40d637d4b78378a8dacaa43cfd`,
the exact source of the frozen matrix experiments. Its upstream acquisition
script uses these historical website feeds:

| Provider | Actual report and request | Preserved coverage | Government listing |
| --- | --- | --- | --- |
| TWSE | `exchangeReport/MI_INDEX`, GET with `response=json`, `date=YYYYMMDD`, `type=ALLBUT0999`; individual-stock table | 3,081,617 rows, 3,302 sessions, 2013-01-02–2026-07-30 | [Listed-stock daily transactions](https://data.gov.tw/en/datasets/11549), published 2017-05-23 |
| TPEx | `www/en-us/afterTrading/dailyQuotes`, POST with `date=YYYY/MM/DD`, `response=json` | 2,452,704 rows, 3,302 sessions, same date range | [OTC stock quotes](https://data.gov.tw/en/datasets/11370), published 2015-12-03 |

Both official listings identify Open Government Data License 1.0. They point
to different delivery routes: TWSE `STOCK_DAY_ALL?response=open_data` and TPEx
`DAILY_CLOSE_quotes/stk_quote_result.php?l=zh-tw&o=data`. This is evidence of
relevant open datasets, but does not yet establish coverage of every historical
report used here. The portal publication dates are metadata, **not inferred
legal coverage start dates**. There are 911,394 TWSE rows and 463,084 TPEx rows
before those respective dates; these counts identify the scope question and
do not classify those records as unauthorized.

## Field correspondence

| Curated field | TWSE stock table | TPEx stock table | Matrix use |
| --- | --- | --- | --- |
| `date`, `market` | Requested date and feed identity | Requested date and feed identity | Window membership/provenance |
| `stock_id` | 證券代號 | Code | Stock selection and stable node identity |
| `close` | 收盤價 | Close | Return calculation |
| `spread` | 漲跌價差, signed with 漲跌(+/-) | Change | Exchange-reference return calculation |
| `value_traded` | 成交金額 | Trans. Amount (NTD) | Liquidity selection |
| Name, OHLC, volume, trade count | Corresponding named fields | Corresponding named fields | Retained in upstream panel; not extra matrix features |

The parser reads stock quotes inside MI_INDEX, not index time series. Later
reports also contain bid/ask and other columns; those unused columns are not
inputs to the frozen correlation builder. First trading-day (2013-01-02) and
last-day report headers/hashes are recorded. The January 1 cache examples are
retained in the JSON as observed; TWSE has no stock table on that holiday.
The [return arithmetic audit](taiwan_return_convention.md) remains authoritative:
exchange-reference returns with a raw-close fallback, not independently audited
cash-dividend total returns.

[TWSE website terms](https://wwwc.twse.com.tw/en/terms/use.html), retrieved on
October 6, restrict automated acquisition unless agreed and distinguish
materials authorized through the Government Open Data Platform. The saved HTML
hash is `237bd719cd645462ea963b63f3f3a84088e6d87ee1376af23cbb2db2fa657e0c`.
No further bulk acquisition is planned under an assumed grant. A guessed TPEx
terms URL redirected to an error page; its HTTP 200 response is not terms evidence.

## Remaining concrete question

Confirm that the two listed open datasets cover these stock-quote fields and
historical dates when obtained via the recorded website feeds, and identify the
provider/version attribution for a release of derived return correlations with
stock IDs and window dates. This does not ask to redistribute the entire raw
quote archive. If documented scope already covers this use, record it and apply
its conditions. Otherwise obtain that specific clarification before declaring
the reference data cleared. No provider contact has been sent.
