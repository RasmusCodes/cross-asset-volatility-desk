# 01 · Data: sources, coverage, cleaning and quality

Everything downstream rests on one pull from WRDS, made once (scripts/pull_data.py) and recorded in
`data/raw/manifest.json` (SQL, row count, columns, byte size and sha256 of every file). The raw data are licensed
and are not in this repository; the code reproduces them for anyone with the same subscriptions.

## What was pulled

| Stage | Files | Rows | Size (parquet) | Content |
|---|---:|---:|---:|---|
| reference | 19 | 2,160,824 | 18.8 MB | zero curve, index dividend yields, distributions, CRSP S&P 500 membership and monthly prices, CRSP-OptionMetrics link, VIX/VXO/VXN/VXD, Cboe rate curve and calendar, SPX contract list, Datastream VIX futures, Fed rates |
| panel | 180 | 18,723,037 | 532.1 MB | underlying prices, forwards, standardized ATM options and volatility surfaces for 28 index/ETF ids and 166 single stocks |
| cboe | 30 | 18,989,002 | 287.5 MB | Cboe SPX prices and IVs 2025-07-01 .. 2026-11-03, and 13 sample dates 1998-2025 for the cross-vendor check |
| spx | 30 | 48,335,273 | 1,412.0 MB | every OptionMetrics SPX option row, 1996-01-04 .. 2025-08-29 |
| etf | 60 | 11,316,074 | 338.3 MB | SPY/QQQ/IWM: full chains at month-ends, plus daily near-the-money standard monthlies |
| **total** | **319** | **99,524,210** | **2,588.7 MB** | |

Sources (all through WRDS under the account holder's subscription): OptionMetrics IvyDB US (option prices, implied
vols and Greeks, zero curve, dividends, forwards, standardized surfaces), Cboe option end-of-day data, Thomson
Reuters Datastream futures (VIX futures settlements), CRSP (S&P 500 membership, market caps) and the WRDS
CRSP-OptionMetrics link. Coverage: OptionMetrics ends on 2025-08-29; Cboe extends SPX to 2026-11-03; the Cboe VIX
index file ends 2026-10-30; VIX futures settlements run to 2026-11-19.

## Cleaning (voldesk.data.chains)

* **Time.** Quotes are end-of-day; valuation at 16:00 ET. AM-settled contracts settle at 09:30 on the settlement date
  (OptionMetrics records pre-2015 standard expiries on the Saturday; settlement is the Friday), PM-settled at 16:00.
  T = ACT/365 between those instants.
* **Discounting.** OptionMetrics' zero curve (continuously compounded, linear in r·T between nodes); Cboe's rate curve
  after 2025-08-29.
* **Forwards from put-call parity**, never from an assumed dividend yield: for each date and expiry, pairs of calls and
  puts with two-sided quotes near the money give F = K + (C − P)/D, combined with inverse-spread weights after
  trimming outliers. 97.4-100% of expiries per year had enough pairs (the rest fall back to OptionMetrics' dividend
  yield and are flagged). Our forwards differ from OptionMetrics' own by a median of −26 to +9 basis points by year
  (row-weighted mean of monthly medians; largest in 2008 and 2002).
* **Implied vols are recomputed** (Black-76 with our forward and discount factor) for bid, mid and ask. Against
  OptionMetrics' own vols on out-of-the-money two-sided quotes the median gap is between −0.18 and +0.02 vol points
  in every year, mean absolute gap 0.12-0.76 vol points (Cboe's vols in 2025-26: −0.02 median, 0.09-0.10 mean
  absolute).
* **Flags** (kept as columns, never silently dropped; ranges over the OptionMetrics years): crossed quotes
  (0.000-0.002% of rows), no bid (2.4-10.5%), spread wider than the mid (3.3-14.6%), mid below intrinsic value
  (2.0-9.5%, mostly deep in the money), vendor IV missing (5.2-12.5%), our IV not solvable (2.4-10.3%). Table:
  `results/data_quality/summary_yearly.csv`.

## Cross-vendor check: OptionMetrics vs Cboe on the same SPX contracts

`results/data_quality/cross_vendor.csv` matches contracts by expiry, type, strike and settlement style.

| Date | Matched two-sided | Same bid | Mid difference, median | IV difference, median (vol pts) |
|---|---:|---:|---:|---:|
| 1998-06-17 | 619 | 100% | 0.00% | +0.38 |
| 2005-06-15 | 784 | 100% | 0.00% | +0.16 |
| 2008-10-15 | 2,140 | 7% | 2.50% | +0.59 |
| 2015-08-26 | 8,023 | 20% | 0.50% | −0.01 |
| 2020-03-18 | 18,470 | 3% | 0.94% | +3.08 |
| 2024-08-07 | 25,908 | 12% | 0.30% | +0.02 |
| 2025-07-15 (overlap) | 22,745 | 30% | 0.05% | +0.13 |
| 2025-08-15 (overlap) | 23,118 | 41% | 0.12% | −0.01 |

Until the mid-2000s the two vendors' quotes are identical (apparently the same source); their implied vols still
differ because their rates, dividends and models differ. From 2008 the quotes themselves often differ, by little on
calm days and by a lot on stressed ones (3.1 vol points at the median on 18 March 2020), which is what different
end-of-day snapshot times would produce on fast markets. This is a measured difference, not a correction: the
2025-08-29 → 2025-09-02 splice from OptionMetrics to Cboe in the SPX history therefore carries a small vendor effect,
which the two overlap dates bound at about 0.1 vol point.

## Survivorship and the single-stock universe

The single-stock panel is the 50 largest S&P 500 members by market cap at each month end (CRSP msp500list +
msf), mapped to OptionMetrics ids through the WRDS link valid on that date: 356 months, 165 companies, 166
OptionMetrics ids, 100% linked. Members that later left the index or were acquired are included in the months they
were members. CRSP's legacy files end 2024-12-31, so January-August 2025 reuse the December 2024 ranking (stated
wherever those months are used).

## What is not here, and why

* No raw data in the repository or on the website (vendor licences). The demo ships fitted parameters, aggregated
  statistics and backtest results only; `tests/test_web_export.py` checks that no quote arrays leak into the bundle.
* No single-stock option chains (only standardized surfaces), so dispersion trades are variance-based (report 05).
* SPY/QQQ/IWM chains were pulled at month-ends (full) and daily for near-the-money standard monthlies; they are
  American options, used here only for surface comparisons (report 02).
* The free Cboe delayed-quote API used by an earlier project is blocked for scripts since 2026-11-20; it was not used.
* Synthetic data exist only in `voldesk.data.synthetic` for tests and CI, labelled `SYNTHETIC` in every frame; no
  research script reads them.
