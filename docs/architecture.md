# Architecture

```
 WRDS (OptionMetrics, Cboe, Datastream, CRSP)
        │  scripts/pull_data.py  ── voldesk.data.wrds (COPY → CSV → parquet, manifest.json) + voldesk.data.catalog (SQL)
        ▼
 data/raw/  (vendor tables as received, sha256 in the manifest)
        │  scripts/build_store.py ── voldesk.data.chains (calendar, curves, parity forwards, implied vols, flags)
        ▼
 data/store/  (one cleaned row per contract and day; forwards per expiry)
        │
        ├── voldesk.surface   slices → SVI / SSVI / eSSVI / SABR / Heston fits → arbitrage, local vol, rough checks
        │      scripts/surface_panel.py, surface_sensitivity.py, rough_study.py
        ├── voldesk.backtest  daily engine over the store + voldesk.strategies (costs, margin, attribution)
        │      scripts/run_backtests.py, summarize_backtests.py
        ├── voldesk.forecast  walk-forward models over daily panels (SPX bars, VIX, surfaces, futures, members)
        │      scripts/forecast_study.py, forecast_economics.py
        └── voldesk.dispersion  index vs top-50 variance and correlation trades
               scripts/dispersion_study.py
        ▼
 results/ (small derived tables) ── scripts/make_figures.py ──► reports/figures
        │
        └── scripts/export_demo.py ── voldesk.web.export ──► demo/data/*.json ──► demo/app (browser)
```

## Design decisions

* **Raw is immutable, everything else is rebuilt.** Pulls land in `data/raw` exactly as received, each file
  recorded with its SQL and checksum; reruns skip files whose SQL is unchanged. Cleaning never overwrites raw data
  and keeps quality problems as flags rather than deleting rows.
* **Our own forwards and implied vols.** Vendor vols depend on vendor dividend and rate assumptions; recomputing them
  from parity forwards makes 1996-2025 OptionMetrics and 2025-26 Cboe data consistent, and the vendor values stay in
  the store for comparison.
* **One harness for all models.** `voldesk.surface.compare` fits every model to the same slices and computes the same
  fit and arbitrage statistics on the same grid, so differences come from the models, not the scoring.
* **One pass, all strategies.** The backtest engine loads one year of chains and steps every strategy through each
  day, recording cost components so one run serves every cost model. Attribution terms always add up to the mid
  change (the residual is defined as the rest), and the tests check that identity.
* **Walk-forward only.** Every forecast is refitted on expanding windows using only rows whose targets were
  observed at the refit date (`voldesk.forecast.evaluate.walk_forward`, tested for look-ahead).
* **The demo is derived-only.** The browser receives fitted parameters, aggregates and backtest curves, and does all
  pricing itself (Black-76, the five models including Heston by the COS method); its self-test compares its numbers
  with Python-generated test vectors.

## Numerics worth knowing

* Black-76 prices use Jäckel's normalized form with the erfcx tail (accurate far out of the money); implied vols use a
  bracketed Halley iteration on the out-of-the-money price, so in-the-money inputs are inverted through their
  better-conditioned counterpart.
* Heston: COS pricing (1,024 terms) during calibration, adaptive quadrature of the Lewis integral for reported
  errors; the two agree to 1e-7 in price in the tests.
* SVI: Zeliade's quasi-explicit start, then SLSQP with box bounds and butterfly/Lee constraints on a grid reaching far
  beyond the quotes; calendar order is imposed only where both slices are quoted and only if it is cheap.
* Rough Bergomi: exact joint Gaussian simulation of the Volterra process and the price driver by Cholesky
  factorization of their covariance (fine for a few hundred steps).
