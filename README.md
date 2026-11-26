# Cross-Asset Volatility Desk

A research and trading platform for equity-index volatility: thirty years of real option data, five surface models
compared on 371 dates, seventeen option strategies backtested day by day with Greek attribution, costs and margin,
walk-forward forecasts with economic tests, dispersion and correlation trades, and an interactive terminal.

**Interactive demo:** [rasmuscodes.github.io/projects/volatility-desk/](https://rasmuscodes.github.io/projects/volatility-desk/):
3D implied-vol surfaces for SPX, SPY, QQQ and IWM on crisis and calm dates, five models side by side, a strategy
builder with Greeks and payoffs, spot/vol/time scenarios with P&L attribution, historical stress scenarios and the
backtest results. Everything in it is computed in the browser from fitted parameters.

## What it found

* **Surfaces** ([report 02](reports/02_surface_models.md)). On 371 SPX month-ends (1996-2026) SVI fits closest
  (median 0.24 vol points, 89% of quotes inside the bid-ask) and interpolates held-out expiries best, but carries
  calendar arbitrage in its extrapolated wings (8% of grid points). eSSVI and SSVI are arbitrage-free on every date
  at two to three times the error; SABR (β = 1) is the practical compromise and extrapolates wings best; Heston fits
  worst (0.97) and its short-dated skew is too flat. The ATM-skew power law of rough volatility is in the data
  (slope −0.33, H ≈ 0.17), but fitted Heston surfaces reproduce the same slope over one week to six months.
* **Strategies** ([report 03](reports/03_strategies.md)). Selling index volatility earned a premium before costs
  (short ATM straddle +5.3% of notional a year, Sharpe 0.44) paid for with crash losses (worst day −10.5%). Costs decide
  most of the rest: base costs halve the delta-hedged straddle's return and stressed costs erase it; calendars,
  diagonals, butterflies and risk reversals are near zero before costs and negative after. The delta-hedged short
  25-delta put and call had the best risk-adjusted results (Sharpe 0.59 and 0.69 at base costs).
* **Forecasting** ([report 04](reports/04_forecasting.md)). Log-HAR with the VIX beats EWMA by 7.5% and 26% in QLIKE
  at 1 and 5 days (out of sample 2000-2026); simple AR/HAR models beat a random walk on the VIX through mean
  reversion, and VIX futures do not; implied plus realized correlation forecasts next month's correlation best
  (R² 36.5%). **No forecast-based timing of the variance premium beat a constant allocation after costs**;
  volatility targeting improved the index's drawdowns and Sharpe ratio, with plain EWMA doing best.
* **Dispersion** ([report 05](reports/05_dispersion.md)). Implied correlation exceeded the next month's realized by
  +0.04 on average (t = 5.1). Whether variance dispersion survives costs depends on strikes this study cannot observe
  for single stocks; the correlation-swap trade survives assumed costs (t = 2.7) with severe crash losses.
* **Data** ([report 01](reports/01_data.md)). 99.5 million rows from WRDS (OptionMetrics, Cboe, Datastream, CRSP),
  cleaned with our own parity forwards and implied vols, checked against both vendors.

Backtests are historical simulations with estimated costs; nothing here is evidence of a tradable edge, and nothing
here is investment advice.

## Layout

```
src/voldesk/
  pricing/      Black-Scholes / Black-76, Greeks to third order, implied vol, dividends, American lattices
  data/         WRDS pulls with a manifest (wrds, catalog), cleaning (chains, calendar, rates), store readers,
                point-in-time universe, SYNTHETIC fallback for tests
  surface/      SVI / SSVI / eSSVI, SABR, Heston (COS + quadrature), local vol, Andreasen-Huge, rough Bergomi,
                arbitrage diagnostics, the model-comparison harness (compare)
  strategies/   positions, Greeks, Taylor attribution, builders, costs, margin (strategy-based + portfolio),
                scenario risk / VaR, hedging simulations
  backtest/     daily mark-to-market engine, strategy library, metrics
  forecast/     realized measures, GARCH/HAR models, level forecasts, correlation, regimes (HMM), evaluation
  dispersion/   variance-based dispersion and correlation trades
  web/          the demo's data bundle (derived numbers only)
scripts/        pulls, store build, studies, figures, demo export (each script's docstring has its command)
reports/        01-05 research reports and figures
results/        small derived tables the reports cite
demo/           the browser terminal (vanilla JS, no build step) and its data bundle
docs/           architecture, data access, limitations
tests/          35 tests on synthetic data (no WRDS needed)
```

## Reproduce

Requires Python 3.11+ and, for the data, WRDS access to OptionMetrics IvyDB US, Cboe option data, Datastream
futures and CRSP.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
pytest                                   # 35 tests, synthetic data only
```

With WRDS access (log in once so the PostgreSQL password file holds the WRDS line; see [docs/data.md](docs/data.md)):

```bash
python scripts/pull_data.py --stage all          # ~100M rows, ~2.6 GB parquet, resumable
python scripts/build_store.py --source om         # clean SPX 1996-2025
python scripts/build_store.py --source cboe       # SPX 2025-07 .. 2026-11
python scripts/surface_panel.py --set monthend    # model comparison (hours; checkpointed)
python scripts/run_backtests.py && python scripts/summarize_backtests.py
python scripts/forecast_study.py --section all && python scripts/forecast_economics.py
python scripts/dispersion_study.py
python scripts/make_figures.py && python scripts/export_demo.py
```

## What is and isn't in this repository

In: all code, tests, reports, figures, derived result tables, and the demo with its bundle of fitted parameters and
aggregate statistics. Not in: any vendor data (licensed: OptionMetrics, Cboe, Datastream, CRSP via WRDS), the cleaned
store or caches. `tests/test_web_export.py` checks that the demo bundle contains no quote arrays. See
[docs/limitations.md](docs/limitations.md) for what the results can and cannot support.

## License

MIT © 2026 RasmusCodes
