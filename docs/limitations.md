# Limitations

What the results can and cannot support, in one place.

## Data

* **Two vendors, one splice.** SPX runs on OptionMetrics to 2025-08-29 and Cboe after. The vendors' quotes differ
  on stressed days (up to 3 vol points at the median on 18 March 2020), most likely because of different end-of-day
  snapshot times; on the two overlap dates in 2025 they agree to about 0.1 vol point.
* **End-of-day quotes only.** No intraday data: no intraday hedging, no execution modelling beyond a share of the
  quoted spread, no special opening quotation (AM-settled positions are closed the day before settlement instead).
* **Single stocks have no option chains here**, only OptionMetrics' standardized surfaces: dispersion is
  variance-based with implied-vol strikes, not traded option portfolios.
* **ETF options are American.** SPY/QQQ/IWM surface fits use European-equivalent vols of out-of-the-money quotes.
* **Index membership after 2024** reuses the December 2024 CRSP ranking (CRSP's legacy files end there).
* **Missing marks.** A held contract with no quote on a day is marked by Black-76 at its previous vol: 5-34 days per
  strategy over 30 years, 225 for the 100-leg variance strip (mostly 1996).

## Models

* Surface comparisons use AM-settled standard monthlies from 7 days to 3 years: nothing is said about weeklies,
  dailies or 0DTE options, which is also why the rough-volatility test cannot separate rough from Markovian models.
* Local-stochastic volatility and rough Bergomi were calibrated on two dates (30 June 2017 and 31 March 2020), not
  on the panel: they show what the models do on a calm and a stressed surface, not how they rank over 30 years.
* The LSV variance process takes the full vol-of-vol of each date's Heston fit. How much of the smile should come
  from stochastic rather than local volatility (the "mixing") is not identified by vanilla quotes; it would take
  forward-start or cliquet prices, which this dataset does not contain. The exotic prices in report 02 are model
  comparisons, not marks.
* Rough Bergomi gets its forward variance curve from the eSSVI surface's variance-swap strip, which leans on eSSVI's
  extrapolated wings, and is scored only on maturities up to about four months.
* Rough Bergomi Monte Carlo converges slowly in the time step at short expiries. The fits use at least 400 steps
  inside every expiry; the call wing of the 17- and 21-day smiles is still 0.2-0.3 vol points from its 1,600-step
  value, so on the calm date the comparison with Heston (0.61 against 0.58) is a tie within numerical error.
* LSV reprices the vanillas close to, not entirely within, Monte Carlo error: small systematic deviations remain at
  the shortest expiries (up to 0.08-0.12 vol points on average), of the same size as pure local vol's on the same
  time grid.

## Backtests

* Historical simulations on one index over one 30-year path with estimated costs. Results describe what the
  structures did; they are not forecasts and not evidence of an exploitable edge.
* Costs are assumptions (a share of the quoted spread, commissions, hedge basis points) and change conclusions: the
  reports show frictionless, base and stressed results side by side.
* Hedges are futures-like, without financing or margin costs; margin is computed at entry under strategy-based
  rules only (not intramonth, not house requirements).
* Taylor attribution has a residual that is large on crash days (third- and higher-order moves); it is reported, not
  hidden.

## Forecasts

* Out-of-sample evaluation over 2000-2026 for most targets; vol of vol from 2008 (VIX options from 2006).
* The correlation-timing rule and its threshold were fixed in advance but tested on a single sample; treat it as a
  hypothesis for new data, not a result.
* Dozens of comparisons were run; individual significant results should be read with that multiplicity in mind.
  The model confidence sets and bootstrap intervals in the reports are per comparison, not adjusted across them.

## Demo

* Shows fitted parameters and statistics, never quotes: market points are not plotted, only the model surfaces, the
  quoted moneyness range and the per-expiry fit errors.
* Historical scenarios are SPX shocks applied to whichever underlying is selected; American ETF options are valued
  as European in the strategy builder.
