# 02 · Volatility surface models compared

Five models fitted the same way to the same SPX quotes on every month-end from January 1996 to October 2026, plus
3 November 2026 (371 dates; OptionMetrics to August 2025, Cboe after), then stressed with resampled quotes, held-out
expiries and held-out wings. Two advanced models, local-stochastic volatility and rough Bergomi, are calibrated on
two of those dates. Scripts: `scripts/surface_panel.py`, `scripts/surface_sensitivity.py`, `scripts/rough_study.py`,
`scripts/lsv_study.py`. Tables: `results/surfaces/`, `results/rough/`, `results/lsv/`.

**Inputs per date.** AM-settled standard-cycle expiries with 7 days to 3 years left (the one expiry family that
exists throughout 1996-2026; median 10 expiries and 561 quotes per date), out-of-the-money quotes with two-sided
markets and forward |delta| ≥ 1%, implied vols from our parity forwards (report 01). Every model is scored on
model-minus-mid vol (vol points) and on the share of quotes it puts inside the bid-ask.

## The models

| Model | Parameters | Fitted how | Static arbitrage |
|---|---|---|---|
| SVI (raw) | 5 per expiry | quasi-explicit start + SLSQP with butterfly and Lee constraints on a wide grid; calendar order imposed where cheap | free only where constraints hold |
| SABR, β = 1 | 3 per expiry | least squares on Hagan's lognormal formula; expiries joined linearly in total variance | not guaranteed |
| eSSVI | 3 per expiry | slice by slice under the closed-form sufficient conditions | free everywhere by construction |
| SSVI (power law) | 3 + one ATM variance per expiry | one joint fit | free everywhere by construction |
| Heston | 5 in total | joint least squares (COS pricing during calibration, adaptive quadrature for the reported errors) | free (a consistent model) |
| Local vol (Dupire) | none of its own | derived from any of the above: σ²_LV = ∂w/∂T / g(k, T) | valid only where the parent surface is arbitrage-free |
| Local-stochastic vol (two dates) | Heston's 5 + a leverage function | particle method against eSSVI's local vol | inherits eSSVI's |
| Rough Bergomi (two dates) | 3 (H, η, ρ) + forward variance read off eSSVI | least squares, Monte Carlo with common random numbers | free (a consistent model) |

## Results

| Model | Median RMSE (vol pts) | 90th pct | Inside bid-ask (median date) | Butterfly violations, extrapolated | Calendar violations, extrapolated | Local variance valid | Seconds per date |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVI | **0.24** | 0.52 | **89%** | 0.06% | 8.2% | 95.5% | 6.6 |
| SABR | 0.37 | 0.66 | 70% | 0% | 3.1% | 98.8% | 0.25 |
| eSSVI | 0.52 | 0.99 | 63% | 0% | 0% | 100% | 0.50 |
| SSVI | 0.78 | 1.24 | 41% | 0% | 0% | 100% | 0.18 |
| Heston | 0.97 | 1.65 | 26% | 1.2% (numerical) | 0.7% | 97.9% | 11.7 |

Violation shares are of grid points (k from −1.5 to 0.6 at every expiry and two points between expiries); none of the
models violates butterfly arbitrage inside the quoted range. SVI's constraints were all met on 56% of dates: on the
rest at least one slice ended marginally infeasible and was kept as the best nearly-feasible fit, which is where its
extrapolated calendar crossings come from.

![Fit error by model over time](figures/model_rmse_time.png)

**By maturity** (median RMSE, vol points). Every model is worst at the short end, and the ranking holds at every
maturity except that eSSVI overtakes SABR beyond a year:

| | SVI | SABR | eSSVI | SSVI | Heston |
|---|---:|---:|---:|---:|---:|
| ≤ 30 days | 0.37 | 0.53 | 0.91 | 1.37 | 1.96 |
| 31-91 days | 0.23 | 0.29 | 0.51 | 0.67 | 0.79 |
| 92-365 days | 0.18 | 0.30 | 0.38 | 0.53 | 0.62 |
| > 1 year | 0.17 | 0.31 | 0.24 | 0.51 | 0.57 |

**By era.** 2008-2012 and 2020-2026 are hardest for every model; eSSVI degrades most after 2020 (0.37 → 0.90),
when short-dated smiles grew steeper than its three parameters per expiry can follow:

| | SVI | SABR | eSSVI | SSVI | Heston |
|---|---:|---:|---:|---:|---:|
| 1996-2001 | 0.24 | 0.32 | 0.37 | 0.57 | 0.90 |
| 2002-2007 | 0.16 | 0.31 | 0.37 | 0.63 | 0.85 |
| 2008-2012 | 0.30 | 0.47 | 0.59 | 0.92 | 1.31 |
| 2013-2019 | 0.24 | 0.37 | 0.51 | 0.73 | 0.88 |
| 2020-2026 | 0.36 | 0.43 | 0.90 | 1.04 | 1.03 |

**Stress days** (single-date fits, RMSE in vol points): on 24 August 2015, the morning of the flash crash, every
model misses by 2.8-4.9 points; on 10 October 2008 and 16 March 2020 SVI is the only model under 2 (1.47, 1.84).
Calm dates such as 30 June 2017 (SVI 0.19, SABR 0.27) are the opposite extreme.

**Heston parameters** across the 371 dates: median κ 3.5, θ 0.055, σ (vol of variance) 1.16, ρ −0.71, Feller
ratio 0.32 (never above 1: the fitted variance process always reaches zero). σ ranges 0.34-3.4 and κ hits its bound
of 20 on some dates: one diffusive factor cannot match short- and long-dated skews at once.

## Sensitivity to input data (8 dates, 1998-2026)

| | Noise: s.d. of fitted vol when mids are redrawn inside the bid-ask (vol pts) | | Held-out expiry RMSE | Wing extrapolation (fit on \|Δ\| ≥ 10%, score 1-10% Δ) | |
|---|---:|---:|---:|---:|---:|
| | 30-day, k = −0.2 | 30-day ATM | | RMSE | bias |
| SVI | 0.090 | 0.014 | **0.32** | 0.99 | −0.25 |
| SABR | 0.046 | 0.022 | 0.41 | **0.94** | −0.11 |
| eSSVI | 0.031 | 0.014 | 0.58 | 1.54 | −0.72 |
| SSVI | 0.027 | 0.009 | 0.67 | 1.76 | −0.77 |
| Heston | 0.032 | 0.008 | 0.71 | 2.22 | −0.88 |

SVI's flexibility shows up as the largest response to quote noise in the short-dated put wing (three times eSSVI's),
yet it still interpolates held-out expiries best. Every model understates wing vols when it does not see them
(negative bias); SABR's wings, set by its vol-of-vol, extrapolate best.

## SPY, QQQ and IWM

Fitted on month-end chains on five dates each (2018-2025) with the same code. These are American options; the fits
use European-equivalent vols of out-of-the-money quotes, which overstate put vols by the early-exercise premium
(small for short-dated out-of-the-money puts, larger for long-dated ones and at high rates). Median RMSE (vol points):

| | SVI | SABR | eSSVI | SSVI | Heston |
|---|---:|---:|---:|---:|---:|
| SPY | 0.35 | 0.92 | 1.07 | 1.34 | 1.27 |
| QQQ | 0.46 | 0.59 | 1.06 | 1.17 | 1.43 |
| IWM | 0.34 | 0.91 | 1.20 | 1.46 | 1.61 |

The ordering matches SPX; all models do worse than on SPX (wider ETF spreads put fewer quotes inside; the
European approximation adds error to the puts).

## Rough volatility (advanced extension)

Rough models predict an at-the-money skew that scales like T^(H−1/2) with H well below ½. On the 371 SVI-fitted SPX
surfaces the slope of log ATM skew on log T (7 days to 6 months) has median −0.33 (interquartile −0.39 to −0.27; the
power law fits with median R² 0.99), an implied H of about 0.17, in the range the literature reports. **But fitted
Heston surfaces produce the same slope over these maturities (median −0.38):** with a fast mean reversion (κ ≈ 3.5)
Heston's skew decays over a week to six months in a way a power law describes equally well. The two separate only
below a week, which the standard-expiry panel deliberately excludes, so the panel does not discriminate between them.

![ATM skew slope](figures/rough_skew_slope.png)

**Rough Bergomi, calibrated.** `voldesk.surface.rough` prices rough Bergomi (Bayer, Friz & Gatheral 2016) by Monte
Carlo: exact joint simulation of the Volterra process and its Brownian driver (Cholesky factorisation), the price's
independent Brownian part integrated out in closed form (the mixing estimator, as in McCrickerd & Pakkanen 2018),
and random numbers held fixed across evaluations, so the fitted smile is a smooth function of (H, η, ρ) and a
least-squares optimiser (trust-region reflective, several starts) can be used. The forward variance curve
ξ0(t) = E[v_t] is not fitted: it is read off the date's arbitrage-free eSSVI surface as its variance-swap strip
(log-contract replication of each slice), so only three parameters are calibrated. Rough Bergomi and Heston are
fitted to the same quotes with the same weights, every quote of the four expiries between 17 days and four months
(407 on 30 June 2017, 1,083 on 31 March 2020), and scored on model-minus-mid vol. Script `scripts/rough_study.py`,
tables `results/rough/rbergomi_*.csv`. Pooled RMSE in vol points:

| Model | Parameters | 30 Jun 2017 | 31 Mar 2020 |
|---|---|---:|---:|
| Rough Bergomi, forward variance from the variance-swap strip | 3 | 0.61 | 1.78 |
| Heston, refitted to these quotes | 5 | **0.58** | **1.59** |
| Rough Bergomi, forward variance from ATM implied variance (cross-check) | 3 | 2.66 | 6.39 |
| Heston, the panel's whole-surface fit | 5 | 1.36 | 3.18 |
| SVI / SABR / eSSVI, the panel's per-expiry fits | 5 / 3 / 3 per expiry | 0.25 / 0.26 / 0.74 | 0.40 / 1.75 / 1.20 |

**On these two dates rough Bergomi does not fit the short end better than Heston.** It ties on the calm date (0.61
against 0.58) and loses in March 2020 (1.78 against 1.59), with two parameters fewer. By expiry the two trade places:
rough Bergomi is ahead at 21 and 77 days in 2017 (0.89 against 0.90, 0.28 against 0.36) and at 45 days in 2020
(1.72 against 1.85), Heston everywhere else, most clearly on the 17-day smile of March 2020 (2.39 against 1.73).
The fitted rough models have H = 0.080 and 0.120, η = 2.63 and 2.15 and ρ = −0.93 and −0.91. These H do not match
the ATM-skew slopes of the same surfaces (implied H 0.17 and 0.03 in `atm_skew_power_law.csv`): fitted to whole
smiles, H trades off against η and ρ. The Heston refits get to the short end only with κ ≈ 13 and a vol-of-variance
of 1.8 and 4.4 (the whole-surface fits have κ 3.0 and 5.6), parameters that serve these four expiries and nothing
beyond them. Both are far behind per-expiry SVI.

The level of the forward variance matters more than the model. With a negative skew the ATM implied variance lies
below the variance-swap variance (by 2.3 to 2.8 vol points in 2017 and 7.4 to 8.1 in March 2020 at these expiries,
`rbergomi_xi0_gap.csv`), and a rough Bergomi whose E[v_t] is set at the ATM level cannot recover, whatever (H, η, ρ):
2.66 and 6.39 vol points. That, not the optimiser, is what failed in this report's earlier coarse calibration (a flat
forward variance at the ATM level, a 60-point grid), which these results replace.

**Time steps decide the comparison.** The Monte Carlo uses the variance at the start of each step, which misses how a
rough variance reacts within the step: with H around 0.1 and η around 2 the variance moves by about its own size
inside a step even at thousands of steps a year, and the short-dated smile converges slowly as the grid is refined.
At the shortest expiry, the calibrated model's call vol one standard deviation out of the money sits 0.76 (2017) and
1.89 (2020) vol points above its 1,600-step value at 50 steps (about 900 and 1,100 steps a year, fine by usual
standards) and 0.51 and 1.12 at 100 steps (`rbergomi_convergence.csv`). On a uniform grid of 1,460 steps a year (83
and 67 steps inside the shortest expiry) the same calibration drives ρ to its bound of −0.999 and reports 0.69 and
1.47 vol points, on paper better than Heston in March 2020, by fitting the discretisation error as much as the model
(`rbergomi_coarse.csv`). The fits above therefore use at least 400 steps inside every expiry (7,000 to 8,700 a year
on the first). There the call vol is within 0.18 and 0.30
of its 1,600-step value and the at-the-money vol within 0.02 and 0.14, and recomputing every fitted vol from the same
paths on a grid twice as fine moves them by 0.05 at the median and 0.17 to 0.19 at the 95th percentile, and the
pooled RMSE by +0.02 to +0.03. Monte Carlo standard errors of the fitted vols are 0.04 and 0.08 at the median. The
remaining numerical error is well below the 0.18 vol-point gap of March 2020 but not below the 0.03 gap of June 2017:
the calm date is a tie within numerical error.

## Local-stochastic volatility (advanced extension)

Local-stochastic volatility (LSV) keeps a stochastic variance but scales it by a leverage function L(t, S) chosen so
that the model reprices the whole vanilla surface. By Gyöngy's theorem that holds when L(t, S)² E[V_t | S_t = S]
equals the Dupire local variance. `voldesk.surface.lsv` solves this forward in time with the particle method of Guyon
& Henry-Labordère (2012). The variance is the date's fitted Heston process at its full vol-of-vol, the target is the
Dupire local vol of the arbitrage-free eSSVI surface (σ²_LV = ∂w/∂T / g), and E[V_t | S_t] is estimated at every
time step by kernel regression across 200,000 particles. Prices then come from an independent simulation of 400,000
paths with the leverage frozen, so their standard errors are exact. Pure local vol and pure Heston run through the
same discretisation and time grid: full-truncation Euler for the variance, a log-price step that keeps the forward
an exact martingale, steps of 1/2,920 year for the first five weeks rising to 1/1,460 year from about five months
(3,944 and 4,294 steps to the last expiry). Differences between the three are therefore model differences. Script
`scripts/lsv_study.py`, tables in `results/lsv/`, dates 30 June 2017 and 31 March 2020.

**Validation with a known answer.** Aimed at a Heston model's own local variance (by Fourier inversion) with the same
Heston variance process, the exact leverage is 1. On both dates' Heston parameters the particle method returns a
leverage between 0.96 and 1.06 for the 1st to 99th percentile of particles at every reported time (medians 0.994 to
1.002). The LSV smile matches the Heston smile to 0.012, 0.038 and 0.057 vol points (2017) and 0.055, 0.106 and 0.086
(2020) at three weeks, three months and one year (largest |z| 2.7). Two settings differ from the paper, both chosen
on this test with the 2017 parameters (RMSE in vol points against the exact Heston smile at nine strikes, largest
|z| in brackets):

| Settings | 3 weeks | 3 months | 1 year |
|---|---:|---:|---:|
| Paper: Nadaraya-Watson regression, bandwidth floor t_min = 0.25, daily steps | 0.147 (17.7) | 0.100 (9.6) | 0.018 (1.0) |
| Local-linear regression, t_min = 0.25, daily steps | 0.133 (16.7) | 0.075 (8.0) | 0.028 (2.6) |
| Local-linear regression, t_min = 0.25, graded steps | 0.047 (9.8) | 0.065 (4.9) | 0.065 (3.0) |
| **Used:** local-linear regression, t_min = 0.05, graded steps | **0.012 (1.8)** | **0.038 (2.5)** | 0.057 (2.7) |

The paper's bandwidth floor over-smooths a smile three weeks long; at one year its settings happened to land closer
(0.018 against 0.057 vol points).

**Repricing the quoted vanillas.** LSV against the eSSVI surface it was calibrated to, at every quoted strike and
maturity (862 quotes on 11 expiries in 2017, 2,180 on 14 in 2020); z is the error over its Monte Carlo standard
error, so noise alone would put 95.4% of quotes within ±2 and 99.7% within ±3:

| Date | Model | RMSE vs eSSVI | Bias | MC s.e. | \|z\| ≤ 2 | \|z\| ≤ 3 | Max \|z\| | RMSE vs mid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 30 Jun 2017 | LSV | 0.054 | −0.030 | 0.027 | 90.7% | 99.7% | 3.3 | 0.56 |
| | LSV, second particle seed | 0.047 | −0.015 | 0.027 | 81.0% | 95.4% | 3.9 | 0.55 |
| | Local vol, same scheme | 0.031 | −0.003 | 0.027 | 94.2% | 99.0% | 3.8 | 0.54 |
| | Heston (the date's fit) | 0.619 | −0.029 | 0.027 | 7.3% | 14.0% | 37.4 | 0.96 |
| 31 Mar 2020 | LSV | 0.086 | −0.029 | 0.052 | 93.1% | 98.8% | 3.3 | 0.94 |
| | LSV, second particle seed | 0.113 | −0.049 | 0.052 | 79.6% | 95.4% | 4.4 | 0.94 |
| | Local vol, same scheme | 0.084 | 0.068 | 0.052 | 62.1% | 93.8% | 4.0 | 0.96 |
| | Heston (the date's fit) | 2.613 | 0.144 | 0.052 | 4.8% | 7.1% | 134.2 | 2.54 |

(Vol points; the eSSVI surface itself misses the mids by 0.53 and 0.97.) LSV reprices the surface to within a few
hundredths of a vol point, close to Monte Carlo error but not entirely inside it. What remains is systematic and
small: the largest bias is at the shortest expiries (−0.08 vol points on average at three weeks in 2017, in both
particle seeds; up to −0.12 at six weeks in 2020), and the worst single quotes miss by 0.19 (2017) and 0.56 (2020).
Pure local vol through the same time grid errs by similar amounts (0.03 and 0.08 RMSE, with a +0.07 bias in 2020
from time discretisation), and a second particle seed gives errors of the same size (0.047 and 0.113): what is left
is discretisation and regression noise, not a failure to calibrate. Against the market LSV is as
good as the surface it was built on (0.56 and 0.94 vol points against eSSVI's 0.53 and 0.97). The calibrated leverage
stays between 0.76 and 1.37 (2017) and 0.64 and 1.63 (2020) for the 1st to 99th percentile of particles.

**What the dynamics change: forward smile and a cliquet.** With vanillas pinned down, the three models still disagree
about products that depend on the future smile. Three-month implied vols starting today and starting in six months
(strikes as a share of the forward; forward-start options pay on S(9m)/S(6m) against the forward return), Monte
Carlo standard errors at most 0.061 vol points:

| Date | Model | Spot start, 90 / 100 / 110% | Starting in 6m, 90 / 100 / 110% | Spot skew 90−110 | Forward skew 90−110 | Forward / spot |
|---|---|---|---|---:|---:|---:|
| 30 Jun 2017 | Local vol | 17.6 / 10.2 / 8.2 | 20.7 / 15.8 / 15.3 | 9.4 | 5.4 | 57% |
| | LSV | 17.6 / 10.2 / 8.3 | 21.7 / 14.0 / 14.2 | 9.3 | 7.5 | 81% |
| | Heston | 17.7 / 10.3 / 8.6 | 21.8 / 14.1 / 14.6 | 9.1 | 7.3 | 80% |
| 31 Mar 2020 | Local vol | 45.7 / 38.6 / 31.6 | 30.9 / 27.5 / 27.0 | 14.1 | 3.9 | 27% |
| | LSV | 45.7 / 38.5 / 31.4 | 31.1 / 21.7 / 19.2 | 14.2 | 11.9 | 83% |
| | Heston | 45.7 / 37.9 / 30.3 | 31.0 / 21.4 / 19.2 | 15.4 | 11.8 | 77% |

A cliquet with twelve monthly resets paying max(0, Σ clip(R_i, −1%, +1%)) after one year, R_i each month's return in
excess of carry (per cent of notional, undiscounted, standard errors at most 0.005):

| Date | Local vol | LSV | LSV, second particle seed | Heston | Positive months, LV / LSV / Heston |
|---|---:|---:|---:|---:|---|
| 30 Jun 2017 | 1.81 | 3.07 | 3.06 | 3.06 | 6.55 / 7.61 / 7.61 of 12 |
| 31 Mar 2020 | 1.64 | 3.93 | 3.93 | 3.94 | 6.27 / 7.93 / 7.94 of 12 |

**With the variance process at Heston's full vol-of-vol, LSV prices the vanillas like local vol and the forward smile
like Heston.** Its forward skew keeps 81% and 83% of the spot skew (local vol: 57% and 27%), and its cliquet price is
within 0.01% of notional of Heston's and 1.7 and 2.4 times local vol's. In local vol the skew is a function of spot
and flattens as the start date recedes; in a stochastic-volatility model it regenerates, so more months end up
positive under the skewed return distribution. Calibration noise is small next to these differences: the second
particle seed moves forward vols by at most 0.09 vol points and the cliquet by less than 0.01% of notional. Vanilla
quotes do
not say which dynamics are right. The split between local and stochastic volatility (the "mixing", here simply all
of Heston's vol-of-vol) would have to come from forward-start or cliquet prices, which this dataset does not contain.

## Practical strengths and weaknesses

* **SVI** fits closest and interpolates best, and is the right choice for marking and interpolating quotes inside
  the quoted range. Its extrapolated wings are where its arbitrage lives (8% of extrapolated calendar grid points),
  so anything that integrates over the whole surface (local vol, variance swaps, densities) needs care with it.
* **SABR (β = 1)** is the best compromise for a desk: second on fit, best on wing extrapolation, cheap (0.25 s per
  date), parameters with trading meaning (ρ = skew, ν = curvature). It is a per-expiry model with no guarantee
  across expiries.
* **eSSVI** is the right input for local volatility and anything that must be arbitrage-free everywhere: zero
  violations on 371 dates, at the price of roughly twice SVI's error and a short end it cannot follow since 2020.
* **SSVI** buys global consistency with very few parameters and pays for it in fit (41% inside the bid-ask).
* **Heston** is a dynamic model, not an interpolator: worst fit, most stable parameters under quote noise, and a
  consistent price for path-dependent products, but not one that agrees with the vanillas. Its short-dated skew is
  too flat and its vol-of-variance is far above Feller's bound.
* **Local vol** inherits every flaw of its parent surface: 95.5% of the SVI grid gives a valid local variance, 100%
  of the eSSVI and SSVI grids. It reprices the vanillas exactly by construction and flattens the forward skew.
* **Local-stochastic vol** reprices the parent surface's vanillas (to 0.05-0.11 vol points here) and adds
  stochastic-volatility dynamics, the right structure for path-dependent products. Its exotic prices depend on how
  much vol-of-vol it is given (with all of Heston's, a cliquet comes out 1.7-2.4 times the local-vol price), which
  vanilla quotes cannot settle. Calibration takes 2-4 minutes per date at 200,000 particles.
