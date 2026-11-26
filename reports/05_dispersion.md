# 05 · Dispersion and correlation trades

Index volatility is the volatility of a basket, so it depends on the average correlation of its members. Selling
index variance against bought member variance (dispersion) or selling a correlation swap is a bet that realized
correlation comes in below implied. Script: `scripts/dispersion_study.py`; module `voldesk.dispersion`; tables in
`results/dispersion/`.

## Construction (theoretical, variance-based)

Single-stock option chains were not pulled (166 companies × 30 years); the member legs use OptionMetrics'
standardized 30-day at-the-money implied vols, so these are **variance-swap and correlation-swap trades on
implied-vol strikes, not traded option portfolios**:

* Members: the point-in-time 50 largest S&P 500 members (survivorship handled, report 01), market-cap weights.
* Entries: the same 354 monthly roll dates as the SPX backtests through July 2025, held 21 trading days.
* Dispersion P&L = Σ wᵢ (RVᵢ − Kᵢ) − (RV_index − K_index), K = (30-day ATM implied vol)², RV = annualized realized
  variance over the holding period. A second version strikes the index leg at VIX² (the model-free variance-swap
  rate, which includes the index skew premium); true variance strikes for both legs would lie between the two.
* Correlation swap P&L = implied equicorrelation − realized equicorrelation (Cboe's formula).
* Costs (assumptions): 0.5 vol point on the index variance swap, 1.5 on each member, 0.02 correlation points per
  correlation swap, all per entry.

## Results (monthly cycles, 1996-2025)

| Trade | Mean per cycle | Hit rate | Sharpe (annualized) | t | Worst cycle |
|---|---:|---:|---:|---:|---:|
| Dispersion, ATM strikes, gross | +0.0073 var pts | 48% | 0.75 | 4.1 | −0.121 |
| Dispersion, ATM strikes, net of base costs | −0.0030 | 27% | −0.31 | −1.7 | −0.156 |
| Dispersion, ATM strikes, net of 2× costs | −0.0132 | 18% | −1.37 | −7.4 | −0.192 |
| Dispersion, index at VIX², gross | +0.0175 | 73% | 1.75 | 9.5 | −0.061 |
| Dispersion, index at VIX², net | +0.0072 | 46% | 0.75 | 4.0 | −0.083 |
| Short correlation swap, gross | +0.043 corr pts | 63% | 0.93 | 5.1 | −0.48 |
| Short correlation swap, net | +0.023 | 56% | 0.50 | 2.7 | −0.50 |

(Variance points are annualized variances: 0.01 is, for example, a realized vol of 20% against an implied vol of
17.3%, since 0.20² − 0.173² ≈ 0.01; P&L is per unit of variance notional.)

* **There is a correlation premium**: implied correlation averaged 0.382 and the following month's realized 0.339 on
  these dates (+0.043 per cycle, t = 5.1), paid with severe losses when correlation spikes (worst cycle −0.48).
* **Whether dispersion survives costs depends on the strikes.** With at-the-money strikes on both legs the gross
  premium (+0.0073) is smaller than the assumed costs; with the index sold at the VIX² rate it survives (t = 4.0 net).
  Since real single-stock variance strikes also sit above ATM implied variance, the tradable result lies between the
  two rows; without traded single-stock quotes this study cannot say where.

**Timing with the correlation forecast** (report 04): trading the correlation swap only when implied correlation
exceeds the walk-forward HAR forecast of realized correlation by more than 0.10 traded 23% of cycles (307 cycles with
forecasts, 2000-2025) and earned +0.040 per cycle across all of them (Sharpe 1.40, t 7.1, worst cycle −0.30). The
0.10 threshold was fixed before running and no other threshold was tried; it is still one rule on one sample, an
in-sample result that a new period would have to confirm. The same rule did not rescue the ATM-strike dispersion
trade (−0.0035 per cycle net).

## Limits

* No traded single-stock options: strikes are standardized implied vols, not executable variance-swap levels, and
  costs are assumptions.
* The index leg is the S&P 500, the basket its 50 largest members: realized index variance is not the basket's
  (Cboe's implied correlation indices share this mismatch).
* Variance and correlation swaps are OTC; their availability and margin are not modelled.
