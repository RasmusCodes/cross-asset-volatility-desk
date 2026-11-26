"""Dispersion and correlation trades between the S&P 500 and its 50 largest members (theoretical, variance-based).

What is traded. Option chains were pulled for the index and ETFs, not for 166 single stocks, so the single-stock legs
are priced off OptionMetrics' standardized 30-day at-the-money implied vols rather than traded contracts. The trades
are therefore defined on variance, with strikes set from implied vol and payoffs from realized variance:

  dispersion   long a basket of member variance swaps (weights = point-in-time top-50 market-cap weights), short an
               index variance swap. Per cycle (entry at the close, held for `horizon` trading days):
                   P&L = sum_i w_i (RV_i - K_i) - (RV_I - K_I),   K = (30-day ATM implied vol)^2,
               RV = annualised realized variance of daily log returns over the holding period. The index strike uses
               the SPX ATM vol by default; strike="vix" uses VIX^2 (the model-free 30-day variance swap rate) instead.
  correlation  short a correlation swap on the same basket: P&L = rho_implied - rho_realized (equicorrelation, see
               voldesk.forecast.correlation), in correlation points.
Costs (assumptions, varied in the report): variance swaps cost spread_I vol points on the index and spread_i on each
member per entry, converted to variance as 2 sigma x spread; correlation swaps cost `corr_cost` points per entry.
Using ATM vol for variance strikes ignores the skew premium in true variance-swap strikes (larger for the index than
for single stocks), so the ATM version understates what an index variance seller would collect; the VIX variant
brackets it from the other side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..forecast.correlation import equicorr, weights_at


def dispersion_cycles(index_ret: pd.Series, names_ret: pd.DataFrame, weights: pd.DataFrame, index_iv: pd.Series,
                      names_iv: pd.DataFrame, entries: list, horizon: int = 21, vix: pd.Series | None = None,
                      spread_index: float = 0.005, spread_name: float = 0.015, corr_cost: float = 0.02) -> pd.DataFrame:
    rows = []
    idx = index_ret.index
    for e in entries:
        e = pd.Timestamp(e)
        if e not in idx or e not in index_iv.index or e not in names_iv.index:
            continue
        i0 = idx.get_loc(e)
        if i0 + horizon >= len(idx):
            break
        days = idx[i0 + 1: i0 + 1 + horizon]
        w = weights_at(weights, e)
        w = w[w.index.isin(names_ret.columns)]           # members without OptionMetrics price rows drop out
        ivn = names_iv.loc[e].reindex(w.index)
        R = names_ret.reindex(days)[w.index]
        ok = ivn.notna() & (ivn > 0) & R.notna().all(axis=0)
        if ok.sum() < 30:
            continue
        ww = (w[ok] / w[ok].sum()).to_numpy(float)
        iv_i = ivn[ok].to_numpy(float)
        Rn = np.log1p(R.loc[:, ok].to_numpy(float))
        rv_i = (Rn ** 2).mean(axis=0) * 252
        ri = index_ret.reindex(days).to_numpy(float)
        rv_I = float(np.mean(ri ** 2) * 252)
        s_I = float(index_iv.loc[e])
        K_I = s_I ** 2
        K_vix = float((vix.loc[e] / 100.0) ** 2) if vix is not None and e in vix.index and np.isfinite(vix.loc[e]) else np.nan
        basket_K = float(np.sum(ww * iv_i ** 2))
        basket_RV = float(np.sum(ww * rv_i))
        cost = 2 * s_I * spread_index + float(np.sum(ww * 2 * iv_i * spread_name))
        rho_impl = equicorr(K_I, ww, iv_i ** 2)
        rb = Rn @ ww
        rho_real = equicorr(float(np.var(rb, ddof=0) * 252), ww, rv_i)
        rows.append({"entry": e, "exit": days[-1], "n_names": int(ok.sum()), "index_iv": s_I, "basket_iv_rms": np.sqrt(basket_K),
                     "index_rv": np.sqrt(rv_I), "basket_rv_rms": np.sqrt(basket_RV),
                     "pnl_dispersion": (basket_RV - basket_K) - (rv_I - K_I),
                     "pnl_dispersion_vixstrike": (basket_RV - basket_K) - (rv_I - K_vix) if np.isfinite(K_vix) else np.nan,
                     "cost_dispersion": cost, "rho_implied": rho_impl, "rho_realized": rho_real,
                     "pnl_corr": rho_impl - rho_real, "cost_corr": corr_cost})
    return pd.DataFrame(rows).set_index("entry")
