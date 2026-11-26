"""Average correlation of the largest index members: realized, implied, and forecasts of it.

Equicorrelation. For a basket with weights w and member variances s_i^2, the single correlation rho that reproduces
the basket variance s_B^2 is
    rho = (s_B^2 - sum w_i^2 s_i^2) / ((sum w_i s_i)^2 - sum w_i^2 s_i^2),
the formula behind Cboe's implied correlation indices.
  realized   s_B^2 and s_i^2 from daily returns over a window, weights = point-in-time top-50 market-cap weights
             (renormalized over members with returns on every day of the window)
  implied    s_B = SPX 30-day ATM implied vol, s_i = each member's 30-day ATM implied vol (OptionMetrics standardized
             options), same weights. Like Cboe's index, this uses the index's own option-implied vol for a basket of
             its 50 largest members; the basket is not the index, so the measure is a proxy and can exceed 1.
Forward target: realized equicorrelation over days t+1 .. t+h with the weights known at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equicorr(var_b: float, w: np.ndarray, var_i: np.ndarray) -> float:
    a = w * np.sqrt(var_i)
    den = a.sum() ** 2 - (a ** 2).sum()
    return float((var_b - (w ** 2 * var_i).sum()) / den) if den > 0 else np.nan


def weights_at(weights: pd.DataFrame, date) -> pd.Series:
    i = weights.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if i < 0:
        return pd.Series(dtype=float)
    w = weights.iloc[i]
    return w[w > 0]


def window_corr(ret: pd.DataFrame, w: pd.Series, rows: slice) -> float:
    R = ret.iloc[rows][w.index.intersection(ret.columns)]
    R = R.dropna(axis=1, how="any")
    if R.shape[0] < 5 or R.shape[1] < 10:
        return np.nan
    ww = w.reindex(R.columns).to_numpy(float)
    ww = ww / ww.sum()
    rb = R.to_numpy() @ ww
    return equicorr(np.var(rb, ddof=1), ww, np.var(R.to_numpy(), axis=0, ddof=1))


def realized_corr(ret: pd.DataFrame, weights: pd.DataFrame, window: int = 22, step: int = 1) -> pd.Series:
    """Trailing realized equicorrelation at each date (days t-window+1 .. t, weights known at t)."""
    out = np.full(len(ret), np.nan)
    for t in range(window - 1, len(ret), step):
        w = weights_at(weights, ret.index[t])
        if len(w):
            out[t] = window_corr(ret, w, slice(t - window + 1, t + 1))
    return pd.Series(out, index=ret.index)


def forward_corr(ret: pd.DataFrame, weights: pd.DataFrame, h: int = 22, step: int = 1) -> pd.Series:
    """Realized equicorrelation over days t+1 .. t+h with weights known at t (the forecast target)."""
    out = np.full(len(ret), np.nan)
    for t in range(0, len(ret) - h, step):
        w = weights_at(weights, ret.index[t])
        if len(w):
            out[t] = window_corr(ret, w, slice(t + 1, t + h + 1))
    return pd.Series(out, index=ret.index)


def implied_corr(index_iv: pd.Series, names_iv: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    out = {}
    for d, s_b in index_iv.dropna().items():
        if d not in names_iv.index:
            continue
        w = weights_at(weights, d)
        iv = names_iv.loc[d].reindex(w.index)
        ok = iv.notna() & (iv > 0)
        if ok.sum() < 30:
            continue
        ww = (w[ok] / w[ok].sum()).to_numpy(float)
        out[d] = equicorr(float(s_b) ** 2, ww, iv[ok].to_numpy(float) ** 2)
    return pd.Series(out).sort_index()
