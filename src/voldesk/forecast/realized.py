"""Daily variance measures from OHLC bars, and h-day realized-variance targets.

All measures are per day, in squared log-return units (multiply by 252 for an annual variance). With O, H, L, C the
day's open, high, low and close and C' the previous close:
  squared return   r^2 = ln(C/C')^2                                  (unbiased for the close-to-close variance)
  Parkinson        ln(H/L)^2 / (4 ln 2)                               (intraday range; ignores the overnight gap)
  Garman-Klass     0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2             (open-to-close, no drift)
  Rogers-Satchell  ln(H/C) ln(H/O) + ln(L/C) ln(L/O)                  (open-to-close, unbiased under any drift)
  overnight        ln(O/C')^2
The range-based measures are unbiased for the variance of a continuously observed Brownian path over the trading
session; with discrete ticks and a stale open they are biased down, which the forecast regressions absorb.

The forecast target at the close of day t is RV(t, h) = sum of r^2 over days t+1 .. t+h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def parkinson(high, low):
    return np.log(high / low) ** 2 / (4.0 * LN2)


def garman_klass(open_, high, low, close):
    return 0.5 * np.log(high / low) ** 2 - (2.0 * LN2 - 1.0) * np.log(close / open_) ** 2


def rogers_satchell(open_, high, low, close):
    return np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)


def overnight(open_, prev_close):
    return np.log(open_ / prev_close) ** 2


def forward_sum(x: pd.Series, h: int) -> pd.Series:
    """Sum of x over the next h observations (t+1 .. t+h), aligned at t; NaN where the window is incomplete or holds
    a NaN."""
    return x.rolling(h, min_periods=h).sum().shift(-h)


def annual_vol(var_per_day, h: int = 1):
    """Annualized volatility in percent from a variance over h days."""
    return 100.0 * np.sqrt(252.0 * np.asarray(var_per_day, dtype=float) / h)


def daily_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-day measures from a DataFrame with open/high/low/close (index = dates)."""
    o, h, l, c = (bars[k].astype(float) for k in ("open", "high", "low", "close"))
    pc = c.shift(1)
    out = pd.DataFrame(index=bars.index)
    out["r"] = np.log(c / pc)
    out["r2"] = out["r"] ** 2
    out["parkinson"] = parkinson(h, l)
    out["rs"] = rogers_satchell(o, h, l, c)
    out["overnight"] = overnight(o, pc)
    out["rs_on"] = out["rs"] + out["overnight"]            # range-based close-to-close variance: overnight gap + session
    return out
