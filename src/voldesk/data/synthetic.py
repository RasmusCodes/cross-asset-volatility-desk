"""SYNTHETIC option chains: a clearly labelled fallback for tests, CI and demonstrations without WRDS access.

Nothing here is market data. Every frame this module returns carries `synthetic = True` and `source = "SYNTHETIC"`,
and nothing in the research scripts reads it: they read data/raw and data/store only, and fail when those are
missing rather than falling back silently.

Model: a Heston underlying (parameters typical of an S&P 500 fit) prices European options on a strike grid for a set
of expiries; quotes get a bid-ask spread that widens with the price and in the wings, and are rounded to the tick.
The output has the columns of OptionMetrics opprcd, so the whole cleaning pipeline (voldesk.data.chains) and the
surface fits run on it unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..surface.heston import HestonParams, heston_price_cos

DEFAULT_PARAMS = HestonParams(v0=0.025, kappa=2.0, theta=0.04, sigma=0.9, rho=-0.75)


def synthetic_chain(date: str = "2024-06-03", spot: float = 5000.0, r: float = 0.04, q: float = 0.013,
                    params: HestonParams = DEFAULT_PARAMS, expiry_days=(7, 14, 30, 60, 91, 182, 365, 730),
                    n_strikes: int = 61, seed: int = 0, secid: int = 0, expiries=None, am_settlement: int = 0,
                    strike_step: float | None = None) -> pd.DataFrame:
    """One trade date of synthetic European option quotes in OptionMetrics opprcd columns (strike x 1000).
    `expiries` (absolute dates) overrides `expiry_days`; `strike_step` puts strikes on a fixed grid (so the same
    contracts exist on consecutive days, which multi-day backtest tests need)."""
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp(date)
    rows = []
    if expiries is None:
        exps = []
        for days in expiry_days:
            ex = d0 + pd.Timedelta(days=int(days))
            while ex.dayofweek >= 5:
                ex += pd.Timedelta(days=1)
            exps.append(ex)
    else:
        exps = [pd.Timestamp(e) for e in expiries if pd.Timestamp(e) > d0]
    for ex in exps:
        T = ((ex + pd.Timedelta(hours=16)) - (d0 + pd.Timedelta(hours=16))).total_seconds() / (365 * 86400)
        F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
        sd = np.sqrt(params.theta * T)
        if strike_step is not None:
            lo, hi = F * np.exp(-4.5 * sd - 0.05), F * np.exp(2.5 * sd + 0.05)
            K = np.arange(np.floor(lo / strike_step) * strike_step, hi + strike_step, strike_step)
        else:
            k = np.linspace(-4.5 * sd - 0.02, 2.5 * sd + 0.02, n_strikes)
            tick = 5.0 if spot > 1000 else 0.5
            K = np.unique(np.round(F * np.exp(k) / tick) * tick)
        for cp in (1, -1):
            px = heston_price_cos(np.full(K.shape, float(cp)), F, K, T, D, params)
            px = np.maximum(px, 0.0)
            half = np.maximum(0.05, 0.02 * px + 0.002 * spot * np.abs(np.log(K / F)) / max(np.sqrt(T), 0.05) * 0.01)
            noise = rng.normal(0, 0.15, K.size) * half
            mid = np.maximum(px + noise, 0.0)
            bid = np.maximum(np.floor((mid - half) / 0.05) * 0.05, 0.0)
            ask = np.ceil((mid + half) / 0.05) * 0.05
            for j in range(K.size):
                oid = int(ex.strftime("%Y%m%d")) * 100000 + int(round(K[j])) * 10 + (1 if cp > 0 else 2)
                rows.append({"secid": secid, "date": d0, "exdate": ex, "last_date": d0, "cp_flag": "C" if cp > 0 else "P",
                             "strike_price": K[j] * 1000.0, "best_bid": bid[j], "best_offer": ask[j], "volume": 0,
                             "open_interest": 100, "impl_volatility": np.nan, "delta": np.nan, "gamma": np.nan,
                             "vega": np.nan, "theta": np.nan, "optionid": oid, "cfadj": 1.0,
                             "am_settlement": am_settlement, "contract_size": 100.0, "ss_flag": "0", "forward_price": np.nan,
                             "expiry_indicator": None, "root": "SYN", "suffix": None})
    out = pd.DataFrame(rows)
    out["synthetic"] = True
    out["source"] = "SYNTHETIC"
    return out


def synthetic_curve(r: float = 0.04):
    """A flat discount curve object with the ZeroCurve interface (rate(date, T), discount(date, T))."""
    class Flat:
        label = "SYNTHETIC flat curve"

        def rate(self, date, T):
            return np.full(np.shape(np.atleast_1d(T)), r, dtype=float)

        def discount(self, date, T):
            T = np.atleast_1d(np.asarray(T, float))
            return np.exp(-r * T)
    return Flat()


def synthetic_paths(n_days: int = 1500, s0: float = 3000.0, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC daily index bars from a two-regime GARCH-like process (for forecasting-code tests)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    h = np.empty(n_days)
    r = np.empty(n_days)
    h[0] = 1e-4
    for t in range(n_days):
        if t:
            h[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    close = s0 * np.exp(np.cumsum(r))
    rng_hl = np.abs(rng.normal(0, 1, n_days)) * np.sqrt(h) * 0.8
    df = pd.DataFrame({"open": close * np.exp(-r / 2), "high": close * np.exp(rng_hl), "low": close * np.exp(-rng_hl),
                       "close": close}, index=dates)
    df["synthetic"] = True
    return df
