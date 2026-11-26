"""Forecast evaluation and the walk-forward engine.

Losses on a variance forecast f against a realized-variance proxy x (both over the same h days):
  QLIKE   log f + x / f. It differs from Patton's x/f - log(x/f) - 1 only by terms in x alone, so model differences,
          rankings and tests are the same, and it stays finite when x = 0 (a day with an unchanged close).
  MSE     (x - f)^2.
Both rank forecasts correctly with a noisy but conditionally unbiased proxy (Patton 2011), such as the sum of squared
daily returns.

Tests: Diebold-Mariano (mean loss differential over its Newey-West standard error; overlapping h-day targets make the
differentials serially correlated, so the lag length is at least h) and the model confidence set of Hansen, Lunde and
Nason (2011) from the arch package.

Walk-forward: at each refit date t_k (every refit_every trading days from the start of the evaluation period), a
model is refitted on training rows tau in [first usable row, t_k - h], i.e. only rows whose h-day target is already
observed at t_k, and then forecasts every day of the next block from data up to that day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qlike(f, x):
    f = np.asarray(f, dtype=float)
    return np.log(f) + np.asarray(x, dtype=float) / f


def mse(f, x):
    return (np.asarray(x, dtype=float) - np.asarray(f, dtype=float)) ** 2


def mincer_zarnowitz(f, x) -> dict:
    """OLS of the proxy on the forecast: intercept, slope and R^2 (slope 1, intercept 0 for an unbiased forecast)."""
    import statsmodels.api as sm

    r = sm.OLS(np.asarray(x, dtype=float), sm.add_constant(np.asarray(f, dtype=float))).fit()
    return {"intercept": float(r.params[0]), "slope": float(r.params[1]), "r2": float(r.rsquared)}


def dm_test(loss_a, loss_b, lags: int) -> dict:
    """Diebold-Mariano test of equal expected loss. d = loss_a - loss_b; positive t means b is better."""
    import statsmodels.api as sm

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    r = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return {"mean_diff": float(r.params[0]), "t": float(r.tvalues[0]), "p": float(r.pvalues[0])}


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, block: int = 22, reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS (arch.bootstrap.MCS, range statistic, stationary bootstrap). Returns each model's MCS
    p-value and whether it is in the set at the given size."""
    from arch.bootstrap import MCS

    m = MCS(losses, size=size, reps=reps, block_size=block, method="R", bootstrap="stationary", seed=seed)
    m.compute()
    pv = m.pvalues.iloc[:, 0]
    return pd.DataFrame({"mcs_p": pv, "in_mcs": pv >= size}).reindex(losses.columns)


def refit_points(dates: pd.DatetimeIndex, start, every: int) -> list[int]:
    """Positions of the refit dates: the first date on or after start, then every `every` trading days."""
    i0 = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(start))))
    return list(range(i0, len(dates), every))


def walk_forward(dates: pd.DatetimeIndex, start, every: int, h: int, fit_predict, first_train: int = 0) -> pd.Series:
    """Generic walk-forward. fit_predict(train_slice, pred_slice) -> array of forecasts for pred_slice, where
    train_slice covers positions [first_train, t_k - h] (targets observed by t_k) and pred_slice the block
    [t_k, t_k + every)."""
    out = np.full(len(dates), np.nan)
    pts = refit_points(dates, start, every)
    for j, k in enumerate(pts):
        end = pts[j + 1] if j + 1 < len(pts) else len(dates)
        train = slice(first_train, k - h + 1)
        pred = slice(k, end)
        out[pred] = fit_predict(train, pred)
    return pd.Series(out, index=dates)
