"""Forecasts of persistent level series h days ahead (implied vol, the term-structure slope, skew, implied
vol-of-vol), and their evaluation against the random walk.

Models, all direct h-step projections refitted on expanding windows by ``evaluate.walk_forward``:
  random walk      y(t+h) = y(t): the baseline every other model has to beat
  AR(1)            OLS of y(t+h) on y(t)
  HAR              OLS of y(t+h) on y(t) and its means over the last 5 and 22 days
  HAR-X            HAR plus economically motivated predictors
  Heston form      for a volatility index: OLS of y(t+h)^2 on y(t)^2, square-rooted (sqrt_affine_forecaster)
  gradient boost   trees on the change y(t+h) - y(t), added back to y(t). Trees cannot extrapolate, so they model
                   the change: a level beyond the training range stays reachable
Targets modelled in logs (VIX, VVIX) are fitted on log levels and retransformed with Duan's smearing factor.

Evaluation against the random walk: RMSE and MAE in the target's units; out-of-sample R^2 = 1 - SSE / SSE(random
walk); the share of days on which the forecast change and the realised change have the same sign (days where either
is zero are left out); Diebold-Mariano tests on squared errors (Newey-West, lags max(h, 5)); and the model confidence
set on squared errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as ev

GBM_LEVEL_PARAMS = {"loss": "squared_error", "learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 15,
                    "min_samples_leaf": 100, "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}


def har_frame(y: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_d": y, f"{prefix}_w": y.rolling(5, min_periods=5).mean(),
                         f"{prefix}_m": y.rolling(22, min_periods=22).mean()})


def ols_forecaster(X: np.ndarray, y_future: np.ndarray, log: bool):
    """fit_predict for walk_forward: OLS with an intercept of y(t+h) (log y(t+h) when log=True) on X; forecasts in
    level units (log targets retransformed with Duan's smearing factor)."""
    X = np.asarray(X, dtype=float)
    yt_all = np.log(y_future) if log else np.asarray(y_future, dtype=float)

    def fp(train, pred):
        Xt, yt = X[train], yt_all[train]
        ok = np.all(np.isfinite(Xt), axis=1) & np.isfinite(yt)
        A = np.column_stack([np.ones(int(ok.sum())), Xt[ok]])
        beta, *_ = np.linalg.lstsq(A, yt[ok], rcond=None)
        smear = float(np.mean(np.exp(yt[ok] - A @ beta))) if log else 1.0
        Xp = X[pred]
        out = np.full(len(Xp), np.nan)
        okp = np.all(np.isfinite(Xp), axis=1)
        z = np.column_stack([np.ones(int(okp.sum())), Xp[okp]]) @ beta
        out[okp] = np.exp(z) * smear if log else z
        return out

    return fp


def sqrt_affine_forecaster(x_now: np.ndarray, y_future: np.ndarray):
    """fit_predict for walk_forward: the Heston form for a volatility index. In one-factor Heston the squared index is
    affine in the variance state, so its conditional mean h days ahead is affine in today's square: OLS of y(t+h)^2 on
    x(t)^2. The forecast of y(t+h) is the square root of the fitted value times c = training mean of y(t+h) /
    sqrt(fitted). The square root of a conditional mean exceeds the conditional mean of the square root (Jensen); c
    removes that gap when the log index has a roughly constant conditional variance, the role Duan's smearing factor
    plays for the log models."""
    X = np.asarray(x_now, dtype=float) ** 2
    yl = np.asarray(y_future, dtype=float)
    Y = yl ** 2

    def fp(train, pred):
        x, yv, yr = X[train], Y[train], yl[train]
        ok = np.isfinite(x) & np.isfinite(yv)
        A = np.column_stack([np.ones(int(ok.sum())), x[ok]])
        beta, *_ = np.linalg.lstsq(A, yv[ok], rcond=None)
        c = float(np.mean(yr[ok] / np.sqrt(np.maximum(A @ beta, 1e-12))))
        return c * np.sqrt(np.maximum(beta[0] + beta[1] * X[pred], 1e-12))

    return fp


def gbm_change_forecaster(F: np.ndarray, y_now: np.ndarray, y_future: np.ndarray, log: bool, params=None):
    """fit_predict for walk_forward: boosted trees on the change (log change when log=True) from y(t) to y(t+h)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    F = np.asarray(F, dtype=float)
    y_now = np.asarray(y_now, dtype=float)
    y_future = np.asarray(y_future, dtype=float)
    dy = np.log(y_future) - np.log(y_now) if log else y_future - y_now

    def fp(train, pred):
        Ft, dt = F[train], dy[train]
        ok = np.all(np.isfinite(Ft), axis=1) & np.isfinite(dt)
        m = HistGradientBoostingRegressor(**(params or GBM_LEVEL_PARAMS)).fit(Ft[ok], dt[ok])
        smear = float(np.mean(np.exp(dt[ok] - m.predict(Ft[ok])))) if log else 0.0
        Fp, yp = F[pred], y_now[pred]
        out = np.full(len(Fp), np.nan)
        okp = np.all(np.isfinite(Fp), axis=1) & np.isfinite(yp)
        p = m.predict(Fp[okp])
        out[okp] = yp[okp] * np.exp(p) * smear if log else yp[okp] + p
        return out

    return fp


def evaluate_levels(fc: pd.DataFrame, y_future: pd.Series, y_now: pd.Series, h: int, start,
                    base: str = "Random walk", block: int = 22, reps: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Table of RMSE, MAE, out-of-sample R^2 against the random walk, direction hit rate, DM t-statistic against the
    random walk (positive = better than it) and MCS p-value, on the dates where every model has a forecast."""
    sample = fc.loc[start:].dropna().index.intersection(y_future.dropna().index).intersection(y_now.dropna().index)
    F, Y, Y0 = fc.loc[sample], y_future.loc[sample], y_now.loc[sample]
    E2 = pd.DataFrame({m: (Y - F[m]) ** 2 for m in F})
    sse_rw = float(((Y - Y0) ** 2).sum())
    lags = max(h, 5)
    rows = []
    for m in F:
        dpred, dreal = F[m] - Y0, Y - Y0
        keep = (dpred.abs() > 1e-12) & (dreal.abs() > 1e-12)
        rows.append({"model": m, "rmse": float(np.sqrt(E2[m].mean())), "mae": float((Y - F[m]).abs().mean()),
                     "r2_oos_vs_rw": 1.0 - float(E2[m].sum()) / sse_rw,
                     "hit_rate": float((np.sign(dpred[keep]) == np.sign(dreal[keep])).mean()) if keep.any() else np.nan,
                     "dm_t_vs_rw": np.nan if m == base else ev.dm_test(E2[base], E2[m], lags)["t"]})
    tab = pd.DataFrame(rows).set_index("model").join(ev.model_confidence_set(E2, size=0.10, block=max(h, block), reps=reps))
    meta = {"h": h, "n": int(len(sample)), "first": str(sample[0].date()), "last": str(sample[-1].date()),
            "rmse_rw": float(np.sqrt(sse_rw / len(sample)))}
    return tab, meta
