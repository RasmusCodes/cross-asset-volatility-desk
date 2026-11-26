"""Volatility forecasting models. At the close of day t each produces a forecast of the variance over days t+1 .. t+h
(squared log-return units, so comparable with realized.forward_sum of squared returns), from data up to day t only.

  historical vol   h x mean of the last n squared returns
  EWMA             RiskMetrics: s(t+1) = lam s(t) + (1 - lam) r(t)^2, forecast h x s(t+1)
  GARCH / GJR      constant mean, s(t) = omega + (alpha + gamma 1[e(t-1) < 0]) e(t-1)^2 + beta s(t-1), Gaussian
                   quasi-likelihood; arch's conventions (returns in percent; backcast = 0.94-weighted mean of the first
                   75 squared residuals, computed once from the residuals about the sample mean); forecast = sum of the
                   expected daily variances, which revert to omega / (1 - persistence) at rate persistence =
                   alpha + gamma/2 + beta
  linear           the h-day mean daily variance on regressors (HAR components, implied variance): weighted least
                   squares in levels, or OLS in logs with Duan's smearing retransformation (see fit_linear)
  gradient boost   scikit-learn HistGradientBoostingRegressor with the gamma deviance loss, which for a variance
                   forecast is the QLIKE loss up to terms that do not depend on the forecast

Implied variance as a forecast (Heston). In the one-factor Heston model both the squared VIX (expected 30-day variance
under the pricing measure) and the expected future integrated variance under the physical measure are affine in the
instantaneous variance v(t). So E[RV(t, h)] = a + b VIX(t)^2 exactly, and a regression of realized variance on VIX^2
is Heston's forecast with the parameters (including the variance risk premium) absorbed in a and b.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


# ---------------------------------------------------------------- historical volatility and EWMA
def hv_forecast(r2: pd.Series, window: int, h: int) -> pd.Series:
    return h * r2.rolling(window, min_periods=window).mean()


def backcast(x) -> float:
    """0.94-weighted mean of the first 75 values (arch's variance backcast)."""
    x = np.asarray(x, dtype=float)
    tau = min(75, x.size)
    w = 0.94 ** np.arange(tau)
    return float(np.sum(x[:tau] * w / w.sum()))


def ewma_next(r2: pd.Series, lam: float = 0.94) -> pd.Series:
    """s(t+1 | t) aligned at t: the RiskMetrics recursion started from the backcast of the first squared returns."""
    v = r2.to_numpy(dtype=float)
    first = int(np.flatnonzero(np.isfinite(v))[0])
    out = np.full(v.size, np.nan)
    x = v[first:]
    x = np.where(np.isfinite(x), x, 0.0)
    out[first:] = lfilter([1.0 - lam], [1.0, -lam], x, zi=[lam * backcast(x)])[0]
    return pd.Series(out, index=r2.index)


# ---------------------------------------------------------------- GARCH(1,1) and GJR-GARCH(1,1,1)
@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    loglik: float
    backcast: float
    success: bool
    n: int
    scale: float = 100.0              # returns were multiplied by this before fitting (percent)

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def unconditional(self) -> float:
        return self.omega / (1.0 - self.persistence)


def garch_sigma2(e: np.ndarray, omega, alpha, gamma, beta, bc: float) -> np.ndarray:
    """Conditional variances s(t), t = 0 .. n-1, given residuals e; s(0) uses the backcast for every lagged term."""
    u = np.empty_like(e)
    u[0] = omega + (alpha + 0.5 * gamma + beta) * bc
    u[1:] = omega + (alpha + gamma * (e[:-1] < 0)) * e[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], u)


def _nll(p, y, bc, asym):
    mu, omega, alpha, gamma, beta = (p[0], p[1], p[2], p[3], p[4]) if asym else (p[0], p[1], p[2], 0.0, p[3])
    e = y - mu
    s2 = garch_sigma2(e, omega, alpha, gamma, beta, bc)
    if not np.all(s2 > 0):
        return 1e12
    return 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2))


def fit_garch(returns, asym: bool = False, scale: float = 100.0) -> GarchFit:
    """Gaussian QML fit of GARCH(1,1) (asym=False) or GJR-GARCH(1,1,1) (asym=True) with a constant mean."""
    y = np.asarray(returns, dtype=float) * scale
    y = y[np.isfinite(y)]
    v = float(np.var(y))
    bc = backcast((y - y.mean()) ** 2)
    best = None
    for a in (0.02, 0.05, 0.1):                              # small grid of starting values, then SLSQP
        for pers in (0.9, 0.97, 0.99):
            g = (0.1 if asym else 0.0)
            b = pers - a - 0.5 * g
            if b <= 0:
                continue
            x0 = [y.mean(), v * (1 - pers), a] + ([g] if asym else []) + [b]
            f = _nll(x0, y, bc, asym)
            if best is None or f < best[0]:
                best = (f, x0)
    x0 = best[1]
    bounds = [(-10 * np.sqrt(v), 10 * np.sqrt(v)), (1e-10 * v, 10 * v), (0.0, 1.0)] + ([(-1.0, 2.0)] if asym else []) + [(0.0, 1.0)]
    cons = [{"type": "ineq", "fun": (lambda p: 1.0 - 1e-6 - (p[2] + 0.5 * p[3] + p[4])) if asym else
             (lambda p: 1.0 - 1e-6 - (p[2] + p[3]))}]
    if asym:
        cons.append({"type": "ineq", "fun": lambda p: p[2] + p[3]})           # alpha + gamma >= 0
    res = minimize(_nll, x0, args=(y, bc, asym), method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    p = res.x
    mu, omega, alpha, gamma, beta = (p[0], p[1], p[2], p[3], p[4]) if asym else (p[0], p[1], p[2], 0.0, p[3])
    return GarchFit(float(mu), float(omega), float(alpha), float(gamma), float(beta), -float(res.fun), bc,
                    bool(res.success), int(y.size), scale)


def garch_next(fit: GarchFit, returns: pd.Series) -> pd.Series:
    """s(t+1 | t) aligned at t, in squared log-return units, filtering the whole series with the fitted parameters
    (the recursion starts at the first finite return with the fit's backcast)."""
    r = returns.to_numpy(dtype=float)
    first = int(np.flatnonzero(np.isfinite(r))[0])
    e = (np.where(np.isfinite(r[first:]), r[first:], 0.0) * fit.scale) - fit.mu
    s2 = garch_sigma2(e, fit.omega, fit.alpha, fit.gamma, fit.beta, fit.backcast)
    nxt = fit.omega + (fit.alpha + fit.gamma * (e < 0)) * e ** 2 + fit.beta * s2
    out = np.full(r.size, np.nan)
    out[first:] = nxt / fit.scale ** 2
    return pd.Series(out, index=returns.index)


def garch_h_day(fit: GarchFit, s_next, h: int):
    """Sum over k = 1..h of E[r(t+k)^2] = E[s(t+k)] + mu^2 (squared log-return units) from s(t+1 | t)."""
    p = fit.persistence
    ubar = fit.unconditional / fit.scale ** 2
    s_next = np.asarray(s_next, dtype=float)
    decay = h if abs(1.0 - p) < 1e-12 else (1.0 - p ** h) / (1.0 - p)
    return h * ubar + (s_next - ubar) * decay + h * (fit.mu / fit.scale) ** 2


# ---------------------------------------------------------------- HAR and other linear forecasts
def har_components(d: pd.Series) -> pd.DataFrame:
    """Corsi's HAR regressors: the day's value, and its means over the last 5 and 22 days."""
    return pd.DataFrame({"d": d, "w": d.rolling(5, min_periods=5).mean(), "m": d.rolling(22, min_periods=22).mean()})


@dataclass
class LinearFit:
    beta: np.ndarray
    smear: float                       # log models: Duan's smearing factor, the mean of exp(residual)
    log: bool
    floor: float                       # smallest forecast allowed (a level model can predict a negative variance)


def fit_linear(X: np.ndarray, y: np.ndarray, log: bool = False, wls: bool = True, floor: float | None = None) -> LinearFit:
    """Linear forecast of y (h-day mean daily variance) on X with an intercept.

    Levels: feasible weighted least squares (Patton and Sheppard's remedy for HAR in levels). Variance errors scale
    with the variance level, so plain OLS is dominated by crisis days and can produce a negative intercept that pushes
    calm-market forecasts towards zero. A first OLS pass gives fitted levels f0, then WLS with weights 1 / f0^2. A level
    forecast is never allowed below `floor` (under QLIKE a near-zero variance forecast costs without limit). The
    forecasting script passes the lowest 22-day mean daily variance in the training window, a bound on a plausible
    expected variance that does not depend on the horizon; the default, 1% of the mean target, is only a numerical
    guard.
    Logs: OLS of log y on log X; the forecast is exp(fitted) times Duan's smearing factor mean(exp(residual)), which
    needs no normality (log squared returns are far from normal, and the lognormal correction exp(s^2 / 2) would
    overstate one-day forecasts several times)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if log:
        A = np.column_stack([np.ones(len(y)), np.log(X)])
        beta, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        return LinearFit(beta, float(np.mean(np.exp(np.log(y) - A @ beta))), True, 0.0)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    floor = 0.01 * float(np.mean(y)) if floor is None else float(floor)
    if wls:
        f0 = np.maximum(A @ beta, 0.01 * float(np.mean(y)))            # first-pass levels, bounded for the weights
        w = 1.0 / f0
        beta, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return LinearFit(beta, 1.0, False, floor)


def predict_linear(fit: LinearFit, X: np.ndarray) -> tuple[np.ndarray, int]:
    """Forecast of the h-day mean daily variance, and how many forecasts hit the floor (levels only)."""
    X = np.asarray(X, dtype=float)
    if fit.log:
        z = np.column_stack([np.ones(len(X)), np.log(X)]) @ fit.beta
        return np.exp(z) * fit.smear, 0
    yhat = np.column_stack([np.ones(len(X)), X]) @ fit.beta
    low = yhat < fit.floor
    return np.where(low, fit.floor, yhat), int(low.sum())


# ---------------------------------------------------------------- gradient-boosted trees
GBM_PARAMS = {"loss": "gamma", "learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 100,
              "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}


def fit_gbm(X, y, **overrides):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(**{**GBM_PARAMS, **overrides})
    return m.fit(np.asarray(X, dtype=float), np.maximum(np.asarray(y, dtype=float), 1e-12))
