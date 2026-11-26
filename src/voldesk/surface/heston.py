"""Heston (1993) stochastic volatility: European prices and calibration.

    dS/S = (r - q) dt + sqrt(v) dW1,   dv = kappa (theta - v) dt + sigma sqrt(v) dW2,   d<W1, W2> = rho dt.

Characteristic function of X_T = ln(S_T / F_T) in the "little trap" form (Albrecher, Mayer, Schoutens
& Tistaert 2007), which avoids the branch-cut discontinuity of Heston's original expression:
    a = kappa - rho sigma i u,  d = sqrt(a^2 + sigma^2 (i u + u^2)),  g = (a - d)/(a + d),
    phi(u) = exp( kappa theta / sigma^2 [(a - d) T - 2 ln((1 - g e^{-dT})/(1 - g))]
                  + v0 (a - d)/sigma^2 (1 - e^{-dT})/(1 - g e^{-dT}) ).
Prices by Lewis (2001): with k = ln(K/F),
    C = D [F - sqrt(F K)/pi * int_0^inf Re(e^{-i u k} phi(u - i/2)) / (u^2 + 1/4) du],
puts by put-call parity. The integral is computed by adaptive quadrature over [0, inf) for all strikes
of one maturity at once (scipy.integrate.quad_vec). Reference: QuantLib's AnalyticHestonEngine (tests).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import least_squares

from ..pricing.implied_vol import implied_vol_black


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / sigma^2 (>= 1: the variance process stays strictly positive)."""
        return 2 * self.kappa * self.theta / self.sigma ** 2


def heston_cf(u, T, p: HestonParams):
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    a = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(a * a + p.sigma ** 2 * (iu + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = p.kappa * p.theta / p.sigma ** 2 * ((a - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    Dv = (a - d) / p.sigma ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + Dv * p.v0)


def heston_price(cp, F, K, T, D, p: HestonParams, *, epsabs: float = 1e-13, epsrel: float = 1e-11):
    """European prices for strikes K (array) at one maturity T, forward F, discount factor D."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    cp = np.broadcast_to(np.asarray(cp, dtype=float), K.shape)
    k = np.log(K / F)

    def integrand(u):
        return np.real(np.exp(-1j * u * k) * heston_cf(u - 0.5j, T, p)) / (u * u + 0.25)

    integral, _ = quad_vec(integrand, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=2000)
    call = D * (F - np.sqrt(F * K) / np.pi * integral)
    return np.where(cp > 0, call, call - D * (F - K))


def heston_price_cos(cp, F, K, T, D, p: HestonParams, *, n: int = 1024, L: float = 12.0):
    """COS method (Fang & Oosterlee 2008) for European prices at one maturity, vectorised over strikes.

    Y = ln(S_T / F_T) has characteristic function heston_cf; with x = ln(F / K) the put value is
        P = D K sum'_j Re[phi(u_j) exp(i u_j (x - a))] V_j,   u_j = j pi / (b - a),
        V_j = 2/(b - a) (psi_j(a, 0) - chi_j(a, 0))
    (the prime halves the j = 0 term); calls follow from put-call parity, which is exact here. The truncation range
    [a, b] is +-L standard deviations of the mean integrated variance, widened by (1 + sigma) for heavy tails.
    Validated against the adaptive Lewis integral (heston_price) in tests/test_surface_models.py.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    cp = np.broadcast_to(np.asarray(cp, dtype=float), K.shape)
    x = np.log(F / K)
    ekt = np.exp(-p.kappa * T)
    mean_var = p.theta * T + (p.v0 - p.theta) * (1 - ekt) / p.kappa
    sd = np.sqrt(max(mean_var, 1e-12)) * (1.0 + p.sigma)
    a, b = -0.5 * mean_var - L * sd, -0.5 * mean_var + L * sd
    j = np.arange(n)
    u = j * np.pi / (b - a)
    phi = heston_cf(u, T, p)
    # chi and psi on [a, 0] (put payoff K (1 - e^y)^+ in y = ln(S_T / K) + ... coordinates)
    c, d = a, 0.0
    w = u
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
               + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)) / (1 + w * w)
        psi = np.where(j == 0, d - c, (np.sin(w * (d - a)) - np.sin(w * (c - a))) / np.where(j == 0, 1.0, w))
    Vj = 2.0 / (b - a) * (psi - chi)
    Vj[0] *= 0.5
    terms = np.real(phi[None, :] * np.exp(1j * np.outer(x - a, u)))
    put = D * K * (terms @ Vj)
    put = np.maximum(put, np.maximum(D * (K - F), 0.0))
    return np.where(cp > 0, put + D * (F - K), put)


def heston_iv(F, K, T, D, p: HestonParams, method: str = "cos"):
    """Black implied vols of Heston prices (OTM side used for the inversion). method: 'cos' (fast) or 'quad'."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    cp = np.where(K >= F, 1.0, -1.0)
    px = heston_price_cos(cp, F, K, T, D, p) if method == "cos" else heston_price(cp, F, K, T, D, p)
    return implied_vol_black(px, cp, F, K, T, D)


@dataclass
class HestonFit:
    params: HestonParams
    rmse_vol: float
    within_bid_ask: float
    per_slice_rmse: np.ndarray
    success: bool
    n_quotes: int


def fit_heston(slices, *, starts=None, max_nfev: int = 300) -> HestonFit:
    """Calibrate (v0, kappa, theta, sigma, rho) to slices' mid vols.

    Residuals are (model - mid vol) / max(bid-ask width, 0.001), each slice weighted equally. The Feller
    condition is reported, not imposed. Several starting points; the lowest cost wins.
    """
    sl = sorted(slices, key=lambda x: x.T)
    data = []
    for s in sl:
        wt = 1.0 / np.maximum(s.iv_spread, 1e-3)
        wt = wt / np.sqrt(np.mean(wt ** 2)) / np.sqrt(s.K.size)
        data.append((s, wt))
    atm_short = float(np.interp(0.0, sl[0].k, sl[0].iv_mid)) if sl else 0.2
    atm_long = float(np.interp(0.0, sl[-1].k, sl[-1].iv_mid)) if sl else 0.2

    def resid(x):
        p = HestonParams(*x)
        out = []
        for s, wt in data:
            iv = heston_iv(s.F, s.K, s.T, s.D, p)
            out.append(np.where(np.isfinite(iv), (iv - s.iv_mid) * wt, 1.0))
        return np.concatenate(out)

    lo = [1e-4, 1e-2, 1e-4, 1e-2, -0.999]
    hi = [1.0, 20.0, 1.0, 5.0, 0.999]
    if starts is None:
        starts = [(atm_short ** 2, 2.0, atm_long ** 2, 1.0, -0.7), (atm_short ** 2, 5.0, atm_long ** 2, 2.0, -0.9),
                  (atm_short ** 2, 1.0, atm_long ** 2 * 1.5, 0.5, -0.5)]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=max_nfev, x_scale="jac", diff_step=1e-4)
        if best is None or sol.cost < best.cost:
            best = sol
    p = HestonParams(*best.x)
    errs, inside, per = [], [], []
    for s, _ in data:
        iv = heston_iv(s.F, s.K, s.T, s.D, p)
        e = iv - s.iv_mid
        errs.append(e)
        per.append(np.sqrt(np.nanmean(e ** 2)))
        inside.append((iv >= s.iv_bid) & (iv <= s.iv_ask))
    errs = np.concatenate(errs)
    return HestonFit(p, float(np.sqrt(np.nanmean(errs ** 2))), float(np.mean(np.concatenate(inside))), np.array(per),
                     bool(best.success), int(errs.size))
