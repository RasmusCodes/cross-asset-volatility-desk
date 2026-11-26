"""Rough volatility (advanced extension): the term structure of the at-the-money skew, and a rough Bergomi pricer.

ATM skew. psi(T) = |d sigma_BS / dk| at k = 0. Rough volatility models with Hurst exponent H imply
psi(T) ~ c T^(H - 1/2) for short maturities (Alos, Leon & Vives 2007; Fukasawa 2011; Bayer, Friz & Gatheral 2016),
while one-factor Markovian models such as Heston give a skew that flattens to a constant as T -> 0. Regressing
log psi on log T across a date's maturities therefore gives an implied H. From a raw SVI slice,
    sigma(k) = sqrt(w(k)/T),  d sigma/dk at 0 = w'(0) / (2 sqrt(w(0) T)).

Rough Bergomi (Bayer, Friz & Gatheral 2016), flat forward variance xi0:
    v_t = xi0 exp(eta sqrt(2H) int_0^t (t-s)^(H-1/2) dW_s - eta^2 t^(2H)/2),  dS/S = sqrt(v) dZ,  d<W,Z> = rho dt.
Simulated exactly on a time grid by a Cholesky factorization of the joint covariance of the Volterra process and Z
(the method of the original paper; O(n^3) once, fine for a few hundred steps).
"""
from __future__ import annotations

import numpy as np
from scipy.special import hyp2f1

from ..pricing.implied_vol import implied_vol_black


def svi_atm_skew(a, b, rho, m, s, T) -> float:
    w0 = a + b * (rho * (-m) + np.sqrt(m * m + s * s))
    dw0 = b * (rho + (-m) / np.sqrt(m * m + s * s))
    return float(abs(dw0) / (2.0 * np.sqrt(max(w0, 1e-12) * T)))


def power_law_exponent(T, psi, T_max: float = 0.5, T_min: float = 7 / 365) -> dict:
    T, psi = np.asarray(T, float), np.asarray(psi, float)
    ok = (T >= T_min) & (T <= T_max) & np.isfinite(psi) & (psi > 0)
    if ok.sum() < 3:
        return {"slope": np.nan, "H": np.nan, "n": int(ok.sum()), "r2": np.nan}
    x, y = np.log(T[ok]), np.log(psi[ok])
    A = np.column_stack([np.ones(ok.sum()), x])
    beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2) if ok.sum() > 2 else np.nan
    return {"slope": float(beta[1]), "H": float(beta[1] + 0.5), "n": int(ok.sum()), "r2": float(r2)}


def rbergomi_cov(t: np.ndarray, H: float, rho: float) -> np.ndarray:
    """Covariance of (V_t1..V_tn, Z_t1..Z_tn) with V_t = sqrt(2H) int_0^t (t-s)^(H-1/2) dW_s."""
    g = H - 0.5
    n = len(t)
    C = np.empty((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            u, v = min(t[i], t[j]), max(t[i], t[j])
            x = u / v
            C[i, j] = 2 * H * u ** (2 * H) * x ** (-g) / (1 + g) * hyp2f1(-g, 1, 2 + g, x) / 1.0 if u > 0 else 0.0
            C[n + i, n + j] = u
            # Cov(V_ti, Z_tj) = rho sqrt(2H)/(H+1/2) (t_i^(H+1/2) - (t_i - min(t_i,t_j))^(H+1/2))
            m = min(t[i], t[j])
            C[i, n + j] = rho * np.sqrt(2 * H) / (H + 0.5) * (t[i] ** (H + 0.5) - (t[i] - m) ** (H + 0.5))
            C[n + j, i] = C[i, n + j]
    np.fill_diagonal(C[:n, :n], t ** (2 * H))
    return C


def rbergomi_prices(T_list, k_list, H, eta, rho, xi0, n_paths=20000, steps_per_year=252, seed=0):
    """European OTM prices (forward 1, undiscounted) and Black implied vols at log-moneyness k for each maturity."""
    T_max = max(T_list)
    n = max(int(np.ceil(T_max * steps_per_year)), 8)
    t = np.linspace(T_max / n, T_max, n)
    C = rbergomi_cov(t, H, rho)
    L = np.linalg.cholesky(C + 1e-12 * np.eye(2 * n))
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_paths, 2 * n)) @ L.T
    V, Z = G[:, :n], G[:, n:]
    v = xi0 * np.exp(eta * V - 0.5 * eta ** 2 * t ** (2 * H))
    dZ = np.diff(np.concatenate([np.zeros((n_paths, 1)), Z], axis=1), axis=1)
    dt = np.diff(np.concatenate([[0.0], t]))
    v_prev = np.concatenate([np.full((n_paths, 1), xi0), v[:, :-1]], axis=1)
    logS = np.cumsum(np.sqrt(v_prev) * dZ - 0.5 * v_prev * dt, axis=1)
    out = {}
    for T in T_list:
        j = int(np.argmin(np.abs(t - T)))
        ST = np.exp(logS[:, j])
        ST = ST / ST.mean()                                  # martingale correction of the discretisation drift
        Ks = np.exp(np.asarray(k_list, float))
        cp = np.where(Ks >= 1.0, 1.0, -1.0)
        px = np.array([np.mean(np.maximum(c * (ST - K), 0.0)) for c, K in zip(cp, Ks)])
        out[float(t[j])] = implied_vol_black(px, cp, 1.0, Ks, float(t[j]), 1.0)
    return out
