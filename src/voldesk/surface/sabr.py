"""SABR smiles: Hagan, Kumar, Lesniewski & Woodward (2002) lognormal implied-vol approximation.

    sigma_B(K) = alpha / ((F K)^((1-b)/2) [1 + (1-b)^2/24 L^2 + (1-b)^4/1920 L^4]) * z / x(z)
                 * [1 + ((1-b)^2/24 alpha^2/(F K)^(1-b) + rho b nu alpha / (4 (F K)^((1-b)/2))
                       + (2 - 3 rho^2)/24 nu^2) T],
    L = ln(F/K), z = (nu/alpha) (F K)^((1-b)/2) L, x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho)),
with z/x(z) -> 1 - rho z / 2 as z -> 0. Reference: QuantLib's sabrVolatility (tests).

It is an asymptotic formula: for long maturities or large nu it can imply negative densities in the
wings, which the arbitrage diagnostics measure for every fitted slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def sabr_iv(K, F, T, alpha, beta, rho, nu):
    K = np.asarray(K, dtype=float)
    L = np.log(F / K)
    ob = 1.0 - beta
    fkb = (F * K) ** (ob / 2)
    denom = fkb * (1 + ob ** 2 / 24 * L ** 2 + ob ** 4 / 1920 * L ** 4)
    z = nu / alpha * fkb * L
    with np.errstate(divide="ignore", invalid="ignore"):
        xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        zx = np.where(np.abs(z) < 1e-7, 1 - rho * z / 2, z / xz)
    corr = 1 + (ob ** 2 / 24 * alpha ** 2 / fkb ** 2 + rho * beta * nu * alpha / (4 * fkb) + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    return alpha / denom * zx * corr


@dataclass
class SABRFit:
    alpha: float
    beta: float
    rho: float
    nu: float
    T: float
    F: float
    rmse_vol: float
    within_bid_ask: float
    success: bool

    def iv(self, K):
        return sabr_iv(K, self.F, self.T, self.alpha, self.beta, self.rho, self.nu)


def fit_sabr_slice(K, F, T, iv_mid, *, iv_bid=None, iv_ask=None, beta: float = 1.0) -> SABRFit:
    """Fit alpha, rho, nu with beta fixed; residuals (model - mid) / max(bid-ask width, 0.001)."""
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv_mid, dtype=float)
    wt = 1.0 / np.maximum(np.asarray(iv_ask) - np.asarray(iv_bid), 1e-3) if iv_bid is not None else np.ones_like(K)
    wt = wt / np.sqrt(np.mean(wt ** 2))
    atm = float(np.interp(0.0, np.log(K / F), iv)) if np.any(K < F) and np.any(K >= F) else float(np.median(iv))
    a0 = atm * F ** (1 - beta)

    def resid(x):
        return (sabr_iv(K, F, T, x[0], beta, x[1], x[2]) - iv) * wt

    lo, hi = [1e-6, -0.999, 1e-4], [10 * a0 + 1.0, 0.999, 20.0]
    best = None
    for rho0, nu0 in ((-0.7, 1.0), (-0.3, 0.5), (-0.9, 3.0)):
        x0 = np.clip([a0, rho0, nu0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=2000, x_scale="jac")
        if np.all(np.isfinite(sol.fun)) and (best is None or sol.cost < best.cost):
            best = sol
    a, rho, nu = best.x
    model = sabr_iv(K, F, T, a, beta, rho, nu)
    inside = float(np.mean((model >= iv_bid) & (model <= iv_ask))) if iv_bid is not None else np.nan
    return SABRFit(a, beta, rho, nu, T, F, float(np.sqrt(np.mean((model - iv) ** 2))), inside, bool(best.success))
