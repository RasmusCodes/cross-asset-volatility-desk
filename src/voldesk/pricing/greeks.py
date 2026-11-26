"""Closed-form Black-Scholes-Merton Greeks (continuous dividend yield q).

All Greeks are plain partial derivatives of the price, in "raw" units:

=========  ==========================  ======================================================
name       definition                  formula (th = +1 call, -1 put; b = r - q; v = sigma sqrt T)
=========  ==========================  ======================================================
delta      dV/dS                       th e^{-qT} N(th d1)
gamma      d2V/dS2                     e^{-qT} phi(d1) / (S v)
vega       dV/dsigma  (per 1.00 vol)   S e^{-qT} phi(d1) sqrt T
theta      dV/dt = -dV/dT (per year)   -S e^{-qT} phi(d1) sigma/(2 sqrt T) - th r K e^{-rT} N(th d2)
                                       + th q S e^{-qT} N(th d1)
rho        dV/dr  (per 1.00 rate)      th K T e^{-rT} N(th d2)
psi        dV/dq  (dividend rho)       -th S T e^{-qT} N(th d1)
vanna      d2V/dS dsigma               -e^{-qT} phi(d1) d2 / sigma
volga      d2V/dsigma2 (vomma)         vega d1 d2 / sigma
charm      d delta / dt = -d2V/dS dT   th q e^{-qT} N(th d1) - e^{-qT} phi(d1) (2 b T - d2 v)/(2 T v)
speed      d3V/dS3                     -gamma (d1/v + 1) / S
color      d gamma / dt = -d3V/dS2 dT  gamma [q + 1/(2T) + d1 (2 b T - d2 v)/(2 T v)]
=========  ==========================  ======================================================

t is calendar time passing (T = time to expiry shrinks as t grows), so theta, charm and color
are "per year of time passing". Sign conventions for charm and color differ between sources
(some quote dX/dT); these are validated against computer-algebra derivatives in the tests.
``to_trader_units`` rescales to per-vol-point / per-day / per-1%-rate quantities.

Degenerate inputs (T = 0 or sigma = 0): price and the first-order Greeks take their limits
(delta = th e^{-qT} 1{th(F-K) > 0}, theta/rho/psi likewise on the in-the-money-forward side);
gamma, vega and the higher-order Greeks are 0 away from the forward and NaN exactly at it.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .bsm import bsm_price, cp_sign, norm_pdf

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho", "psi",
               "vanna", "volga", "charm", "speed", "color")

# Multipliers from raw units to common desk units.
TRADER_UNITS = {
    "price": (1.0, "currency"),
    "delta": (1.0, "per 1 unit of S"),
    "gamma": (1.0, "per 1 unit of S, squared"),
    "vega": (0.01, "per 1 vol point"),
    "theta": (1.0 / 365.0, "per calendar day"),
    "rho": (0.01, "per 1% rate"),
    "psi": (0.01, "per 1% dividend yield"),
    "vanna": (0.01, "delta change per 1 vol point"),
    "volga": (1e-4, "vega (per vol pt) change per 1 vol point"),
    "charm": (1.0 / 365.0, "delta change per calendar day"),
    "speed": (1.0, "gamma change per 1 unit of S"),
    "color": (1.0 / 365.0, "gamma change per calendar day"),
}


def bsm_greeks(cp, S, K, T, r, sigma, q=0.0) -> dict[str, np.ndarray]:
    """Price and all Greeks in raw units (see module docstring). Broadcasts over arrays."""
    th = cp_sign(cp)
    S, K, T, r, sigma, q, th = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q, th)))
    bad = ~(np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(sigma)
            & np.isfinite(q) & (S > 0) & (K > 0) & (T >= 0) & (sigma >= 0))
    out = {"price": np.asarray(bsm_price(th, S, K, T, r, sigma, q), dtype=float)}

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sqT = np.sqrt(np.maximum(T, 0.0))
        v = sigma * sqT
        b = r - q
        x = np.log(S / K) + b * T                 # ln(F/K)
        d1 = x / v + 0.5 * v
        d2 = d1 - v
        Dq = np.exp(-q * T)
        Dr = np.exp(-r * T)
        pdf = norm_pdf(d1)
        Nd1 = ndtr(th * d1)
        Nd2 = ndtr(th * d2)
        dd1_dT = (2.0 * b * T - d2 * v) / (2.0 * T * v)     # d(d1)/dT at fixed S, K, r, q, sigma

        g = {}
        g["delta"] = th * Dq * Nd1
        g["gamma"] = Dq * pdf / (S * v)
        g["vega"] = S * Dq * pdf * sqT
        g["theta"] = -S * Dq * pdf * sigma / (2.0 * sqT) - th * r * K * Dr * Nd2 + th * q * S * Dq * Nd1
        g["rho"] = th * K * T * Dr * Nd2
        g["psi"] = -th * S * T * Dq * Nd1
        g["vanna"] = -Dq * pdf * d2 / sigma
        g["volga"] = g["vega"] * d1 * d2 / sigma
        g["charm"] = th * q * Dq * Nd1 - Dq * pdf * dd1_dT
        g["speed"] = -g["gamma"] * (d1 / v + 1.0) / S
        g["color"] = g["gamma"] * (q + 0.5 / T + d1 * dd1_dT)

        # Limits for v == 0 (T == 0 or sigma == 0).
        deg = (v == 0) & ~bad
        if np.any(deg):
            itm = (th * x > 0) & deg
            atf = (x == 0) & deg                                 # exactly at the forward
            ind = np.where(itm, 1.0, np.where(atf, 0.5, 0.0))
            g["delta"] = np.where(deg, th * Dq * ind, g["delta"])
            g["theta"] = np.where(deg, np.where(atf, np.nan, np.where(itm, th * (q * S * Dq - r * K * Dr), 0.0)), g["theta"])
            g["rho"] = np.where(deg, np.where(itm, th * K * T * Dr, 0.0), g["rho"])
            g["psi"] = np.where(deg, np.where(itm, -th * S * T * Dq, 0.0), g["psi"])
            for k in ("gamma", "vega", "vanna", "volga", "charm", "speed", "color"):
                g[k] = np.where(deg, np.where(atf, np.nan, 0.0), g[k])

    for k in GREEK_NAMES:
        out[k] = np.where(bad, np.nan, g[k])[()]
    out["price"] = out["price"][()]
    return out


def to_trader_units(greeks: dict) -> dict:
    """Rescale raw Greeks to desk units (vega per vol point, theta per calendar day, ...)."""
    return {k: (v * TRADER_UNITS[k][0] if k in TRADER_UNITS else v) for k, v in greeks.items()}
