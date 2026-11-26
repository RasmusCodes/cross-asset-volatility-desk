"""Dupire local volatility from a total-variance surface, and a forward-PDE repricing check.

Dupire in total-variance form (Gatheral 2006, eq. 1.10), k = ln(K/F_T):
    sigma_loc^2(k, T) = (dw/dT) / g(k, T),
where g is the Gatheral-Jacquier butterfly function of the slice at T (voldesk.surface.svi.g_function).
So the local variance is non-negative exactly where the surface is free of calendar (dw/dT >= 0) and
butterfly (g > 0) arbitrage. Negative or non-finite values are clipped to 0 and counted.

Repricing check. Undiscounted, forward-normalised calls c(y, T) = C / (D F_T) as functions of
y = ln(K/F_T) satisfy the forward (Dupire) equation
    dc/dT = 1/2 sigma_loc^2(y, T) (c_yy - c_y),   c(y, 0) = (1 - e^y)^+,
with no drift term because both the strike variable and the price are measured against the forward.
It is solved by Crank-Nicolson with a Rannacher start (four implicit half-steps), on a uniform y grid with
Dirichlet boundaries c(-Y) = 1 - e^-Y and c(Y) = 0, local variance frozen at mid-step times, and time
levels graded towards T = 0 (t_i = T_max (i/n)^2) where the payoff kink lives. On an arbitrage-free
surface a correct Dupire implementation reproduces the surface's own call prices up to discretisation
error: that round trip is the validation.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from .svi import g_function


def local_variance(surface, k, T, *, return_clipped: bool = False):
    """Dupire local variance at log-moneyness k (array) and maturity T (scalar)."""
    k = np.asarray(k, dtype=float)
    w = surface.w(k, T)
    g = g_function(k, w, surface.dw_dk(k, T), surface.d2w_dk2(k, T))
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = surface.dw_dT(k, T) / g
    bad = ~np.isfinite(lv) | (lv < 0)
    lv = np.where(bad, 0.0, lv)
    return (lv, int(bad.sum())) if return_clipped else lv


def local_vol(surface, k, T):
    return np.sqrt(local_variance(surface, k, T))


def forward_pde(surface, T_out, *, y_max: float | None = None, n_y: int = 801, n_t: int = 400,
                grading: float = 2.0, rannacher_steps: int = 2):
    """Solve the forward equation with the Dupire local variance of `surface` up to max(T_out).
    Returns (y grid, {T: c(y, T)}, n_clipped_total)."""
    T_out = np.sort(np.unique(np.asarray(T_out, dtype=float)))
    if y_max is None:
        k_probe = np.linspace(-1.0, 1.0, 21)
        y_max = float(np.clip(6.0 * np.sqrt(np.max(surface.w(k_probe, float(T_out[-1])))) + 0.5, 1.0, 5.0))
    return _forward(lambda y, t: local_variance(surface, y, t, return_clipped=True), T_out, y_max, n_y, n_t,
                    grading, rannacher_steps)


def forward_pde_local_var(local_var, T_out, *, y_max: float, n_y: int = 2001, n_t: int = 2000,
                          grading: float = 2.0, rannacher_steps: int = 2):
    """Forward equation with a given local-variance function local_var(y, t) (y array, t scalar), e.g. a
    calibrated Andreasen-Huge local vol squared. Same scheme as `forward_pde`; T_out are exact time nodes,
    so a local vol that is piecewise constant between those maturities is never straddled by a step.
    Returns (y grid, {T: c(y, T)})."""
    T_out = np.sort(np.unique(np.asarray(T_out, dtype=float)))
    y, out, _ = _forward(lambda yy, t: (np.asarray(local_var(yy, t), dtype=float), 0), T_out, y_max, n_y, n_t,
                         grading, rannacher_steps)
    return y, out


def _forward(lv_fn, T_out, y_max, n_y, n_t, grading, rannacher_steps):
    T_max = float(T_out[-1])
    y = np.linspace(-y_max, y_max, n_y)
    h = y[1] - y[0]
    base = T_max * (np.arange(n_t + 1) / n_t) ** grading
    times = np.unique(np.concatenate([base, T_out]))
    c = np.maximum(1.0 - np.exp(y), 0.0)
    out, clipped = {}, 0
    for n in range(times.size - 1):
        t0, t1 = times[n], times[n + 1]
        dt = t1 - t0
        sub = [(dt / 2, 1.0), (dt / 2, 1.0)] if n < rannacher_steps else [(dt, 0.5)]
        t = t0
        for step, theta in sub:
            lv, nc = lv_fn(y, max(t + step / 2, 1e-10))
            clipped += nc
            half = 0.5 * lv
            lo = half * (1 / h ** 2 + 1 / (2 * h))        # coefficient of c_{i-1}
            di = half * (-2 / h ** 2)
            up = half * (1 / h ** 2 - 1 / (2 * h))        # coefficient of c_{i+1}
            Lc = di * c
            Lc[1:] += lo[1:] * c[:-1]
            Lc[:-1] += up[:-1] * c[1:]
            rhs = c + (1 - theta) * step * Lc
            ab = np.zeros((3, y.size))
            ab[0, 1:] = -theta * step * up[:-1]
            ab[1, :] = 1 - theta * step * di
            ab[2, :-1] = -theta * step * lo[1:]
            ab[1, 0], ab[0, 1] = 1.0, 0.0                 # Dirichlet rows
            ab[1, -1], ab[2, -2] = 1.0, 0.0
            rhs[0], rhs[-1] = 1.0 - np.exp(y[0]), 0.0
            c = solve_banded((1, 1), ab, rhs)
            t += step
        if np.any(np.isclose(t1, T_out, rtol=0, atol=1e-14)):
            out[float(T_out[np.argmin(np.abs(T_out - t1))])] = c.copy()
    return y, out, clipped
