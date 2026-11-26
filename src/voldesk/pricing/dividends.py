"""Discrete dividends: schedules, the escrowed-dividend European price, and model conventions.

Two models of a cash dividend D paid at ex-dividend time t_d are supported by the pricers in
``voldesk.trees``, ``voldesk.fd`` and ``voldesk.montecarlo``:

``"spot"`` (piecewise lognormal; QuantLib's ``CashDividendModel.Spot``)
    The spot follows geometric Brownian motion with volatility sigma between ex-dates and jumps
    S -> S - D at t_d (floored at 0). The European price has no closed form. Lattices and the PDE
    impose the jump condition V(S, t_d-) = V(S - D, t_d+); Monte Carlo simulates the jump.

``"escrowed"`` (QuantLib's ``CashDividendModel.Escrowed``; Hull's textbook model)
    S = S* + E_t, with S* lognormal with volatility sigma and E_t the escrowed amount of the
    dividends still to be paid before expiry, E_t = sum D_i exp(-b (t_i - t)), b = r - q. The
    European price is the BSM price on S*_0 = S_0 - E_0 (``escrowed_european_price``). American
    pricers build the lattice / PDE on S* and test early exercise against S* + E_t.
    Discounting at the cost of carry b (QuantLib's AnalyticDividendEuropeanEngine does the same)
    gives both models the same forward, S e^{bT} - sum D_i e^{b (T - t_i)}; with q = 0 it is the
    plain present value at r.

For the same sigma the two models give different prices: the escrowed model's S has a lower
instantaneous volatility (sigma S*/S). Neither model is "the" market model; the choice is a
convention to be stated with every number.

A schedule may also hold proportional dividends (S -> S (1 - f) at t_d, spot model only). A
proportional dividend is the discrete version of a continuous yield: n proportional dividends
with (1 - f)^n = exp(-q T) leave the terminal distribution exactly lognormal with yield q, which
gives an exact European check of the jump machinery and the continuous-yield limit for American
options.

Timing convention: times are year fractions from the valuation instant (ACT/365F in the data
layer). A dividend counts for an option expiring at T if 0 < t_d < T. Ex-dividend means the drop
happens at t_d, so an American holder can exercise just before it (cum-dividend).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bsm import bsm_price


@dataclass(frozen=True, init=False)
class DividendSchedule:
    """Ex-dividend times (years from valuation) with a cash amount and a proportional fraction each.

    At an ex time the spot jumps S -> max(S (1 - fraction) - cash, 0). Events are kept sorted by
    time. Construct with ``DividendSchedule(times, cash=..., fraction=...)``.
    """

    times: np.ndarray
    cash: np.ndarray
    fraction: np.ndarray

    def __init__(self, times, cash=None, fraction=None):
        t = np.atleast_1d(np.asarray(times, dtype=float))
        c = np.zeros_like(t) if cash is None else np.broadcast_to(np.asarray(cash, dtype=float), t.shape).copy()
        f = np.zeros_like(t) if fraction is None else np.broadcast_to(np.asarray(fraction, dtype=float), t.shape).copy()
        if t.ndim != 1:
            raise ValueError("dividend times must be one-dimensional")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(c)) and np.all(np.isfinite(f))):
            raise ValueError("dividend times and amounts must be finite")
        if np.any(c < 0) or np.any((f < 0) | (f >= 1)):
            raise ValueError("cash amounts must be >= 0 and fractions in [0, 1)")
        order = np.argsort(t, kind="stable")
        object.__setattr__(self, "times", t[order])
        object.__setattr__(self, "cash", c[order])
        object.__setattr__(self, "fraction", f[order])

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def has_proportional(self) -> bool:
        return bool(np.any(self.fraction > 0))

    @property
    def has_cash(self) -> bool:
        return bool(np.any(self.cash > 0))

    def window(self, T: float, t0: float = 0.0) -> "DividendSchedule":
        """Events with t0 < t < T (the dividends that affect an option expiring at T)."""
        m = (self.times > t0) & (self.times < T)
        return DividendSchedule(self.times[m], self.cash[m], self.fraction[m])

    def pv_cash(self, T, r, t0=0.0):
        """Value at t0 of the cash dividends with t0 < t_i < T, discounted at the flat rate ``r``.

        Pass r - q for the escrowed amount (see module docstring). Broadcasts over T, r and t0.
        """
        T, r, t0 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (T, r, t0)))
        out = np.zeros(T.shape)
        for ti, ci in zip(self.times, self.cash):
            if ci == 0.0:
                continue
            live = (ti > t0) & (ti < T)
            out = out + np.where(live, ci * np.exp(-r * (ti - t0)), 0.0)
        return out[()]


EMPTY = DividendSchedule([])


def as_schedule(dividends) -> DividendSchedule:
    """Accept None, a DividendSchedule, or an iterable of (time, cash) pairs."""
    if dividends is None:
        return EMPTY
    if isinstance(dividends, DividendSchedule):
        return dividends
    pairs = list(dividends)
    if not pairs:
        return EMPTY
    t, c = zip(*pairs)
    return DividendSchedule(t, cash=c)


def escrowed_spot(S, T, r, dividends, q=0.0) -> np.ndarray:
    """S* = S - sum D_i exp(-(r - q) t_i) over cash dividends with 0 < t_i < T (the lognormal part)."""
    sched = as_schedule(dividends)
    if sched.has_proportional:
        raise ValueError("the escrowed model is defined for cash dividends only")
    b = np.asarray(r, dtype=float) - np.asarray(q, dtype=float)
    return (np.asarray(S, dtype=float) - sched.pv_cash(T, b))[()]


def escrowed_european_price(cp, S, K, T, r, sigma, q=0.0, dividends=None):
    """European price under the escrowed-dividend model: BSM on S* = S - E_0 (dividends before T).

    NaN where S* <= 0 (dividends worth more than the stock). Reference: QuantLib's
    AnalyticDividendEuropeanEngine implements the same model and convention.
    """
    s_star = escrowed_spot(S, T, r, dividends, q)
    with np.errstate(invalid="ignore"):
        return bsm_price(cp, np.where(s_star > 0, s_star, np.nan), K, T, r, sigma, q)


def proportional_european_price(cp, S, K, T, r, sigma, q=0.0, dividends=None):
    """Exact European price when every dividend before T is proportional (spot model).

    S_T is lognormal with forward S exp((r - q) T) prod(1 - f_i), so the price is BSM on
    S prod(1 - f_i). Raises if the schedule holds cash amounts (no closed form then).
    """
    sched = as_schedule(dividends)
    if sched.has_cash:
        raise ValueError("closed form only for purely proportional dividends")
    T = np.asarray(T, dtype=float)
    factor = np.ones(T.shape)
    for ti, fi in zip(sched.times, sched.fraction):
        factor = factor * np.where((ti > 0) & (ti < T), 1.0 - fi, 1.0)
    return bsm_price(cp, np.asarray(S, dtype=float) * factor, K, T, r, sigma, q)
