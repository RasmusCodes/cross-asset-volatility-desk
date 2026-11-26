"""Positions in European index options and the index, valued off a Market; Greeks; P&L attribution.

Legs:
  OptionLeg  qty contracts (+ long, - short) of a European option with absolute expiry T (the Market's clock),
             multiplier 100 (SPX). A leg built from a listed contract keeps its root and expiration date.
  IndexLeg   qty units of the index held as a zero-cost delta-one hedge (a futures-like exposure: its value is
             qty (S - entry)); financing over a day is ignored.

Valuation. model value: Black-76 off the Market's forwards, discount factors and surface. mid value: the listed
contracts' mid quotes where the Market has them (stored closes). execution value: what entering costs at the quotes
(buy at the ask, sell at the bid). An option whose time to expiry is <= 0 is worth its intrinsic value at the Market's
spot (a stand-in for the settlement value).

Greeks (``greeks``) are the Black-Scholes-Merton Greeks of each leg at its own implied vol, with the rate and carry
implied by the leg's discount factor and forward: sticky-strike sensitivities, summed with signs and multipliers.
``risk`` converts them to desk units: P&L for a +1% index move (first order), the second-order P&L of a 1% move,
per vol point, per calendar day.

Attribution (``attribute``) splits a leg's value change between two Markets into a second-order Taylor expansion in
its own forward F, implied vol s (at its strike), time to expiry tau and discount factor D:
  delta dV/dF dF,  gamma 1/2 d2V/dF2 dF^2,  vega dV/ds ds,  vanna d2V/dF ds dF ds,  volga 1/2 d2V/ds2 ds^2,
  theta dV/dtau dtau (at fixed F and D),  rates dV/dD dD,  and the residual (third and higher orders).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..pricing.bsm import black_price, cp_sign, norm_pdf
from ..pricing.greeks import bsm_greeks
from .market import Market

TERMS = ("delta", "gamma", "vega", "vanna", "volga", "theta", "rates")


@dataclass(frozen=True)
class OptionLeg:
    cp: int
    K: float
    T: float
    qty: float
    mult: float = 100.0
    root: str = ""
    expiration: str = ""

    @property
    def label(self) -> str:
        side = "C" if self.cp > 0 else "P"
        exp = self.expiration or f"T={self.T:.4f}"
        return f"{self.qty:+g} {self.root or 'opt'} {exp} {self.K:g}{side}"


@dataclass(frozen=True)
class IndexLeg:
    qty: float
    entry: float

    @property
    def label(self) -> str:
        return f"{self.qty:+.4g} index @ {self.entry:.2f}"


@dataclass
class Position:
    name: str
    legs: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def options(self) -> list:
        return [g for g in self.legs if isinstance(g, OptionLeg)]

    def with_legs(self, extra: list, name: str | None = None) -> "Position":
        return Position(name or self.name, list(self.legs) + list(extra), dict(self.meta))

    def scaled(self, k: float, name: str | None = None) -> "Position":
        legs = [OptionLeg(g.cp, g.K, g.T, g.qty * k, g.mult, g.root, g.expiration) if isinstance(g, OptionLeg)
                else IndexLeg(g.qty * k, g.entry) for g in self.legs]
        return Position(name or self.name, legs, dict(self.meta))

    # ------------------------------------------------------------------ valuation
    def leg_values(self, mkt: Market) -> np.ndarray:
        """Model value of each leg (signed, with multipliers)."""
        out = []
        for g in self.legs:
            if isinstance(g, IndexLeg):
                out.append(g.qty * (mkt.spot - g.entry))
                continue
            tau = g.T - mkt.t
            px = float(mkt.price(g.cp, g.K, tau)) if tau > 0 else max(g.cp * (mkt.spot - g.K), 0.0)
            out.append(g.qty * g.mult * px)
        return np.array(out)

    def value(self, mkt: Market) -> float:
        return float(self.leg_values(mkt).sum())

    def mid_value(self, mkt: Market, fallback: bool = True) -> float:
        """Value at the listed mids (model value for legs without a listed quote if fallback, else NaN)."""
        total = 0.0
        for g, v in zip(self.legs, self.leg_values(mkt)):
            if isinstance(g, OptionLeg):
                q = mkt.quote(g.cp, g.K, g.T)
                if q is not None:
                    total += g.qty * g.mult * float(q["mid"])
                    continue
                if not fallback:
                    return float("nan")
            total += v
        return total

    def execution_value(self, mkt: Market) -> float:
        """Value of the legs at the price paid to enter them: asks for purchases, bids for sales."""
        total = 0.0
        for g, v in zip(self.legs, self.leg_values(mkt)):
            q = mkt.quote(g.cp, g.K, g.T) if isinstance(g, OptionLeg) else None
            if q is None:
                total += v
            else:
                total += g.qty * g.mult * float(q["ask"] if g.qty > 0 else q["bid"])
        return total

    # ------------------------------------------------------------------ Greeks
    def greeks(self, mkt: Market) -> dict:
        """Sum of the legs' BSM Greeks in raw units (see greeks.bsm_greeks), with signs and multipliers."""
        names = ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm", "speed", "color")
        tot = dict.fromkeys(names, 0.0)
        for g in self.legs:
            if isinstance(g, IndexLeg):
                tot["delta"] += g.qty
                continue
            tau = g.T - mkt.t
            if tau <= 0:
                continue
            F, D, s = (float(v) for v in (mkt.forward(tau), mkt.discount(tau), mkt.iv(g.K, tau)))
            r = -np.log(D) / tau
            q = r - np.log(F / mkt.spot) / tau
            gk = bsm_greeks(g.cp, mkt.spot, g.K, tau, r, s, q)
            for n in names:
                tot[n] += g.qty * g.mult * float(gk[n])
        return tot

    def risk(self, mkt: Market) -> dict:
        """Desk units: P&L of a +1% index move (first order), of a 1% move at second order, per +1 vol point, per
        calendar day; rho per +1bp; vanna as the change in the 1% delta per vol point."""
        g = self.greeks(mkt)
        S = mkt.spot
        return {"delta_1pct": g["delta"] * S / 100, "gamma_1pct": 0.5 * g["gamma"] * (S / 100) ** 2,
                "vega_1vp": g["vega"] / 100, "theta_1d": g["theta"] / 365, "rho_1bp": g["rho"] * 1e-4,
                "vanna_1pct_1vp": g["vanna"] * S / 1e4, "volga_1vp": 0.5 * g["volga"] * 1e-4,
                "charm_1pct_1d": g["charm"] * S / 100 / 365, "delta_units": g["delta"]}


def black_sensitivities(cp, F, K, tau, sigma, D) -> dict:
    """Black-76 value and its derivatives in F, sigma, tau (at fixed F and D) and D."""
    th = cp_sign(cp)
    F, K, tau, sigma, D = (np.asarray(v, dtype=float) for v in (F, K, tau, sigma, D))
    sq = np.sqrt(tau)
    v = sigma * sq
    d1 = np.log(F / K) / v + 0.5 * v
    d2 = d1 - v
    n = norm_pdf(d1)
    out = {"value": np.asarray(black_price(th, F, K, tau, sigma, D), dtype=float)}
    out["dF"] = D * th * ndtr(th * d1)
    out["dFF"] = D * n / (F * v)
    out["ds"] = D * F * n * sq
    out["dtau"] = D * F * n * sigma / (2.0 * sq)
    out["dFs"] = -D * n * d2 / sigma
    out["dss"] = out["ds"] * d1 * d2 / sigma
    out["dD"] = out["value"] / D
    return out


def attribute(pos: Position, m0: Market, m1: Market) -> dict:
    """Taylor attribution of the model value change from m0 to m1 (see module docstring). Legs that expire in
    between go wholly to the residual."""
    tot = dict.fromkeys(TERMS, 0.0)
    actual = 0.0
    for g in pos.legs:
        if isinstance(g, IndexLeg):
            d = g.qty * (m1.spot - m0.spot)
            tot["delta"] += d
            actual += d
            continue
        t0, t1 = g.T - m0.t, g.T - m1.t
        if t0 <= 0:
            continue
        F0, D0, s0 = (float(v) for v in (m0.forward(t0), m0.discount(t0), m0.iv(g.K, t0)))
        v0 = float(black_price(g.cp, F0, g.K, t0, s0, D0))
        if t1 <= 0:
            actual += g.qty * g.mult * (max(g.cp * (m1.spot - g.K), 0.0) - v0)
            continue
        F1, D1, s1 = (float(v) for v in (m1.forward(t1), m1.discount(t1), m1.iv(g.K, t1)))
        v1 = float(black_price(g.cp, F1, g.K, t1, s1, D1))
        b = black_sensitivities(g.cp, F0, g.K, t0, s0, D0)
        dF, ds, dt, dD = F1 - F0, s1 - s0, t1 - t0, D1 - D0
        k = g.qty * g.mult
        tot["delta"] += k * float(b["dF"]) * dF
        tot["gamma"] += k * 0.5 * float(b["dFF"]) * dF ** 2
        tot["vega"] += k * float(b["ds"]) * ds
        tot["vanna"] += k * float(b["dFs"]) * dF * ds
        tot["volga"] += k * 0.5 * float(b["dss"]) * ds ** 2
        tot["theta"] += k * float(b["dtau"]) * dt
        tot["rates"] += k * float(b["dD"]) * dD
        actual += k * (v1 - v0)
    tot["residual"] = actual - sum(tot[t] for t in TERMS)
    tot["total"] = actual
    return tot


def legs_frame(pos: Position, mkt: Market) -> pd.DataFrame:
    """One row per leg: contract, quantity, time to expiry, strike, implied vol, forward delta, model value, quotes."""
    rows = []
    for g in pos.legs:
        if isinstance(g, IndexLeg):
            rows.append({"leg": g.label, "qty": g.qty, "model_px": mkt.spot, "value": g.qty * (mkt.spot - g.entry)})
            continue
        tau = g.T - mkt.t
        F, s = float(mkt.forward(tau)), float(mkt.iv(g.K, tau))
        d1 = np.log(F / g.K) / (s * np.sqrt(tau)) + 0.5 * s * np.sqrt(tau)
        q = mkt.quote(g.cp, g.K, g.T)
        rows.append({"leg": g.label, "qty": g.qty, "days": tau * 365, "K": g.K, "iv": s,
                     "fwd_delta": g.cp * float(ndtr(g.cp * d1)), "model_px": float(mkt.price(g.cp, g.K, tau)),
                     "bid": None if q is None else float(q["bid"]), "ask": None if q is None else float(q["ask"]),
                     "value": g.qty * g.mult * float(mkt.price(g.cp, g.K, tau))})
    return pd.DataFrame(rows)
