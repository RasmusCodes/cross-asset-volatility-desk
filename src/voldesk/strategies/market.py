"""Pricing state for European options on one underlying, and the scenario moves that risk calculations apply to it.

Time. A Market has a clock t: years (ACT/365F) since a reference instant. Options expire at absolute times T on the
same clock, so time to expiry is tau = T - t.

Curves. Forwards and discount factors are known at node maturities (the listed expiries of a close: parity forwards,
zero-curve discount factors; see voldesk.data.chains) or follow constant rates in a simulation. Between nodes ln F and
ln D are linear in tau (piecewise-constant carry and forward rates), anchored at (0, ln S) and (0, 0); beyond the last
node the average carry and rate to that node continue.

Volatility. vol_fn(k, tau) is the Black implied vol at log forward moneyness k = ln(K / F(tau)): a fitted surface
(voldesk.surface: SVISurface, eSSVI, SSVI, ...) through ``surface_vol``, or a flat / parametric function.

Scenario moves (each returns a new Market):
  move(x, sticky)  every forward times e^x (an index log return x). "moneyness": the smile keeps its shape in K/F, so
                   the vol at a fixed strike moves with the index; "strike": the vol at a fixed strike is unchanged
  scale_vol(m)     implied vols times m(tau), a number or a function of time to expiry (proportional moves)
  shift_vol(a)     implied vols plus a(tau)
  roll(dt)         the clock advances dt years with the index unchanged: forwards to fixed dates lose dt of carry,
                   discount factors to fixed dates rise, and the surface keeps its shape in time to expiry
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from ..pricing.bsm import black_price

VOL_FLOOR = 1e-4


def _as_fn(v) -> Callable:
    return v if callable(v) else (lambda tau, c=float(v): np.full(np.shape(tau), c))


def surface_vol(surface) -> Callable:
    """vol(k, tau) from any surface object with iv(k, T) (SVISurface, SSVISurface, ...)."""
    def f(k, tau):
        k, tau = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(tau, dtype=float))
        out = np.full(k.shape, np.nan)
        ok = np.isfinite(k) & np.isfinite(tau) & (tau > 0)
        for T in np.unique(tau[ok]):
            m = ok & (tau == T)
            out[m] = surface.iv(k[m], float(T))
        return out
    return f


@dataclass(frozen=True)
class Market:
    spot: float
    tau_nodes: np.ndarray
    F_nodes: np.ndarray
    D_nodes: np.ndarray
    vol_fn: Callable
    t: float = 0.0
    label: str = ""
    quotes: pd.DataFrame | None = field(default=None, repr=False)   # listed contracts: root, expiration, cp, strike, bid, ask, mid, T, tau
    meta: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ curves and prices
    def _curve(self, tau, y0: float, y_nodes: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        xs = np.concatenate([[0.0], self.tau_nodes])
        ys = np.concatenate([[y0], y_nodes])
        out = np.interp(tau, xs, ys)
        slope = (ys[-1] - y0) / xs[-1]
        return np.where(tau > xs[-1], y0 + slope * tau, out)

    def forward(self, tau) -> np.ndarray:
        return np.exp(self._curve(tau, np.log(self.spot), np.log(self.F_nodes)))

    def discount(self, tau) -> np.ndarray:
        return np.exp(self._curve(tau, 0.0, np.log(self.D_nodes)))

    def iv(self, K, tau) -> np.ndarray:
        K, tau = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(tau, dtype=float))
        return np.maximum(self.vol_fn(np.log(K / self.forward(tau)), tau), VOL_FLOOR)

    def price(self, cp, K, tau) -> np.ndarray:
        """Black-76 value of European options with time to expiry tau (> 0)."""
        return black_price(cp, self.forward(tau), K, tau, self.iv(K, tau), self.discount(tau))

    # ------------------------------------------------------------------ scenario moves
    def move(self, x: float, sticky: str = "moneyness") -> "Market":
        if sticky not in ("moneyness", "strike"):
            raise ValueError("sticky must be 'moneyness' or 'strike'")
        f = self.vol_fn if sticky == "moneyness" else (lambda k, tau, g=self.vol_fn, x=float(x): g(np.asarray(k) + x, tau))
        g = np.exp(float(x))
        return replace(self, spot=self.spot * g, F_nodes=self.F_nodes * g, vol_fn=f)

    def scale_vol(self, m) -> "Market":
        mf = _as_fn(m)
        return replace(self, vol_fn=lambda k, tau, g=self.vol_fn: mf(np.asarray(tau, dtype=float)) * g(k, tau))

    def shift_vol(self, a) -> "Market":
        af = _as_fn(a)
        return replace(self, vol_fn=lambda k, tau, g=self.vol_fn: np.maximum(g(k, tau) + af(np.asarray(tau, dtype=float)), VOL_FLOOR))

    def roll(self, dt: float) -> "Market":
        keep = self.tau_nodes > dt + 1e-12
        tau, F, D = self.tau_nodes[keep], self.F_nodes[keep], self.D_nodes[keep]
        carry = np.log(F / self.spot) / tau
        rate = -np.log(D) / tau
        q = None
        if self.quotes is not None:
            q = self.quotes.assign(tau=self.quotes["tau"] - dt)
            q = q[q["tau"] > 0]
        return replace(self, t=self.t + dt, tau_nodes=tau - dt, F_nodes=F * np.exp(-carry * dt),
                       D_nodes=D * np.exp(rate * dt), quotes=q)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def flat(cls, spot: float, sigma: float, r: float = 0.0, q: float = 0.0, t: float = 0.0, label: str = "flat") -> "Market":
        """Constant rates and a flat implied vol: the curves are exact at every maturity."""
        tau = np.array([10.0])
        return cls(spot=float(spot), tau_nodes=tau, F_nodes=spot * np.exp((r - q) * tau), D_nodes=np.exp(-r * tau),
                   vol_fn=lambda k, tau, s=float(sigma): np.full(np.broadcast(np.asarray(k), np.asarray(tau)).shape, s),
                   t=float(t), label=label, meta={"r": r, "q": q, "sigma": sigma})

    @classmethod
    def from_nodes(cls, spot: float, tau, F, D, vol_fn: Callable, quotes: pd.DataFrame | None = None, t: float = 0.0,
                   label: str = "", meta: dict | None = None) -> "Market":
        o = np.argsort(np.asarray(tau, dtype=float))
        return cls(spot=float(spot), tau_nodes=np.asarray(tau, float)[o], F_nodes=np.asarray(F, float)[o],
                   D_nodes=np.asarray(D, float)[o], vol_fn=vol_fn, t=float(t), label=label, quotes=quotes,
                   meta=meta or {})

    @classmethod
    def from_chain(cls, chain: pd.DataFrame, vol_fn: Callable, label: str = "") -> "Market":
        """A cleaned one-day chain (voldesk.data.chains): one node per expiry (forward, discount), the spot from the
        chain's close, and its two-sided contracts as listed quotes (root = settlement style)."""
        nodes = chain.groupby(["exdate", "am"], as_index=False).agg(T=("T", "first"), F=("F", "first"), D=("D", "first"))
        nodes = nodes.sort_values("T").drop_duplicates("T")
        q = chain[(chain["bid"] > 0) & (chain["ask"] >= chain["bid"])]
        quotes = pd.DataFrame({"root": np.where(q["am"] == 1, "AM", "PM"),
                               "expiration": pd.to_datetime(q["exdate"]).dt.strftime("%Y-%m-%d"),
                               "cp": q["cp"].astype(int), "strike": q["K"], "bid": q["bid"], "ask": q["ask"],
                               "mid": q["mid"], "tau": q["T"], "T": q["T"], "open_interest": q["oi"]})
        return cls.from_nodes(float(chain["S"].iloc[0]), nodes["T"], nodes["F"], nodes["D"], vol_fn,
                              quotes=quotes.reset_index(drop=True), label=label)

    # ------------------------------------------------------------------ listed contracts
    def listed_expiries(self, root: str | None = None) -> pd.DataFrame:
        if self.quotes is None:
            raise ValueError("this market has no listed contracts")
        q = self.quotes if root is None else self.quotes[self.quotes["root"] == root]
        return q.groupby(["root", "expiration"], as_index=False).agg(T=("T", "first"), tau=("tau", "first"), n=("strike", "size"))

    def quote(self, cp: int, K: float, T: float) -> pd.Series | None:
        """The listed contract with this type, strike and expiry (absolute T, matched to 1 second), if any."""
        if self.quotes is None:
            return None
        q = self.quotes
        m = (q["cp"] == cp) & (np.abs(q["strike"] - K) < 1e-9) & (np.abs(q["T"] - T) < 1.0 / (365.0 * 86400.0))
        return q[m].iloc[0] if m.any() else None
