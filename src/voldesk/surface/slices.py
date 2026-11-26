"""Market smiles ("slices") for surface fitting, built from a cleaned chain (voldesk.data.chains).

A slice is one expiry on one date: forward F, discount factor D, time T, and the out-of-the-money quotes with bid /
mid / ask implied vols. Filters, each learned the hard way on SPX (see docs/models.md):
  * out-of-the-money side only, two-sided quotes, all three implied vols valid and iv_ask >= iv_bid;
  * forward |delta| >= 1% at the mid vol: beyond that SPX wings are minimum-tick bids whose vols reflect the price
    floor, not the smile, and they wreck every fit;
  * at least 6 quotes, T >= 7 calendar days (min_days), and one settlement style per expiry date (AM-settled
    standard contracts and PM-settled ones expire hours apart; fitted separately, their wings would create spurious
    calendar arbitrage over that gap).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..data.chains import otm_quotes


@dataclass
class Slice:
    root: str
    expiration: pd.Timestamp
    settlement: str
    T: float
    F: float
    D: float
    K: np.ndarray
    cp: np.ndarray            # +1 OTM call, -1 OTM put
    bid: np.ndarray
    ask: np.ndarray
    iv_bid: np.ndarray
    iv_mid: np.ndarray
    iv_ask: np.ndarray
    open_interest: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> np.ndarray:
        return np.log(self.K / self.F)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid + self.ask)

    @property
    def w_mid(self) -> np.ndarray:
        return self.iv_mid ** 2 * self.T

    @property
    def iv_spread(self) -> np.ndarray:
        return self.iv_ask - self.iv_bid

    @property
    def label(self) -> str:
        return f"{self.root} {pd.Timestamp(self.expiration):%Y-%m-%d}"

    def forward_delta(self) -> np.ndarray:
        sd = self.iv_mid * np.sqrt(self.T)
        d1 = -self.k / sd + sd / 2
        return np.where(self.cp > 0, norm.cdf(d1), norm.cdf(-d1))

    def subset(self, mask: np.ndarray, **meta) -> "Slice":
        arrays = {f: getattr(self, f)[mask] for f in ("K", "cp", "bid", "ask", "iv_bid", "iv_mid", "iv_ask",
                                                      "open_interest")}
        return Slice(root=self.root, expiration=self.expiration, settlement=self.settlement, T=self.T, F=self.F,
                     D=self.D, meta={**self.meta, **meta}, **arrays)

    def filtered(self, min_delta: float = 0.01) -> "Slice":
        m = self.forward_delta() >= min_delta
        return self.subset(m, min_delta=min_delta, n_before_delta_filter=int(m.size))


def slices_from_chain(chain: pd.DataFrame, *, min_days: float = 7.0, max_days: float = 1100.0, min_quotes: int = 6,
                      min_delta: float = 0.01, etypes=("standard_am",), root: str = "") -> list[Slice]:
    """Slices for one date from a cleaned chain (rows of a single trade date)."""
    if chain["date"].nunique() != 1:
        raise ValueError("slices_from_chain expects one trade date")
    q = otm_quotes(chain[chain["etype"].isin(etypes)])
    out = []
    for (ex, am), g in q.groupby(["exdate", "am"]):
        T = float(g["T"].iloc[0])
        if not (min_days / 365.0 <= T <= max_days / 365.0):
            continue
        g = g.sort_values("K")
        s = Slice(root=root or ("AM" if am else "PM"), expiration=pd.Timestamp(ex), settlement="AM" if am else "PM",
                  T=T, F=float(g["F"].iloc[0]), D=float(g["D"].iloc[0]), K=g["K"].to_numpy(float),
                  cp=g["cp"].to_numpy(float), bid=g["bid"].to_numpy(float), ask=g["ask"].to_numpy(float),
                  iv_bid=g["iv_bid"].to_numpy(float), iv_mid=g["iv_mid"].to_numpy(float),
                  iv_ask=g["iv_ask"].to_numpy(float), open_interest=g["oi"].to_numpy(float),
                  meta={"n_pairs": int(g["n_pairs"].iloc[0]) if "n_pairs" in g else -1,
                        "fwd_source": str(g["fwd_source"].iloc[0]) if "fwd_source" in g else ""})
        s = s.filtered(min_delta)
        if s.K.size >= min_quotes:
            out.append(s)
    # one settlement style per expiry date: keep the slice with more quotes
    best: dict = {}
    for s in out:
        key = pd.Timestamp(s.expiration).date()
        if key not in best or s.K.size > best[key].K.size:
            best[key] = s
    return sorted(best.values(), key=lambda x: x.T)
