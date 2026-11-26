"""The strategy library backtested on SPX (monthly cycle, ~30 days to expiry at entry unless stated).

Every structure is defined in strategy units (contracts per unit, multiplier 100). Strikes by forward delta at our mid
implied vol, at the money (strike nearest the forward), or by moneyness K/F. 'dh' = delta-hedged at every close with a
futures-like position on the near forward.
"""
from __future__ import annotations

import numpy as np

from ..strategies.book import black_sensitivities
from .engine import Leg, Strategy

ATM = None


def variance_strip_builder(sign: float = -1.0):
    """Static replication of a variance swap on the near expiry: out-of-the-money options with weights
    proportional to dK / K^2 (puts below the forward, calls at and above it), scaled so the strip's vega equals that of
    one ATM straddle (fractional contracts allowed). Delta-hedged daily, the strip's P&L tracks the variance swap's,
    up to the strike grid, the wings beyond the quoted strikes and the hedge discretisation."""
    def build(elig, near, far):
        g = elig[elig["exdate"] == near]
        if g.empty:
            return []
        F = float(g["F"].iloc[0])
        otm = g[((g["cp"] == 1) & (g["K"] >= F)) | ((g["cp"] == -1) & (g["K"] < F))].sort_values("K")
        if len(otm) < 6:
            return []
        K = otm["K"].to_numpy()
        dK = np.gradient(K)
        w = dK / K ** 2
        b = black_sensitivities(otm["cp"].to_numpy(), otm["F"].to_numpy(), K, otm["T"].to_numpy(),
                                otm["iv_mid"].to_numpy(), otm["D"].to_numpy())
        strip_vega = float(np.nansum(w * b["ds"]))
        # ATM straddle vega at the strike nearest the forward
        atm = g.iloc[(g["K"] - F).abs().argsort()[:1]]
        T, s, D = float(atm["T"].iloc[0]), float(atm["iv_mid"].iloc[0]), float(atm["D"].iloc[0])
        Ka = float(atm["K"].iloc[0])
        st = black_sensitivities(np.array([1.0, -1.0]), np.array([F, F]), np.array([Ka, Ka]), np.array([T, T]),
                                 np.array([s, s]), np.array([D, D]))
        scale = float(np.sum(st["ds"])) / strip_vega if strip_vega > 0 else 0.0
        return [(int(i), sign * scale * wi) for i, wi in zip(otm.index, w)]
    return build


LIBRARY = [
    Strategy("short_straddle", (Leg(1, -1), Leg(-1, -1)), group="Short volatility",
             description="Sell the ~30-day ATM straddle, unhedged, hold to the last trading day"),
    Strategy("short_strangle_25", (Leg(-1, -1, delta=0.25), Leg(1, -1, delta=0.25)), group="Short volatility",
             description="Sell the 25-delta put and call, unhedged"),
    Strategy("long_straddle", (Leg(1, 1), Leg(-1, 1)), group="Long volatility",
             description="Buy the ~30-day ATM straddle, unhedged"),
    Strategy("long_put_10", (Leg(-1, 1, delta=0.10),), group="Long volatility",
             description="Buy the 10-delta put (tail hedge), unhedged"),
    Strategy("short_straddle_dh", (Leg(1, -1), Leg(-1, -1)), hedge="daily", group="Variance risk premium",
             description="Sell the ATM straddle and delta-hedge at every close"),
    Strategy("short_var_strip_dh", (), hedge="daily", builder=variance_strip_builder(-1.0),
             group="Variance risk premium",
             description="Sell a 1/K^2-weighted strip of OTM options (variance-swap replication), delta-hedged daily"),
    Strategy("long_straddle_dh", (Leg(1, 1), Leg(-1, 1)), hedge="daily", group="Gamma scalping",
             description="Buy the ATM straddle and scalp gamma: rebalance the hedge at every close"),
    Strategy("long_straddle_band", (Leg(1, 1), Leg(-1, 1)), hedge="band:10", group="Gamma scalping",
             description="Buy the ATM straddle; rebalance only when |delta| exceeds 10 deltas per straddle"),
    Strategy("put_spread_25_10", (Leg(-1, -1, delta=0.25), Leg(-1, 1, delta=0.10)), group="Vertical spreads",
             description="Put credit spread: sell the 25-delta put, buy the 10-delta put"),
    Strategy("call_spread_50_25", (Leg(1, 1, delta=0.50), Leg(1, -1, delta=0.25)), group="Vertical spreads",
             description="Call debit spread: buy the 50-delta call, sell the 25-delta call"),
    Strategy("iron_condor_25_10", (Leg(-1, -1, delta=0.25), Leg(-1, 1, delta=0.10), Leg(1, -1, delta=0.25),
                                   Leg(1, 1, delta=0.10)), group="Premium selling",
             description="Short iron condor: sell 25-delta put and call, buy 10-delta wings"),
    Strategy("call_butterfly", (Leg(1, 1, mny=0.95), Leg(1, -2, mny=1.00), Leg(1, 1, mny=1.05)), group="Butterflies",
             description="Long call butterfly on strikes nearest 95% / 100% / 105% of the forward"),
    Strategy("risk_reversal_25_dh", (Leg(1, 1, delta=0.25), Leg(-1, -1, delta=0.25)), hedge="daily", group="Skew",
             description="Buy the 25-delta call, sell the 25-delta put (sell skew), delta-hedged daily"),
    Strategy("calendar_atm_dh", (Leg(1, -1), Leg(1, 1, cycle=1, match=0)), hedge="daily", group="Term structure",
             description="Long ATM call calendar: sell the near month, buy the next month at the same strike, "
                         "delta-hedged daily"),
    Strategy("diagonal_put_25_dh", (Leg(-1, -1, delta=0.25), Leg(-1, 1, delta=0.25, cycle=1)), hedge="daily",
             group="Term structure",
             description="Put diagonal: sell the near 25-delta put, buy the next month's 25-delta put, hedged daily"),
    Strategy("short_put_25_dh", (Leg(-1, -1, delta=0.25),), hedge="daily", group="Delta-hedged options",
             description="Sell the 25-delta put and delta-hedge daily"),
    Strategy("short_call_25_dh", (Leg(1, -1, delta=0.25),), hedge="daily", group="Delta-hedged options",
             description="Sell the 25-delta call and delta-hedge daily"),
]
BY_NAME = {s.name: s for s in LIBRARY}
