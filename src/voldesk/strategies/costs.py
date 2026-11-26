"""Transaction costs.

Options. Each contract traded costs commission plus a share of the quoted half-spread: buying at
mid + f (ask - bid) / 2 and selling at mid - f (ask - bid) / 2. f = 1 crosses the quoted spread (pays the ask,
receives the bid); f = 0.5 trades halfway between the mid and the touch, the usual estimate for effective spreads of
index options; f = 0 is frictionless. The default here is f = 0.5 with f = 1 as the stress case in every report.

Delta hedges. Trading the underlying (futures or the index basket) costs `hedge_bps` basis points of the notional
traded (half-spread plus fees; 1 bp for S&P 500 futures by default).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    spread_fraction: float = 0.5        # share of the quoted half-spread paid per trade
    commission: float = 1.0             # $ per contract (commission + exchange/clearing fees)
    hedge_bps: float = 1.0              # bp of notional for underlying/futures trades
    multiplier: float = 100.0

    def trade_price(self, qty, bid, ask):
        """Execution price per unit: above the mid for purchases (qty > 0), below it for sales."""
        mid = 0.5 * (np.asarray(bid, float) + np.asarray(ask, float))
        half = 0.5 * (np.asarray(ask, float) - np.asarray(bid, float))
        return mid + np.sign(qty) * self.spread_fraction * half

    def option_cost(self, qty, bid, ask) -> float:
        """Cost in $ (always >= 0) of trading qty contracts relative to the mid: spread share + commission."""
        qty = np.asarray(qty, float)
        half = 0.5 * (np.asarray(ask, float) - np.asarray(bid, float))
        return float(np.sum(np.abs(qty) * self.multiplier * self.spread_fraction * np.maximum(half, 0.0)
                            + np.abs(qty) * self.commission))

    def hedge_cost(self, units, price) -> float:
        """Cost in $ of trading `units` of the underlying (index units, i.e. already times the multiplier)."""
        return float(np.sum(np.abs(np.asarray(units, float)) * np.asarray(price, float)) * self.hedge_bps * 1e-4)

    @staticmethod
    def option_components(qty, bid, ask, multiplier: float = 100.0) -> tuple[float, float]:
        """(full quoted half-spread in $, contracts traded): any cost model's option cost is
        spread_fraction x the first + commission x the second."""
        qty = np.abs(np.asarray(qty, float))
        half = np.maximum(0.5 * (np.asarray(ask, float) - np.asarray(bid, float)), 0.0)
        half = np.where(np.isfinite(half), half, 0.0)
        return float(np.sum(qty * multiplier * half)), float(np.sum(qty))

    def from_components(self, spread_full, contracts, hedge_notional):
        return (self.spread_fraction * np.asarray(spread_full, float) + self.commission * np.asarray(contracts, float)
                + self.hedge_bps * 1e-4 * np.asarray(hedge_notional, float))


FRICTIONLESS = CostModel(0.0, 0.0, 0.0)
BASE = CostModel()
STRESS = CostModel(spread_fraction=1.0, commission=1.0, hedge_bps=2.0)
