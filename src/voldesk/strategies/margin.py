"""Margin requirements for option positions (customer accounts), two ways.

Strategy-based (Cboe Rule 10.3 / FINRA Rule 4210 style), per expiry group:
  long option            paid in full (premium); no further requirement
  short option           premium + max(pct x underlying - out-of-the-money amount, min_pct x underlying (calls) or
                         min_pct x strike (puts)); pct = 15% and min_pct = 10% for broad-based indexes and broad-based
                         index ETFs (SPX, SPY, QQQ, IWM, DIA), 20% / 10% for single stocks
  defined-risk structures (verticals, iron condors, butterflies: every short covered by a long of the same type and
                         expiry) the maximum loss at expiry over all underlying prices, computed exactly from the payoff
                         kinks (the premium paid is part of that loss for debit structures)
  short straddle/strangle the larger of the call-side and put-side naked requirements plus the other side's premium
  calendars / diagonals  a short covered by a long of the same type expiring later, with a strike at least as
                         favourable (calls: K_long <= K_short; puts: K_long >= K_short): the net debit; otherwise the
                         short is margined as naked
This module implements the rules exactly as stated above; broker "house" requirements are higher and are not modelled.

Portfolio margin (simplified TIMS / Cboe portfolio margin for broad-based indexes): the largest loss of the whole
position over underlying moves from -8% to +6% (11 points) combined with implied vols scaled by 0.75, 1 and 1.25 (a
vol stress the exchange rule leaves to the clearing model), with a floor of $37.50 per contract. Full revaluation
on the position's Market.
"""
from __future__ import annotations

import numpy as np

from .book import IndexLeg, OptionLeg, Position
from .market import Market

BROAD = {"SPX", "SPY", "QQQ", "IWM", "DIA", "NDX", "RUT", "XSP"}


def naked_requirement(cp: int, K: float, S: float, premium: float, broad: bool = True) -> float:
    """Per unit of underlying (multiply by contracts x multiplier): premium + max(pct S - OTM, min_pct base)."""
    pct = 0.15 if broad else 0.20
    otm = max(K - S, 0.0) if cp > 0 else max(S - K, 0.0)
    floor = 0.10 * (S if cp > 0 else K)
    return premium + max(pct * S - otm, floor)


def _payoff_kinks(legs: list) -> np.ndarray:
    ks = sorted({g.K for g in legs})
    return np.array([0.0] + ks + [2.0 * max(ks) + 1.0])


def expiry_payoff(legs: list, S_T: np.ndarray) -> np.ndarray:
    out = np.zeros_like(S_T, dtype=float)
    for g in legs:
        out += g.qty * g.mult * np.maximum(g.cp * (S_T - g.K), 0.0)
    return out


def strategy_margin(pos: Position, mkt: Market, underlying: str = "SPX") -> dict:
    """Strategy-based requirement in $ (see module docstring) for a Position valued on a Market."""
    legs = []
    for g in pos.options:
        tau = g.T - mkt.t
        px = float(mkt.price(g.cp, g.K, tau)) if tau > 0 else max(g.cp * (mkt.spot - g.K), 0.0)
        legs.append(MarginLeg(g.cp, g.K, g.T, g.qty, g.mult, px))
    out = requirement(legs, mkt.spot, underlying)
    for g in pos.legs:
        if isinstance(g, IndexLeg):
            out["requirement"] += 0.06 * abs(g.qty) * mkt.spot      # futures-style initial margin on the hedge
    return out


class MarginLeg:
    __slots__ = ("cp", "K", "T", "qty", "mult", "px")

    def __init__(self, cp, K, T, qty, mult, px):
        self.cp, self.K, self.T, self.qty, self.mult, self.px = int(cp), float(K), float(T), float(qty), float(mult), float(px)


def requirement(legs: list, S: float, underlying: str = "SPX") -> dict:
    """Strategy-based requirement from legs with known prices (MarginLeg: cp, K, T, qty, mult, px)."""
    broad = underlying.upper() in BROAD
    by_T: dict = {}
    for g in legs:
        by_T.setdefault(round(g.T, 10), []).append(g)
    total, detail = 0.0, []
    covered_ids: set = set()
    # calendars/diagonals: shorts covered by later longs of the same type with a favourable strike -> net debit
    Ts = sorted(by_T)
    for i, T in enumerate(Ts):
        for g in by_T[T]:
            if g.qty >= 0:
                continue
            for T2 in Ts[i + 1:]:
                hit = None
                for h in by_T[T2]:
                    if h.qty >= -g.qty and h.cp == g.cp and id(h) not in covered_ids and (
                            (g.cp > 0 and h.K <= g.K) or (g.cp < 0 and h.K >= g.K)):
                        hit = h
                        break
                if hit is not None:
                    covered_ids.update((id(hit), id(g)))
                    debit = (-g.qty) * g.mult * (hit.px - g.px) + (hit.qty + g.qty) * hit.mult * hit.px
                    total += max(debit, 0.0)
                    detail.append({"T": T, "rule": "calendar/diagonal (net debit)", "requirement": max(debit, 0.0)})
                    break
    for T, group in by_T.items():
        px = {id(g): g.px for g in group}
        free = [g for g in group if id(g) not in covered_ids]
        shorts = [g for g in free if g.qty < 0]
        longs = [g for g in free if g.qty > 0]
        premium_long = sum(g.qty * g.mult * px[id(g)] for g in longs)
        if not shorts:
            req, rule = premium_long, "long premium"
        else:
            def covered_same_expiry(gs, ls):
                for cp in (-1, 1):
                    s_q = sum(-g.qty for g in gs if g.cp == cp)
                    l_q = sum(g.qty for g in ls if g.cp == cp)
                    if s_q > l_q + 1e-12:
                        return False
                return True
            if covered_same_expiry(shorts, longs):
                grid = _payoff_kinks(free)
                pay = expiry_payoff(free, grid)
                net_premium = sum(g.qty * g.mult * px[id(g)] for g in free)   # paid (+) or received (-)
                loss = net_premium - pay
                req, rule = max(float(loss.max()), 0.0), "defined risk (max loss)"
            else:
                reqs = {}
                for cp in (-1, 1):
                    reqs[cp] = sum(-g.qty * g.mult * naked_requirement(g.cp, g.K, S, px[id(g)], broad)
                                   for g in shorts if g.cp == cp)
                prem = {cp: sum(-g.qty * g.mult * px[id(g)] for g in shorts if g.cp == cp) for cp in (-1, 1)}
                if reqs[1] > 0 and reqs[-1] > 0:
                    big = 1 if reqs[1] >= reqs[-1] else -1
                    req = reqs[big] + prem[-big]
                    rule = "short straddle/strangle"
                else:
                    req = reqs[1] + reqs[-1]
                    rule = "naked short"
                req += premium_long
        if not free:
            continue
        total += req
        detail.append({"T": T, "rule": rule, "requirement": req})
    return {"requirement": float(total), "detail": detail}


def portfolio_margin(pos: Position, mkt: Market, moves=None, vol_scales=(0.75, 1.0, 1.25),
                     per_contract_floor: float = 37.5) -> dict:
    """Simplified portfolio margin: worst full-revaluation loss over the move x vol grid, floored per contract."""
    moves = np.linspace(-0.08, 0.06, 11) if moves is None else np.asarray(moves, float)
    v0 = pos.value(mkt)
    worst, at = 0.0, (0.0, 1.0)
    for m in moves:
        moved = mkt.move(np.log1p(m), "moneyness")
        for s in vol_scales:
            pnl = pos.value(moved.scale_vol(s)) - v0
            if -pnl > worst:
                worst, at = -pnl, (float(m), float(s))
    n_contracts = sum(abs(g.qty) for g in pos.options)
    return {"requirement": float(max(worst, per_contract_floor * n_contracts)), "worst_loss": float(worst),
            "worst_at": at}
