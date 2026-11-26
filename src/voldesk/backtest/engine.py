"""Daily mark-to-market backtests of option strategies on historical chains.

One pass over the data steps every strategy through every trading day, so a year of chains is loaded once.

Cycle. A strategy holds one position at a time. On its first day, and on the last trading day of the position's near
expiry, it closes everything at that day's close and opens the next position at the same close. A position still open
on the last day of the data is closed at that close (the trading calendar ends there). The near expiry is
the AM-settled standard (third-Friday cycle) expiry whose days to expiry are closest to the target (at least
`min_dte`); a leg with cycle = 1 uses the next standard expiry after it. Positions are closed at the close of the last
trading day (the business day before the AM settlement), so no settlement price (the special opening quotation,
which the data do not contain) is ever needed.

Strikes. By forward delta at our mid implied vol (N(d1) for calls, N(-d1) for puts; Black-76 with the parity forward),
by moneyness K/F, or at the money (the strike closest to the forward with both a call and a put quoted). Only
contracts with a two-sided quote and a valid implied vol are eligible.

Marks and P&L. Every day each leg is marked at its mid. Daily P&L = change in mid value + hedge P&L - costs. A leg
whose contract has no row on a day is marked at Black-76 with its previous implied vol and that day's forward and
counted in `missing_marks` (the data report gives the frequency).

Hedging. 'none'; 'daily' (rebalance at every close to zero spot delta); 'band:x' (rebalance when the absolute spot
delta exceeds x per strategy unit, in index units per contract of multiplier 100, e.g. band:10 = 10 deltas). The hedge
is a futures-like position on the near expiry's forward: its daily P&L is h x (F_t+1 - F_t) for that expiry. Leg
deltas are converted to spot deltas with dF/dS = F/S, and the hedge ratio to forward units with S/F_near.

Costs. voldesk.strategies.costs.CostModel: a share of the quoted half-spread plus commission per contract on every
option trade (entry, exit, rolls) and basis points of notional on every hedge trade.

Attribution (daily, per leg, Taylor expansion of the Black-76 value in forward F, own implied vol s, time to expiry
tau and discount factor D, from the leg's own day-t values to day t+1): delta, gamma, vega, vanna, volga, theta,
rates; the residual is the rest of the mid change (higher orders, and any gap between mid and model value).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..pricing.bsm import black_price
from ..strategies.book import black_sensitivities
from ..strategies.costs import BASE, CostModel

MULT = 100.0
TERMS = ("delta", "gamma", "vega", "vanna", "volga", "theta", "rates", "residual")


@dataclass(frozen=True)
class Leg:
    cp: int                       # +1 call, -1 put
    qty: float                    # contracts per strategy unit; negative = short
    delta: float | None = None    # target |forward delta|; None with mny None = at the money
    mny: float | None = None      # target K / F
    cycle: int = 0                # 0 near expiry, 1 next standard expiry
    match: int | None = None      # use the strike chosen for legs[match] (calendars), if listed at this expiry


@dataclass(frozen=True)
class Strategy:
    name: str
    legs: tuple = ()
    dte: int = 30
    min_dte: int = 14
    hedge: str = "none"
    description: str = ""
    builder: object = None        # optional callable(day_chain, near_ex, far_ex) -> list[(row index, qty)] (e.g. strips)
    group: str = ""


@dataclass
class Position:
    entry_date: pd.Timestamp
    near_exdate: pd.Timestamp
    near_am: int
    exit_date: pd.Timestamp
    legs: pd.DataFrame            # optionid, cp, K, exdate, am, qty, entry_px, last_mid, last_iv, last_F, last_D, last_T
    hedge_units: float = 0.0      # futures-like units on the near forward (index units)
    margin: float = np.nan
    premium: float = 0.0
    costs: float = 0.0
    pnl: float = 0.0
    last_date: pd.Timestamp | None = None
    last_Fn: float = np.nan
    entry_components: tuple = (0.0, 0.0)


@dataclass
class RunState:
    strategy: Strategy
    pos: Position | None = None
    daily: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    hedge_count: int = 0


def forward_delta(cp, k, iv, T):
    sd = iv * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / sd + 0.5 * sd
    return ndtr(cp * d1)


def last_trading_day(settle: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp:
    i = days.searchsorted(settle, side="left") - 1
    return days[max(i, 0)]


def settlement_date(exdate: pd.Timestamp) -> pd.Timestamp:
    return exdate - pd.Timedelta(days=1) if exdate.dayofweek == 5 else exdate


class Backtester:
    def __init__(self, strategies: list[Strategy], costs: CostModel = BASE, margin_fn=None):
        self.states = [RunState(s) for s in strategies]
        self.costs = costs
        self.margin_fn = margin_fn            # callable(legs_df, day_chain) -> float, optional

    # -------------------------------------------------------------- selection
    @staticmethod
    def eligible(day: pd.DataFrame) -> pd.DataFrame:
        e = day[(day["etype"] == "standard_am") & (day["bid"] > 0) & (day["ask"] >= day["bid"])
                & np.isfinite(day["iv_mid"]) & (day["iv_mid"] > 0)]
        return e

    def choose_expiries(self, elig: pd.DataFrame, date: pd.Timestamp, strat: Strategy):
        ex = elig.groupby("exdate")["T"].first().sort_index()
        dte = (ex.index - date).days
        ok = dte >= strat.min_dte
        if not ok.any():
            return None, None
        cand = ex.index[ok]
        near = cand[int(np.argmin(np.abs((cand - date).days - strat.dte)))]
        later = ex.index[(ex.index - near).days >= 20]
        far = later[0] if len(later) else None
        return near, far

    @staticmethod
    def pick_strike(g: pd.DataFrame, leg: Leg) -> int | None:
        """Row index (into day) of the chosen contract within one expiry's eligible rows g."""
        if leg.delta is not None:
            c = g[g["cp"] == leg.cp]
            if c.empty:
                return None
            fd = forward_delta(leg.cp, c["k"].to_numpy(), c["iv_mid"].to_numpy(), c["T"].to_numpy())
            return int(c.index[int(np.nanargmin(np.abs(fd - leg.delta)))])
        if leg.mny is not None:
            c = g[g["cp"] == leg.cp]
            if c.empty:
                return None
            return int(c.index[int(np.argmin(np.abs(c["K"].to_numpy() / c["F"].to_numpy() - leg.mny)))])
        both = g.groupby("K")["cp"].nunique()
        ks = both.index[both == 2].to_numpy()
        if ks.size == 0:
            return None
        F = float(g["F"].iloc[0])
        K = ks[int(np.argmin(np.abs(ks - F)))]
        c = g[(g["K"] == K) & (g["cp"] == leg.cp)]
        return int(c.index[0]) if len(c) else None

    def open_position(self, st: RunState, day: pd.DataFrame, date: pd.Timestamp, days: pd.DatetimeIndex) -> None:
        strat = st.strategy
        elig = self.eligible(day)
        if elig.empty:
            return
        near, far = self.choose_expiries(elig, date, strat)
        if near is None:
            return
        picks: list = []
        if strat.builder is not None:
            picks = strat.builder(elig, near, far)
        else:
            for leg in strat.legs:
                ex = near if leg.cycle == 0 else far
                if ex is None:
                    return
                g = elig[elig["exdate"] == ex]
                if leg.match is not None:
                    K0 = float(day.at[picks[leg.match][0], "K"])
                    c = g[(g["cp"] == leg.cp) & (np.abs(g["K"] - K0) < 1e-9)]
                    idx = int(c.index[0]) if len(c) else None
                else:
                    idx = self.pick_strike(g, leg)
                if idx is None:
                    return
                picks.append((idx, leg.qty))
        if not picks:
            return
        rows = day.loc[[i for i, _ in picks]]
        qty = np.array([q for _, q in picks], float)
        legs = pd.DataFrame({"optionid": rows["optionid"].to_numpy(), "cp": rows["cp"].to_numpy(),
                             "K": rows["K"].to_numpy(), "exdate": rows["exdate"].to_numpy(), "am": rows["am"].to_numpy(),
                             "qty": qty, "bid": rows["bid"].to_numpy(), "ask": rows["ask"].to_numpy(),
                             "last_mid": rows["mid"].to_numpy(), "last_iv": rows["iv_mid"].to_numpy(),
                             "last_iv_own": rows["iv_own"].to_numpy() if "iv_own" in rows else rows["iv_mid"].to_numpy(),
                             "last_F": rows["F"].to_numpy(), "last_D": rows["D"].to_numpy(),
                             "last_T": rows["T"].to_numpy(), "last_S": rows["S"].to_numpy()})
        legs = legs.groupby(["optionid", "cp", "K", "exdate", "am"], as_index=False).agg(
            qty=("qty", "sum"), bid=("bid", "first"), ask=("ask", "first"), last_mid=("last_mid", "first"),
            last_iv=("last_iv", "first"), last_iv_own=("last_iv_own", "first"), last_F=("last_F", "first"),
            last_D=("last_D", "first"), last_T=("last_T", "first"), last_S=("last_S", "first"))
        legs = legs[legs["qty"] != 0].reset_index(drop=True)
        near_am = int(rows.loc[rows["exdate"] == near, "am"].iloc[0]) if (rows["exdate"] == near).any() else 1
        exit_date = last_trading_day(settlement_date(pd.Timestamp(near)), days)
        if exit_date <= date:
            return
        cost = self.costs.option_cost(legs["qty"].to_numpy(), legs["bid"].to_numpy(), legs["ask"].to_numpy())
        premium = float((legs["qty"] * legs["last_mid"]).sum() * MULT)
        pos = Position(entry_date=date, near_exdate=pd.Timestamp(near), near_am=near_am, exit_date=exit_date,
                       legs=legs, premium=premium, costs=cost, pnl=-cost)
        pos.last_date = date
        pos.entry_components = self.costs.option_components(legs["qty"].to_numpy(), legs["bid"].to_numpy(),
                                                            legs["ask"].to_numpy())
        if self.margin_fn is not None:
            try:
                pos.margin = float(self.margin_fn(legs, day))
            except Exception:  # noqa: BLE001 - margin is a report, never a reason to skip a trade
                pos.margin = np.nan
        st.pos = pos

    # -------------------------------------------------------------- daily step
    @staticmethod
    def spot_delta(legs: pd.DataFrame, S: float) -> float:
        k = np.log(legs["K"].to_numpy() / legs["last_F"].to_numpy())
        b = black_sensitivities(legs["cp"].to_numpy(), legs["last_F"].to_numpy(), legs["K"].to_numpy(),
                                legs["last_T"].to_numpy(), legs["last_iv"].to_numpy(), legs["last_D"].to_numpy())
        dvdf = np.where(np.isfinite(b["dF"]), b["dF"], 0.0)
        return float(np.sum(legs["qty"].to_numpy() * MULT * dvdf * legs["last_F"].to_numpy() / S))

    def step(self, st: RunState, day: pd.DataFrame, by_id: pd.Series, fwd: pd.DataFrame, date: pd.Timestamp,
             S: float, days: pd.DatetimeIndex) -> None:
        rec = {"date": date, "S": S, **{t: 0.0 for t in TERMS}, "hedge": 0.0, "costs": 0.0, "mtm": 0.0,
               "missing_marks": 0, "n_legs": 0, "spread_full": 0.0, "contracts": 0.0, "hedge_notional": 0.0}
        pos = st.pos
        if pos is not None:
            legs = pos.legs
            ids = legs["optionid"].to_numpy()
            present = np.isin(ids, by_id.index.to_numpy())
            if not present.all():
                # ids can change (e.g. at the switch from OptionMetrics to Cboe data): match by contract terms
                for j in np.flatnonzero(~present):
                    m = day[(day["cp"] == legs.at[j, "cp"]) & (np.abs(day["K"] - legs.at[j, "K"]) < 1e-9)
                            & (day["exdate"] == legs.at[j, "exdate"]) & (day["am"] == legs.at[j, "am"])]
                    if len(m):
                        legs.at[j, "optionid"] = m["optionid"].iloc[0]
                ids = legs["optionid"].to_numpy()
                present = np.isin(ids, by_id.index.to_numpy())
            mid1 = np.full(len(legs), np.nan)
            iv1, F1, D1, T1 = (np.full(len(legs), np.nan) for _ in range(4))
            iv1_own = np.full(len(legs), np.nan)
            bid1, ask1 = np.full(len(legs), np.nan), np.full(len(legs), np.nan)
            if present.any():
                rows = day.loc[by_id.loc[ids[present]].to_numpy()]
                mid1[present], iv1[present] = rows["mid"].to_numpy(), rows["iv_mid"].to_numpy()
                iv1_own[present] = rows["iv_own"].to_numpy() if "iv_own" in rows else rows["iv_mid"].to_numpy()
                F1[present], D1[present], T1[present] = rows["F"].to_numpy(), rows["D"].to_numpy(), rows["T"].to_numpy()
                bid1[present], ask1[present] = rows["bid"].to_numpy(), rows["ask"].to_numpy()
            miss = ~present | ~np.isfinite(mid1)
            if miss.any():
                rec["missing_marks"] = int(miss.sum())
                elapsed = (date - pos.last_date).days / 365.0
                for j in np.flatnonzero(miss):
                    ex, am = legs.at[j, "exdate"], legs.at[j, "am"]
                    f = fwd[(fwd["exdate"] == ex) & (fwd["am"] == am)]
                    Tj = max(float(legs.at[j, "last_T"]) - elapsed, 1e-6)
                    Fj = float(f["F"].iloc[0]) if len(f) else float(legs.at[j, "last_F"]) * S / float(legs.at[j, "last_S"])
                    Dj = float(legs.at[j, "last_D"])
                    ivj = float(legs.at[j, "last_iv"]) if np.isfinite(legs.at[j, "last_iv"]) else 0.2
                    mid1[j] = float(black_price(legs.at[j, "cp"], Fj, legs.at[j, "K"], Tj, ivj, Dj))
                    iv1[j], F1[j], D1[j], T1[j] = ivj, Fj, Dj, Tj
                    bid1[j] = ask1[j] = mid1[j]
            q = legs["qty"].to_numpy() * MULT
            dmid = mid1 - legs["last_mid"].to_numpy()
            rec["mtm"] = float(np.sum(q * dmid))
            # attribution from day t-1 values: the contract's own vol when defined on both days (the model value
            # then equals the mid exactly), else the out-of-the-money vol at its strike on both days
            F0, t0, D0 = (legs[c].to_numpy() for c in ("last_F", "last_T", "last_D"))
            own0 = legs["last_iv_own"].to_numpy() if "last_iv_own" in legs else legs["last_iv"].to_numpy()
            use_own = np.isfinite(own0) & (own0 > 0) & np.isfinite(iv1_own) & (iv1_own > 0)
            s0 = np.where(use_own, own0, legs["last_iv"].to_numpy())
            s1 = np.where(use_own, iv1_own, iv1)
            ok = np.isfinite(F0) & np.isfinite(s0) & (s0 > 0) & (t0 > 0) & np.isfinite(s1) & (s1 > 0)
            if ok.any():
                b = black_sensitivities(legs["cp"].to_numpy()[ok], F0[ok], legs["K"].to_numpy()[ok], t0[ok], s0[ok], D0[ok])
                dF, ds, dt, dD = F1[ok] - F0[ok], s1[ok] - s0[ok], T1[ok] - t0[ok], D1[ok] - D0[ok]
                qq = q[ok]
                terms = {"delta": b["dF"] * dF, "gamma": 0.5 * b["dFF"] * dF ** 2, "vega": b["ds"] * ds,
                         "vanna": b["dFs"] * dF * ds, "volga": 0.5 * b["dss"] * ds ** 2, "theta": b["dtau"] * dt,
                         "rates": b["dD"] * dD}
                for t, v in terms.items():
                    rec[t] = float(np.nansum(qq * v))
            rec["residual"] = rec["mtm"] - sum(rec[t] for t in TERMS if t != "residual")
            # hedge P&L on the near forward
            fn = fwd[(fwd["exdate"] == pos.near_exdate) & (fwd["am"] == pos.near_am)]
            Fn1 = float(fn["F"].iloc[0]) if len(fn) else np.nan
            if pos.hedge_units != 0.0 and np.isfinite(Fn1) and np.isfinite(getattr(pos, "last_Fn", np.nan)):
                rec["hedge"] = pos.hedge_units * (Fn1 - pos.last_Fn)
            legs["last_mid"], legs["last_iv"], legs["last_F"], legs["last_D"], legs["last_T"] = mid1, iv1, F1, D1, T1
            legs["last_iv_own"] = iv1_own
            legs["last_S"] = S
            legs["bid"], legs["ask"] = bid1, ask1
            pos.last_Fn = Fn1 if np.isfinite(Fn1) else getattr(pos, "last_Fn", np.nan)
            pos.last_date = date
            pos.pnl += rec["mtm"] + rec["hedge"]
            rec["n_legs"] = len(legs)
            # exit (and the roll happens below at the same close)
            if date >= pos.exit_date:
                c = self.costs.option_cost(legs["qty"].to_numpy(), legs["bid"].to_numpy(), legs["ask"].to_numpy())
                px_h = Fn1 if np.isfinite(Fn1) else S
                c += self.costs.hedge_cost(pos.hedge_units, px_h)
                sf, nc = self.costs.option_components(legs["qty"].to_numpy(), legs["bid"].to_numpy(),
                                                      legs["ask"].to_numpy())
                rec["spread_full"] += sf
                rec["contracts"] += nc
                rec["hedge_notional"] += abs(pos.hedge_units) * px_h
                rec["costs"] += c
                pos.costs += c
                pos.pnl -= c
                st.trades.append(self.trade_record(pos, legs, date, S))
                st.pos = None
        if st.pos is None:
            self.open_position(st, day, date, days)
            if st.pos is not None:
                rec["costs"] += st.pos.costs
                rec["spread_full"] += st.pos.entry_components[0]
                rec["contracts"] += st.pos.entry_components[1]
                fn = fwd[(fwd["exdate"] == st.pos.near_exdate) & (fwd["am"] == st.pos.near_am)]
                st.pos.last_Fn = float(fn["F"].iloc[0]) if len(fn) else np.nan
        # hedge rebalance at the close
        pos = st.pos
        if pos is not None and st.strategy.hedge != "none":
            delta = self.spot_delta(pos.legs, S)
            Fn = getattr(pos, "last_Fn", np.nan)
            if np.isfinite(Fn):
                target = -delta * S / Fn
                do = True
                if st.strategy.hedge.startswith("band:"):
                    band = float(st.strategy.hedge.split(":")[1]) * MULT / 100.0 * max(np.abs(pos.legs["qty"]).max(), 1)
                    do = abs(delta + pos.hedge_units * Fn / S) > band
                if do:
                    trade = target - pos.hedge_units
                    c = self.costs.hedge_cost(trade, Fn)
                    rec["hedge_notional"] += abs(trade) * Fn
                    rec["costs"] += c
                    pos.costs += c
                    pos.pnl -= c
                    pos.hedge_units = target
                    st.hedge_count += 1
        if pos is not None:
            g = self.position_greeks(pos.legs, S)
            g["delta_net"] = g["pos_delta"] + pos.hedge_units * (getattr(pos, "last_Fn", S) / S)
            rec.update(g)
            rec["notional"] = S * MULT * float(np.abs(pos.legs["qty"]).max())
            rec["margin"] = pos.margin
            rec["premium"] = pos.premium
        rec["pnl"] = rec["mtm"] + rec["hedge"] - rec["costs"]
        st.daily.append(rec)

    @staticmethod
    def position_greeks(legs: pd.DataFrame, S: float) -> dict:
        ok = np.isfinite(legs["last_iv"]) & (legs["last_iv"] > 0) & (legs["last_T"] > 0)
        if not ok.any():
            return {"pos_delta": 0.0, "gamma_1pct": 0.0, "vega_1vp": 0.0, "theta_1d": 0.0}
        L = legs[ok]
        b = black_sensitivities(L["cp"].to_numpy(), L["last_F"].to_numpy(), L["K"].to_numpy(), L["last_T"].to_numpy(),
                                L["last_iv"].to_numpy(), L["last_D"].to_numpy())
        q = L["qty"].to_numpy() * MULT
        f_s = L["last_F"].to_numpy() / S
        # position Greeks are named pos_* / *_1pct so they never collide with the attribution terms (delta, gamma...)
        return {"pos_delta": float(np.sum(q * b["dF"] * f_s)),                               # index units
                "gamma_1pct": float(np.sum(q * 0.5 * b["dFF"] * (f_s * S / 100.0) ** 2)),     # $ per (1% move)^2
                "vega_1vp": float(np.sum(q * b["ds"]) / 100.0),                                # $ per vol point
                "theta_1d": float(np.sum(-q * b["dtau"]) / 365.0)}                             # $ per calendar day

    @staticmethod
    def trade_record(pos: Position, legs: pd.DataFrame, date: pd.Timestamp, S: float) -> dict:
        return {"entry_date": pos.entry_date, "exit_date": date, "near_exdate": pos.near_exdate,
                "n_legs": len(legs), "premium": pos.premium, "margin": pos.margin, "costs": pos.costs,
                "pnl": pos.pnl, "S_exit": S,
                "legs": ";".join(f"{q:+g}{'C' if c > 0 else 'P'}{k:g}@{pd.Timestamp(e):%Y-%m-%d}"
                                 for q, c, k, e in zip(legs["qty"], legs["cp"], legs["K"], legs["exdate"]))}

    # -------------------------------------------------------------- driver
    def run(self, frames, days: pd.DatetimeIndex, progress=None) -> None:
        """frames: iterable of (year, chain DataFrame with store columns, forwards DataFrame) in date order."""
        for year, chain, fwd in frames:
            chain = chain.reset_index(drop=True)
            idx = chain.groupby("date").indices
            fidx = fwd.groupby("date").indices
            for date in sorted(idx):
                date = pd.Timestamp(date)
                day = chain.iloc[idx[date]]
                fday = fwd.iloc[fidx[date]] if date in fidx else fwd.iloc[0:0]
                by_id = pd.Series(day.index.to_numpy(), index=day["optionid"].to_numpy())
                by_id = by_id[~by_id.index.duplicated()]
                S = float(day["S"].iloc[0])
                for st in self.states:
                    self.step(st, day, by_id, fday, date, S, days)
            if progress:
                progress(year)

    def results(self) -> dict:
        out = {}
        for st in self.states:
            d = pd.DataFrame(st.daily)
            if len(d):
                d = d.set_index("date")
            out[st.strategy.name] = {"daily": d, "trades": pd.DataFrame(st.trades), "hedges": st.hedge_count,
                                     "strategy": st.strategy}
        return out
