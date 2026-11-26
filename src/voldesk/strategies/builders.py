"""Strategy constructors: the brief's option structures, built on a Market. On a stored close they use listed strikes
and expiries (two-sided quotes only); on a simulated Market, exact strikes and maturities.

Conventions:
  expiry   the listed expiry (SPXW or SPX) whose time to expiry is closest to the target number of calendar days,
           among expiries with at least MIN_STRIKES listed strikes
  strike   by delta: the listed strike whose forward delta at the surface's vol is closest to the target (N(d1) for
           calls, N(-d1) for puts); at the money: the listed strike nearest the forward
  qty      +1 builds the structure as named (a long straddle, a short iron condor, a long calendar, ...); a negative
           qty reverses every leg

Structures (qty = +1):
  straddle        long ATM call and put
  strangle        long put and call at +-delta
  vertical        credit or debit spread: short the short_delta option, long the long_delta option (same type)
  iron_condor     short put and call at short_delta, long put and call at long_delta (a credit)
  calendar        short the near expiry, long the far expiry, same strike (the near forward's ATM strike)
  diagonal        short the near expiry at near_delta, long the far expiry at far_delta (same type)
  risk_reversal   long the call and short the put at the same delta
  delta_hedged    any position plus the index hedge that makes its delta zero (delta-neutral trading)
  vol_arb         a delta-hedged ATM straddle, long when a forecast of realized vol exceeds the straddle's implied vol
                  by more than a band, short when below it
Skew trading uses risk reversals: buying the reversal (qty +1) sells the expensive downside and gains if the skew
flattens; selling it (qty -1) gains if the skew steepens. Gamma scalping and dispersion are dynamic and live in
strategies.hedging_sim.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .book import IndexLeg, OptionLeg, Position
from .market import Market

MIN_STRIKES = 20


def expiry_near(mkt: Market, days: float, roots=None) -> tuple[float, str, str]:
    """(absolute T, root, expiration) of the listed expiry nearest the target, or the exact maturity in a simulation."""
    if mkt.quotes is None:
        return mkt.t + days / 365.0, "", ""
    ex = mkt.listed_expiries()
    ex = ex[ex["n"] >= MIN_STRIKES]
    if roots is not None:
        ex = ex[ex["root"].isin(roots)]
    i = int(np.argmin(np.abs(ex["tau"].to_numpy() - days / 365.0)))
    r = ex.iloc[i]
    return float(r["T"]), str(r["root"]), str(r["expiration"])


def _listed(mkt: Market, root: str, expiration: str, cp: int | None = None) -> np.ndarray:
    q = mkt.quotes
    m = (q["root"] == root) & (q["expiration"] == expiration)
    if cp is not None:
        m &= q["cp"] == cp
    return np.unique(q.loc[m, "strike"].to_numpy(dtype=float))


def fwd_delta(mkt: Market, cp: int, K, tau: float) -> np.ndarray:
    """Size of the forward (undiscounted) Black delta at the surface's vol: N(d1) for calls, N(-d1) for puts."""
    K = np.asarray(K, dtype=float)
    F = float(mkt.forward(tau))
    v = mkt.iv(K, tau) * np.sqrt(tau)
    d1 = np.log(F / K) / v + 0.5 * v
    return ndtr(cp * d1)


def strike_for_delta(mkt: Market, cp: int, T: float, target: float, root: str = "", expiration: str = "") -> float:
    tau = T - mkt.t
    if mkt.quotes is not None and root:
        Ks = _listed(mkt, root, expiration, cp)
        return float(Ks[int(np.argmin(np.abs(fwd_delta(mkt, cp, Ks, tau) - target)))])
    F = float(mkt.forward(tau))
    v = float(mkt.iv(F, tau)) * np.sqrt(tau)                  # ATM standard deviation: bracket within +-10 of them
    k = brentq(lambda k: float(fwd_delta(mkt, cp, F * np.exp(k), tau)) - target, -10 * v, 10 * v, xtol=1e-12)
    return float(F * np.exp(k))


def strike_atm(mkt: Market, T: float, root: str = "", expiration: str = "", also: tuple | None = None) -> float:
    """Listed strike (both types listed; also listed at a second expiry when also = (root, expiration)) nearest the
    forward, or the forward itself in a simulation."""
    F = float(mkt.forward(T - mkt.t))
    if mkt.quotes is None or not root:
        return F
    Ks = np.intersect1d(_listed(mkt, root, expiration, 1), _listed(mkt, root, expiration, -1))
    if also is not None:
        Ks = np.intersect1d(Ks, _listed(mkt, also[0], also[1], None))
    return float(Ks[int(np.argmin(np.abs(Ks - F)))])


def _side(qty: float) -> str:
    return "long" if qty > 0 else "short"


def straddle(mkt: Market, days: float = 30, qty: float = 1.0) -> Position:
    T, root, exp = expiry_near(mkt, days)
    K = strike_atm(mkt, T, root, exp)
    return Position(f"{_side(qty)} {days:g}d straddle", [OptionLeg(1, K, T, qty, 100.0, root, exp), OptionLeg(-1, K, T, qty, 100.0, root, exp)],
                    {"strategy": "straddle", "days": days})


def strangle(mkt: Market, days: float = 30, delta: float = 0.25, qty: float = 1.0) -> Position:
    T, root, exp = expiry_near(mkt, days)
    Kp, Kc = (strike_for_delta(mkt, cp, T, delta, root, exp) for cp in (-1, 1))
    return Position(f"{_side(qty)} {days:g}d {delta:.0%} strangle", [OptionLeg(-1, Kp, T, qty, 100.0, root, exp), OptionLeg(1, Kc, T, qty, 100.0, root, exp)],
                    {"strategy": "strangle", "days": days, "delta": delta})


def vertical(mkt: Market, days: float = 30, cp: int = -1, short_delta: float = 0.25, long_delta: float = 0.10,
             qty: float = 1.0) -> Position:
    T, root, exp = expiry_near(mkt, days)
    Ks, Kl = (strike_for_delta(mkt, cp, T, d, root, exp) for d in (short_delta, long_delta))
    kind = "credit" if short_delta > long_delta else "debit"
    return Position(f"{'put' if cp < 0 else 'call'} {kind} spread {short_delta:.0%}/{long_delta:.0%}",
                    [OptionLeg(cp, Ks, T, -qty, 100.0, root, exp), OptionLeg(cp, Kl, T, qty, 100.0, root, exp)],
                    {"strategy": "vertical", "days": days})


def iron_condor(mkt: Market, days: float = 30, short_delta: float = 0.16, long_delta: float = 0.05, qty: float = 1.0) -> Position:
    T, root, exp = expiry_near(mkt, days)
    legs = []
    for cp in (-1, 1):
        Ks, Kl = (strike_for_delta(mkt, cp, T, d, root, exp) for d in (short_delta, long_delta))
        legs += [OptionLeg(cp, Ks, T, -qty, 100.0, root, exp), OptionLeg(cp, Kl, T, qty, 100.0, root, exp)]
    return Position(f"{'short' if qty > 0 else 'long'} iron condor {short_delta:.0%}/{long_delta:.0%}", legs,
                    {"strategy": "iron_condor", "days": days})


def calendar(mkt: Market, near_days: float = 30, far_days: float = 60, cp: int = 1, qty: float = 1.0) -> Position:
    Tn, rn, en = expiry_near(mkt, near_days)
    Tf, rf, ef = expiry_near(mkt, far_days)
    K = strike_atm(mkt, Tn, rn, en, also=(rf, ef) if rf else None)
    return Position(f"{_side(qty)} {'call' if cp > 0 else 'put'} calendar {near_days:g}/{far_days:g}d",
                    [OptionLeg(cp, K, Tn, -qty, 100.0, rn, en), OptionLeg(cp, K, Tf, qty, 100.0, rf, ef)],
                    {"strategy": "calendar"})


def diagonal(mkt: Market, near_days: float = 30, far_days: float = 60, cp: int = 1, near_delta: float = 0.25,
             far_delta: float = 0.40, qty: float = 1.0) -> Position:
    Tn, rn, en = expiry_near(mkt, near_days)
    Tf, rf, ef = expiry_near(mkt, far_days)
    Kn = strike_for_delta(mkt, cp, Tn, near_delta, rn, en)
    Kf = strike_for_delta(mkt, cp, Tf, far_delta, rf, ef)
    return Position(f"{_side(qty)} {'call' if cp > 0 else 'put'} diagonal {near_days:g}d {near_delta:.0%} / {far_days:g}d {far_delta:.0%}",
                    [OptionLeg(cp, Kn, Tn, -qty, 100.0, rn, en), OptionLeg(cp, Kf, Tf, qty, 100.0, rf, ef)],
                    {"strategy": "diagonal"})


def risk_reversal(mkt: Market, days: float = 30, delta: float = 0.25, qty: float = 1.0) -> Position:
    T, root, exp = expiry_near(mkt, days)
    Kp, Kc = (strike_for_delta(mkt, cp, T, delta, root, exp) for cp in (-1, 1))
    return Position(f"{_side(qty)} {days:g}d {delta:.0%} risk reversal",
                    [OptionLeg(1, Kc, T, qty, 100.0, root, exp), OptionLeg(-1, Kp, T, -qty, 100.0, root, exp)],
                    {"strategy": "risk_reversal", "days": days, "delta": delta})


def delta_hedged(pos: Position, mkt: Market, name: str | None = None) -> Position:
    """The position plus an index leg that makes its delta (in index units) zero at this Market."""
    d = pos.greeks(mkt)["delta"]
    return pos.with_legs([IndexLeg(-d, mkt.spot)], name or f"{pos.name}, delta-hedged")


def vol_arb(mkt: Market, forecast_vol: float, days: float = 30, band: float = 0.0, qty: float = 1.0) -> Position:
    """Delta-hedged ATM straddle: long if forecast realized vol > the straddle's implied vol + band, short if below
    it minus band, empty otherwise. Vols as decimals."""
    base = straddle(mkt, days, 1.0)
    leg = base.legs[0]
    iv = float(mkt.iv(leg.K, leg.T - mkt.t))
    side = 1.0 if forecast_vol > iv + band else (-1.0 if forecast_vol < iv - band else 0.0)
    meta = {"strategy": "vol_arb", "implied": iv, "forecast": forecast_vol, "side": side}
    if side == 0.0:
        return Position("vol arb: no trade", [], meta)
    pos = delta_hedged(base.scaled(side * qty), mkt)
    return Position(f"vol arb: {_side(side)} straddle (forecast {forecast_vol:.1%} vs implied {iv:.1%})", pos.legs, meta)
