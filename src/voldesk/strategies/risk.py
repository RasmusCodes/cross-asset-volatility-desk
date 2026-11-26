"""Portfolio-level risk: desk Greeks by position and in total, vega by expiry bucket, spot-vol scenario grids, and
historical-simulation P&L (VaR, expected shortfall, the worst days).

Historical scenarios. For each historical h-day window the scenario is the index log return x and the log change of
implied vol at five tenors (10, 30, 91, 182 and 365 days). ``tenor_scenarios`` takes any daily table of the index level
and vol levels at those tenors: in this project OptionMetrics' standardized at-the-money vols (stdopd) for 1996-2025,
or the VIX family (VIX9D, VIX, VIX3M, VIX6M, VIX1Y) where a caller prefers variance-swap rates. Missing tenors are filled
with the 30-day log change times a beta estimated by least squares through the origin on the windows where all five
exist. Applied to today's Market:
  the clock advances h days (Market.roll); forwards are multiplied by e^x; every implied vol is multiplied by
  exp(the log change at the option's tenor, interpolated linearly in days, flat beyond 10 and 365); and the smile
  either moves with the index ("moneyness") or stays put in strike ("strike").
This is a proportional, tenor-dependent surface move: the skew's own moves are not modelled by it (the demo's
historical scenarios add a separate 25-delta skew shock; see voldesk.web).

Two revaluations: "full" reprices every leg in every scenario; "taylor" uses each leg's second-order expansion in
forward, vol, time and discount factor (book.black_sensitivities), with the vol change taken from the same surface
move, so the gap between them is the cost of the Taylor approximation alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..pricing.bsm import black_price
from .book import IndexLeg, OptionLeg, Position, black_sensitivities
from .market import VOL_FLOOR, Market

TENOR_DAYS = np.array([10.0, 30.0, 91.0, 182.0, 365.0])
LV_COLS = [f"lv_{int(d)}" for d in TENOR_DAYS]
IV_COLS = [f"iv_{int(d)}" for d in TENOR_DAYS]


def risk_table(positions: list, mkt: Market) -> pd.DataFrame:
    """Desk Greeks (Position.risk) and values for each position and the total."""
    rows = []
    for p in positions:
        r = p.risk(mkt)
        rows.append({"position": p.name, "model_value": p.value(mkt), "mid_value": p.mid_value(mkt),
                     "execution_value": p.execution_value(mkt), **r})
    t = pd.DataFrame(rows).set_index("position")
    t.loc["total"] = t.sum(numeric_only=True)
    return t


def vega_buckets(pos: Position, mkt: Market, edges=(0, 14, 45, 100, 200, 400, 100000)) -> pd.Series:
    """Vega per vol point by days to expiry."""
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = Position("b", [g for g in pos.options if lo <= (g.T - mkt.t) * 365 < hi])
        out[f"{lo}-{hi if hi < 100000 else ''}d"] = sub.risk(mkt)["vega_1vp"] if sub.legs else 0.0
    return pd.Series(out)


def scenario_grid(pos: Position, mkt: Market, moves=(-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
                  vol_scales=(0.8, 0.9, 1.0, 1.25, 1.5), sticky: str = "moneyness", dt: float = 0.0) -> pd.DataFrame:
    """Full-revaluation P&L for simple index moves (columns) and proportional vol moves (rows), after dt years."""
    v0 = pos.value(mkt)
    base = mkt.roll(dt) if dt > 0 else mkt
    data = {}
    for m in moves:
        moved = base.move(np.log1p(m), sticky)
        data[f"{m:+.0%}"] = [pos.value(moved.scale_vol(s)) - v0 for s in vol_scales]
    return pd.DataFrame(data, index=[f"vol x{s:g}" for s in vol_scales])


def tenor_scenarios(levels: pd.DataFrame, horizon: int = 1, start: str | None = None) -> tuple[pd.DataFrame, dict]:
    """h-day scenarios (index log return x, log vol changes lv_* at the five tenors) from a daily table with columns
    'S' (index level) and IV_COLS (vol levels; missing tenors may be NaN or absent), plus the betas used to fill
    missing tenors."""
    d = levels.copy()
    d.index = pd.to_datetime(d.index)
    if start is not None:
        d = d.loc[start:]
    d = d[d["S"].notna()]
    for c in IV_COLS:
        if c not in d:
            d[c] = np.nan
    out = pd.DataFrame(index=d.index)
    out["x"] = np.log(d["S"] / d["S"].shift(horizon))
    for src, col in zip(IV_COLS, LV_COLS):
        out[col] = np.log(d[src] / d[src].shift(horizon))
    allfive = out[LV_COLS].notna().all(axis=1)
    betas = {}
    for col in LV_COLS:
        a, b = out.loc[allfive, col].to_numpy(), out.loc[allfive, "lv_30"].to_numpy()
        betas[col] = float(np.dot(a, b) / np.dot(b, b))
        out[f"{col}_filled"] = out[col].isna() & out["lv_30"].notna()
        out[col] = out[col].fillna(betas[col] * out["lv_30"])
    out = out.dropna(subset=["x"] + LV_COLS)
    five = allfive.reindex(out.index, fill_value=False).to_numpy()
    return out, {"betas": betas, "n_all_five": int(five.sum()), "first_all_five": str(out.index[five][0].date())}


def _tenor_interp(L: np.ndarray, days: float) -> np.ndarray:
    """Per-scenario log vol change at a tenor in days (rows of L hold the five tenor changes)."""
    d = float(np.clip(days, TENOR_DAYS[0], TENOR_DAYS[-1]))
    j = int(min(np.searchsorted(TENOR_DAYS, d, side="right") - 1, len(TENOR_DAYS) - 2))
    w = (d - TENOR_DAYS[j]) / (TENOR_DAYS[j + 1] - TENOR_DAYS[j])
    return (1 - w) * L[:, j] + w * L[:, j + 1]


def scenario_pnl(pos: Position, mkt: Market, scen: pd.DataFrame, sticky: str = "moneyness", horizon_days: float = 1.0,
                 method: str = "full") -> np.ndarray:
    """P&L of the position in each scenario (rows of scen: x and the lv_* columns)."""
    x = scen["x"].to_numpy(dtype=float)
    L = scen[LV_COLS].to_numpy(dtype=float)
    dt = horizon_days / 365.0
    mr = mkt.roll(dt)
    pnl = np.zeros(x.size)
    for g in pos.legs:
        if isinstance(g, IndexLeg):
            pnl += g.qty * mkt.spot * np.expm1(x)
            continue
        t0, t1 = g.T - mkt.t, g.T - mr.t
        k = g.qty * g.mult
        F0, D0, s0 = (float(v) for v in (mkt.forward(t0), mkt.discount(t0), mkt.iv(g.K, t0)))
        v0 = float(black_price(g.cp, F0, g.K, t0, s0, D0))
        if t1 <= 0:
            pnl += k * (np.maximum(g.cp * (mkt.spot * np.exp(x) - g.K), 0.0) - v0)
            continue
        F1 = float(mr.forward(t1)) * np.exp(x)
        D1 = float(mr.discount(t1))
        kk = np.log(g.K / F1)
        base = mr.vol_fn(kk if sticky == "moneyness" else kk + x, np.full(x.shape, t1))
        s1 = np.maximum(base * np.exp(_tenor_interp(L, t1 * 365.0)), VOL_FLOOR)
        if method == "full":
            pnl += k * (np.asarray(black_price(g.cp, F1, g.K, t1, s1, D1)) - v0)
        elif method == "taylor":
            b = black_sensitivities(g.cp, F0, g.K, t0, s0, D0)
            dF, ds, dD = F1 - F0, s1 - s0, D1 - D0
            pnl += k * (b["dF"] * dF + 0.5 * b["dFF"] * dF ** 2 + b["ds"] * ds + b["dFs"] * dF * ds
                        + 0.5 * b["dss"] * ds ** 2 + b["dtau"] * (t1 - t0) + b["dD"] * dD)
        else:
            raise ValueError("method must be 'full' or 'taylor'")
    return pnl


def var_es(pnl, var_level: float = 0.99, es_level: float = 0.975) -> dict:
    """Historical VaR and expected shortfall, as positive losses."""
    loss = -np.asarray(pnl, dtype=float)
    q_es = np.quantile(loss, es_level)
    return {"var": float(np.quantile(loss, var_level)), "es": float(loss[loss >= q_es].mean()),
            "worst": float(loss.max()), "mean": float(-loss.mean()), "n": int(loss.size)}
