"""Performance statistics for backtested option strategies.

Scaling. Daily P&L is in dollars for one strategy unit (contracts of multiplier 100). Two normalisations:
  per notional   P&L / (S x 100), the previous close's index level times the multiplier: the return on the index
                 notional of one strategy unit (the same denominator for every structure, so a fractional-weight
                 strip and a 1x2x1 butterfly are both measured per unit); annualised mean and volatility of this
                 daily series (252 days); drawdowns on its cumulative (non-compounded) sum
  per margin     each trade's P&L / its strategy-based margin at entry (return on the capital the exchange rule
                 requires), averaged per trade and annualised by the trades' average length
Sharpe ratios use the notional series (scale-free) without a risk-free deduction; option P&L is already an excess
return because premia are paid from, and hedges are financed at, the short rate embedded in forwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATTR = ("delta", "gamma", "vega", "vanna", "volga", "theta", "rates", "residual", "hedge")


def with_cost_model(daily: pd.DataFrame, cost_model) -> pd.DataFrame:
    """The daily frame with costs and P&L recomputed for another cost model from the recorded components."""
    d = daily.copy()
    if not len(d) or "spread_full" not in d:
        return d
    d["costs"] = cost_model.from_components(d["spread_full"], d["contracts"], d["hedge_notional"])
    d["pnl"] = d["mtm"] + d["hedge"] - d["costs"]
    return d


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    return float((cum - peak).min())


def summarize(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if daily is None or daily.empty:
        return {}
    d = daily.copy()
    ret = unit_returns(d)
    years = len(d) / 252.0
    gross = d["pnl"] + d["costs"]
    out = {"start": str(d.index.min().date()), "end": str(d.index.max().date()), "days": int(len(d)),
           "total_pnl": float(d["pnl"].sum()), "total_costs": float(d["costs"].sum()),
           "ann_return_notional": float(ret.mean() * 252), "ann_vol_notional": float(ret.std() * np.sqrt(252)),
           "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan,
           "skew_daily": float(ret.skew()), "kurt_daily": float(ret.kurt()),
           "worst_day_notional": float(ret.min()), "best_day_notional": float(ret.max()),
           "max_drawdown_notional": max_drawdown(ret.cumsum()),
           "cost_share_of_gross": float(d["costs"].sum() / abs(gross.sum())) if abs(gross.sum()) > 0 else np.nan,
           "missing_mark_days": int((d.get("missing_marks", 0) > 0).sum()),
           "pct_days_invested": float((d.get("n_legs", 0) > 0).mean())}
    tot = {a: float(d[a].sum()) for a in ATTR if a in d}
    tot["costs"] = -float(d["costs"].sum())
    for k, v in tot.items():
        out[f"attr_{k}"] = v
    if trades is not None and len(trades):
        t = trades.copy()
        t["ret_margin"] = t["pnl"] / t["margin"]
        span = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days.mean()
        out.update({"n_trades": int(len(t)), "win_rate": float((t["pnl"] > 0).mean()),
                    "avg_trade_pnl": float(t["pnl"].mean()), "worst_trade_pnl": float(t["pnl"].min()),
                    "median_margin": float(t["margin"].median()),
                    "avg_return_on_margin": float(t["ret_margin"].mean()),
                    "ann_return_on_margin": float(t["ret_margin"].mean() * 365.0 / span) if span > 0 else np.nan,
                    "worst_trade_on_margin": float(t["ret_margin"].min())})
    return out


def unit_returns(daily: pd.DataFrame, multiplier: float = 100.0) -> pd.Series:
    """Daily P&L divided by the previous close's index notional of one unit (S x 100)."""
    notional = (daily["S"] * multiplier).shift(1)
    notional = notional.fillna(daily["S"] * multiplier)
    return (daily["pnl"] / notional).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def yearly(daily: pd.DataFrame) -> pd.Series:
    ret = unit_returns(daily)
    return ret.groupby(daily.index.year).sum()
