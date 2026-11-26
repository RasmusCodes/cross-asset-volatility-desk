"""Data bundle for the interactive demo (demo/data/). Derived results only: fitted model parameters, fit statistics,
scenario shocks aggregated from standardized surfaces, and backtest equity curves. No vendor quote, implied vol or
price series is written (the raw data are licensed and stay local).

Files
  manifest.json                 underlyings, dates, sources, generation time, repository link
  surfaces/<UND>/<date>.json    per date: spot, expiries (T, forward, discount, quoted k range, ATM vol summary),
                                every model's parameters and per-expiry fit statistics, arbitrage diagnostics
  scenarios.json                historical episodes: index return and ATM-vol / skew changes by tenor (SPX), plus
                                clearly labelled hypothetical stresses
  backtests.json                strategy library: weekly cumulative P&L per unit notional (base costs), summary
                                statistics under three cost models, attribution totals
  models.json                   panel-wide model comparison (medians by maturity bucket and era)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROUND = 6


def _r(x, nd: int = ROUND):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        return float(f"{float(x):.{nd}g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def surface_bundle(rec: dict, underlying: str, source: str) -> dict:
    """Compact demo JSON from a cached panel record (scripts/surface_panel.py)."""
    exps = [{"expiry": s["expiry"], "T": s["T"], "F": s["F"], "D": s["D"], "k_min": s["k_min"], "k_max": s["k_max"],
             "atm_iv": s["atm_iv"], "n": s["n"], "spread_med": s["spread_med"]} for s in rec["slices"]]
    models = {}
    for m, mr in rec.get("models", {}).items():
        if not mr.get("slices"):
            continue
        fit = [{k: s.get(k) for k in ("T", "expiry", "n", "rmse", "mae", "inside", "max_abs")} for s in mr["slices"]]
        entry = {"fit": fit, "arbitrage": mr.get("arbitrage", {}), "seconds": mr.get("seconds"),
                 "success": mr.get("success")}
        p = mr.get("params", {})
        if m in ("svi", "essvi"):
            entry["slices"] = p.get("slices", [])
        elif m == "sabr":
            entry["slices"] = p.get("slices", [])
        elif m == "ssvi":
            entry.update({k: p.get(k) for k in ("rho", "eta", "gamma", "Ts", "thetas")})
        elif m == "heston":
            entry.update({k: p.get(k) for k in ("v0", "kappa", "theta", "sigma", "rho", "feller")})
        models[m] = entry
    return _r({"underlying": underlying, "date": rec["date"], "source": source, "spot": rec.get("spot"),
               "expiries": exps, "models": models})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ scenarios
EPISODES = [
    ("ltcm", "LTCM and the Russian default", "1998-07-17", "1998-10-08"),
    ("sep11", "Markets reopen after 9/11", "2001-09-10", "2001-09-21"),
    ("bear2002", "2002 bear-market capitulation", "2002-07-01", "2002-07-23"),
    ("lehman", "Lehman Brothers collapse", "2008-09-12", "2008-10-10"),
    ("flash2010", "Flash crash", "2010-05-05", "2010-05-07"),
    ("downgrade", "US credit downgrade", "2011-07-22", "2011-08-08"),
    ("china2015", "Yuan devaluation selloff", "2015-08-18", "2015-08-24"),
    ("brexit", "Brexit referendum", "2016-06-23", "2016-06-27"),
    ("volmageddon", "Volmageddon (short-vol ETP blowup)", "2018-02-01", "2018-02-05"),
    ("q4_2018", "December 2018 selloff", "2018-12-03", "2018-12-24"),
    ("covid", "COVID-19 crash", "2020-02-19", "2020-03-16"),
    ("covid_rebound", "Post-COVID rally and vol crush", "2020-03-23", "2020-06-08"),
    ("bear2022", "2022 inflation bear market", "2022-01-03", "2022-06-16"),
    ("yen_carry", "Yen carry-trade unwind", "2024-07-16", "2024-08-05"),
    ("tariffs2025", "April 2025 tariff shock", "2025-04-02", "2025-04-08"),
]
HYPOTHETICAL = [
    ("hyp_crash", "Hypothetical: -20% crash, vols +30 points", -0.20, {10: 0.40, 30: 0.30, 91: 0.20, 182: 0.15, 365: 0.10}, 0.04),
    ("hyp_selloff", "Hypothetical: -10% selloff, vols +10 points", -0.10, {10: 0.14, 30: 0.10, 91: 0.07, 182: 0.05, 365: 0.04}, 0.015),
    ("hyp_melt_up", "Hypothetical: +8% rally, vols -4 points", 0.08, {10: -0.05, 30: -0.04, 91: -0.03, 182: -0.025, 365: -0.02}, -0.005),
]


def scenario_table(spx_close: pd.Series, std_atm: pd.DataFrame, skew: pd.DataFrame) -> list[dict]:
    """std_atm: index dates, columns tenors (days) of ATM implied vol (decimal); skew: index dates, columns
    'skew30', 'skew91' = 25-delta put vol minus 50-delta put vol (decimal)."""
    out = []
    for sid, name, a, b in EPISODES:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a not in spx_close.index or b not in spx_close.index:
            continue
        rec = {"id": sid, "name": name, "start": f"{a:%Y-%m-%d}", "end": f"{b:%Y-%m-%d}",
               "days": int((b - a).days), "spot_return": float(spx_close[b] / spx_close[a] - 1), "kind": "historical"}
        if a in std_atm.index and b in std_atm.index:
            # changes only: the vendor's vol levels themselves are not published
            rec["atm_change"] = {str(int(t)): float(std_atm.at[b, t] - std_atm.at[a, t]) for t in std_atm.columns}
        if a in skew.index and b in skew.index:
            rec["skew_change"] = {c: float(skew.at[b, c] - skew.at[a, c]) for c in skew.columns}
        out.append(rec)
    for sid, name, ret, atm, sk in HYPOTHETICAL:
        out.append({"id": sid, "name": name, "start": None, "end": None, "days": 1, "spot_return": ret,
                    "atm_change": {str(k): v for k, v in atm.items()}, "skew_change": {"skew30": sk, "skew91": sk / 2},
                    "kind": "hypothetical"})
    return _r(out)
