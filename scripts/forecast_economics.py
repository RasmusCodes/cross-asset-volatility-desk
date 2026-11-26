"""Do the forecasts make money after costs? Out-of-sample economic tests of scripts/forecast_study.py's forecasts.

    python scripts/forecast_economics.py

1. Volatility targeting the S&P 500 (daily): exposure w_t = min(15% / forecast vol of the next 22 days, 2), applied
   to the next day's index return, 1 bp per unit of exposure traded. Forecasts compared: EWMA, GJR, log-HAR(Parkinson)
   + VIX, the VIX itself; benchmarks buy-and-hold and constant exposure.
2. Variance-premium timing (monthly cycles of the backtested strategies, base costs): at each entry the forecast
   premium is VIX^2 - forecast realized variance (log-HAR + VIX, 22 days). Rules: trade only when it is positive;
   size by it (premium / its trailing median, capped to [0, 2]).
3. Long/short switching: each cycle, the delta-hedged long straddle when forecast realized variance exceeds VIX^2,
   else the delta-hedged short straddle.
4. Regime de-risking: skip the short-volatility cycle when the forecast probability that the next 22 days' realized
   vol exceeds 25% is above 50% (logistic on log VIX, walk-forward).
Every rule uses only information at the entry close. Sharpe differences come with a cycle-level bootstrap interval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.forecast import datasets as dsets  # noqa: E402

OUT = config.RESULTS / "forecast"
BT = config.CACHE / "backtests" / "run"
FC = config.CACHE / "forecast"


def sharpe(x, periods):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(periods)) if x.std() > 0 else np.nan


def boot_diff(a, b, periods, reps=5000, seed=0):
    """Bootstrap (iid over cycles, paired) interval of Sharpe(a) - Sharpe(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        sa, sb = a[i].std(), b[i].std()
        if sa > 0 and sb > 0:
            d.append((a[i].mean() / sa - b[i].mean() / sb) * np.sqrt(periods))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def vol_targeting() -> pd.DataFrame:
    rv = pd.read_parquet(FC / "rv.parquet").xs(22, level="h")
    d = dsets.spx_daily()
    r_next = d["r"].shift(-1)
    rows = []
    for name in ["EWMA", "GJR", "logHAR_pk_VIX", "VIX_raw"]:
        sig = np.sqrt(rv[name] * 252)
        w = np.minimum(0.15 / sig, 2.0)
        df = pd.DataFrame({"w": w, "r": r_next}).dropna()
        df = df[df.index >= "2000-01-03"]
        turnover = df["w"].diff().abs().fillna(0)
        p = df["w"] * df["r"] - 1e-4 * turnover
        roll = p.rolling(63).std() * np.sqrt(252)
        rows.append({"rule": f"target 15% with {name}", "ann_return": p.mean() * 252, "ann_vol": p.std() * np.sqrt(252),
                     "sharpe": sharpe(p, 252), "max_dd": float((p.cumsum() - p.cumsum().cummax()).min()),
                     "turnover_per_year": turnover.mean() * 252, "rolling_vol_rmse_vs_target": float(np.sqrt(((roll - 0.15) ** 2).mean())),
                     "avg_exposure": df["w"].mean()})
    bh = r_next[r_next.index >= "2000-01-03"].dropna()
    rows.append({"rule": "buy and hold", "ann_return": bh.mean() * 252, "ann_vol": bh.std() * np.sqrt(252),
                 "sharpe": sharpe(bh, 252), "max_dd": float((bh.cumsum() - bh.cumsum().cummax()).min()),
                 "turnover_per_year": 0.0,
                 "rolling_vol_rmse_vs_target": float(np.sqrt(((bh.rolling(63).std() * np.sqrt(252) - 0.15) ** 2).mean())),
                 "avg_exposure": 1.0})
    return pd.DataFrame(rows).set_index("rule")


def cycles(name: str) -> pd.DataFrame:
    t = pd.read_csv(BT / f"{name}_trades.csv", parse_dates=["entry_date", "exit_date"])
    d = pd.read_parquet(BT / f"{name}_daily.parquet")
    S = d["S"]
    t["S_entry"] = t["entry_date"].map(S)
    t["ret"] = t["pnl"] / (t["S_entry"] * 100.0)
    return t.set_index("entry_date")


def timing() -> tuple[pd.DataFrame, pd.DataFrame]:
    rv = pd.read_parquet(FC / "rv.parquet").xs(22, level="h")
    reg = pd.read_parquet(FC / "regime.parquet")
    vix = dsets.vix_daily()["vix"]
    f_rv = rv["logHAR_pk_VIX"]                              # forecast mean daily variance, next 22 days
    iv2 = (vix / 100) ** 2 / 252
    vrp = (iv2 - f_rv).rename("vrp")
    p_high = reg["logit_vix"].rename("p_high")
    rows, series = [], {}
    for strat in ["short_straddle_dh", "short_straddle", "short_put_25_dh", "short_strangle_25", "short_var_strip_dh"]:
        c = cycles(strat)
        c = c.join(vrp, how="left").join(p_high, how="left")
        c = c[(c.index >= "2000-01-03") & c["vrp"].notna() & c["p_high"].notna()]
        base = c["ret"]
        med = c["vrp"].rolling(36, min_periods=12).median().shift(1)
        size = (c["vrp"] / med).clip(0, 2).where(med > 0).fillna(1.0)
        rules = {"always on": base, "only if forecast premium > 0": base.where(c["vrp"] > 0, 0.0),
                 "sized by forecast premium": base * size, "skip if P(high vol) > 50%": base.where(c["p_high"] <= 0.5, 0.0)}
        for rn, x in rules.items():
            lo, hi = boot_diff(x.to_numpy(), base.to_numpy(), 12) if rn != "always on" else (np.nan, np.nan)
            rows.append({"strategy": strat, "rule": rn, "cycles": int(len(x)), "share_traded": float((x != 0).mean()),
                         "mean_cycle_return": float(x.mean()), "ann_return": float(x.mean() * 12),
                         "sharpe_ann": sharpe(x, 12), "worst_cycle": float(x.min()),
                         "max_dd": float((x.cumsum() - x.cumsum().cummax()).min()),
                         "sharpe_diff_ci_low": lo, "sharpe_diff_ci_high": hi})
            series[(strat, rn)] = x
    # long/short switching on hedged straddles
    s = cycles("short_straddle_dh")["ret"]
    lg = cycles("long_straddle_dh")["ret"]
    j = pd.DataFrame({"short": s, "long": lg}).join(vrp, how="left").dropna()
    j = j[j.index >= "2000-01-03"]
    sw = np.where(j["vrp"] < 0, j["long"], j["short"])
    lo, hi = boot_diff(sw, j["short"].to_numpy(), 12)
    rows.append({"strategy": "hedged straddle long/short", "rule": "long if forecast RV > VIX^2, else short",
                 "cycles": int(len(j)), "share_traded": 1.0, "mean_cycle_return": float(np.mean(sw)),
                 "ann_return": float(np.mean(sw) * 12), "sharpe_ann": sharpe(sw, 12), "worst_cycle": float(np.min(sw)),
                 "max_dd": float((pd.Series(sw).cumsum() - pd.Series(sw).cumsum().cummax()).min()),
                 "sharpe_diff_ci_low": lo, "sharpe_diff_ci_high": hi, "share_long": float((j["vrp"] < 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(series)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vt = vol_targeting()
    vt.to_csv(OUT / "econ_vol_targeting.csv", float_format="%.5g")
    print(vt.round(3).to_string())
    tab, ser = timing()
    tab.to_csv(OUT / "econ_timing.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
