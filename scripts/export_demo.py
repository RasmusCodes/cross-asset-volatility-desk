"""Write the interactive demo's data bundle (demo/data/) from cached results. Derived numbers only.

    python scripts/export_demo.py

Reads: data/cache/surface_panel/{monthend,dates,etf}/*.json, data/raw/optionm/{secprd,stdopd_panel,vsurfd_panel},
data/cache/backtests/run/*, results/surfaces/monthend_*.csv. Writes demo/data/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.backtest.metrics import unit_returns  # noqa: E402
from voldesk.data import catalog, store  # noqa: E402
from voldesk.web import export as ex  # noqa: E402

REPO = "https://github.com/RasmusCodes/cross-asset-volatility-desk"
SPX_FEATURED = ["2026-11-03", "2025-08-29", "2025-04-08", "2024-08-05", "2022-10-12", "2020-03-16", "2018-02-05",
                "2017-06-30", "2015-08-24", "2011-08-08", "2008-10-31", "2008-10-10", "2003-03-11", "2000-03-31"]
SOURCE = {"spx": "OptionMetrics IvyDB US via WRDS", "spx_cboe": "Cboe option data via WRDS"}


def load_rec(date: str, sets=("dates", "monthend")) -> dict | None:
    for s in sets:
        p = config.CACHE / "surface_panel" / s / f"{date}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def surfaces(out: Path) -> dict:
    dates = {}
    for d in SPX_FEATURED:
        rec = load_rec(d)
        if rec is None or not rec.get("models"):
            print(f"  SPX {d}: no fit cached (run scripts/surface_panel.py --set dates --dates {d})")
            continue
        b = ex.surface_bundle(rec, "SPX", SOURCE.get(rec.get("source", "spx"), rec.get("source", "")))
        ex.write_json(out / "surfaces" / "SPX" / f"{d}.json", b)
        dates.setdefault("SPX", []).append(d)
    etf_dir = config.CACHE / "surface_panel" / "etf"
    for f in sorted(etf_dir.glob("*.json")) if etf_dir.exists() else []:
        rec = json.loads(f.read_text(encoding="utf-8"))
        und = rec.get("underlying", f.stem.split("_")[0])
        if not rec.get("models"):
            continue
        b = ex.surface_bundle(rec, und, SOURCE["spx"])
        ex.write_json(out / "surfaces" / und / f"{rec['date']}.json", b)
        dates.setdefault(und, []).append(rec["date"])
    return {k: sorted(v, reverse=True) for k, v in dates.items()}


def scenarios(out: Path) -> None:
    spx = catalog.SPX
    px = store.secprd([spx], columns=["secid", "date", "close"]).set_index("date")["close"].sort_index()
    std = store.read_dir("optionm/stdopd_panel", columns=["secid", "date", "days", "impl_volatility", "cp_flag"])
    std = std[std["secid"] == spx]
    std["date"] = pd.to_datetime(std["date"])
    atm = std.groupby(["date", "days"])["impl_volatility"].mean().unstack()[[10, 30, 91, 182, 365]]
    vs = store.read_dir("optionm/vsurfd_panel", columns=["secid", "date", "days", "delta", "impl_volatility"])
    vs = vs[(vs["secid"] == spx) & vs["delta"].isin([-25, -50])]
    vs["date"] = pd.to_datetime(vs["date"])
    p = vs.pivot_table(index="date", columns=["days", "delta"], values="impl_volatility")
    skew = pd.DataFrame({"skew30": p[(30, -25)] - p[(30, -50)], "skew91": p[(91, -25)] - p[(91, -50)]})
    tab = ex.scenario_table(px, atm, skew)
    ex.write_json(out / "scenarios.json", {"source": "SPX close (OptionMetrics secprd); ATM implied vol = mean of "
                                           "OptionMetrics standardized 10/30/91/182/365-day call and put; skew = 25-delta "
                                           "minus 50-delta put vol from the standardized surface",
                                           "scenarios": tab})
    print(f"  scenarios: {len(tab)}")


def backtests(out: Path) -> None:
    src = config.CACHE / "backtests" / "run"
    summ = {}
    for c in ("base", "stress", "frictionless"):
        p = config.RESULTS / "backtests" / f"summary_{c}.csv"
        if p.exists():
            summ[c] = pd.read_csv(p).set_index("strategy")
    if not src.exists() or "base" not in summ:
        print("  backtests: not run yet")
        return
    strategies, weekly_index = [], None
    for f in sorted(src.glob("*_daily.parquet")):
        name = f.name[: -len("_daily.parquet")]
        d = pd.read_parquet(f)
        ret = unit_returns(d)                      # P&L / (previous close x 100), base costs
        cum = ret.cumsum()
        wk = cum.resample("W-FRI").last().dropna()
        weekly_index = wk.index if weekly_index is None or len(wk.index) > len(weekly_index) else weekly_index
        row = summ["base"].loc[name] if name in summ["base"].index else None
        attr = {k[5:]: row[k] for k in row.index if k.startswith("attr_")} if row is not None else {}
        stats = {c: {k: summ[c].loc[name, k] for k in ("ann_return_notional", "ann_vol_notional", "sharpe",
                                                      "max_drawdown_notional", "worst_day_notional", "skew_daily")
                     if name in summ[c].index} for c in summ}
        extra = {k: row[k] for k in ("n_trades", "win_rate", "median_margin", "avg_return_on_margin",
                                     "worst_trade_on_margin", "cost_share_of_gross") if row is not None and k in row}
        strategies.append({"name": name, "group": row["group"] if row is not None else "",
                           "description": row["description"] if row is not None else "",
                           "hedge": row["hedge"] if row is not None else "", "stats": stats, "attribution": attr,
                           "trades": extra, "weekly": wk})
    idx = weekly_index
    for s in strategies:
        s["weekly"] = s["weekly"].reindex(idx).ffill().fillna(0.0).round(5).tolist()
    ex.write_json(out / "backtests.json", ex._r({"dates": [f"{d:%Y-%m-%d}" for d in idx], "strategies": strategies,
                                                  "units": "cumulative P&L per $1 of underlying notional, base costs"}))
    print(f"  backtests: {len(strategies)} strategies, {len(idx)} weeks")


def models(out: Path) -> None:
    p = config.RESULTS / "surfaces" / "monthend_dates.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    s = pd.read_csv(config.RESULTS / "surfaces" / "monthend_slices.csv")
    s["bucket"] = pd.cut(s["T"] * 365, [0, 30, 91, 365, 1200], labels=["<=30d", "31-91d", "92-365d", ">1y"])
    by_bucket = s.pivot_table(index="bucket", columns="model", values="rmse", aggfunc="median", observed=True)
    summary = d.groupby("model").agg(rmse_median=("rmse_pooled", "median"), inside_median=("inside_pooled", "median"),
                                     seconds_median=("seconds", "median"),
                                     butterfly_viol_extrap=("share_g_neg_extrap", "mean"),
                                     calendar_viol_extrap=("share_calendar_viol_extrap", "mean"),
                                     localvar_valid=("share_localvar_valid", "median"))
    ts = d.pivot_table(index="date", columns="model", values="rmse_pooled")
    ex.write_json(out / "models.json", ex._r({"n_dates": int(d["date"].nunique()), "summary": summary.to_dict("index"),
                                              "by_bucket": {str(k): v for k, v in by_bucket.to_dict("index").items()},
                                              "rmse_series": {"dates": ts.index.tolist(),
                                                              **{m: ts[m].round(4).tolist() for m in ts.columns}}}))


def main() -> None:
    out = config.WEB_BUNDLE
    out.mkdir(parents=True, exist_ok=True)
    dates = surfaces(out)
    scenarios(out)
    backtests(out)
    models(out)
    names = {"SPX": "S&P 500 index (European, cash-settled)", "SPY": "SPDR S&P 500 ETF (American)",
             "QQQ": "Invesco QQQ (Nasdaq-100 ETF, American)", "IWM": "iShares Russell 2000 ETF (American)"}
    ex.write_json(out / "manifest.json", {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"), "repo": REPO,
                                          "underlyings": {u: {"name": names.get(u, u), "dates": ds}
                                                          for u, ds in dates.items()},
                                          "files": ["scenarios.json", "backtests.json", "models.json"]})
    print(f"bundle written to {out}: {sum(len(v) for v in dates.values())} surfaces")


if __name__ == "__main__":
    main()
