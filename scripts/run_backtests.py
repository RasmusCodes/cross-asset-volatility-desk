"""Backtest the strategy library on SPX, 1996 .. 2026-11, in one pass; costs for three cost models from the recorded
cost components (trades do not depend on costs, so one run serves all three).

    python scripts/run_backtests.py [--first 1996 --last 2026] [--only short_straddle,...] [--tag _test]

Outputs: data/cache/backtests/run<tag>/<strategy>_daily.parquet (daily P&L before costs, cost components, attribution,
Greeks, margin) and <strategy>_trades.csv (base costs); results/backtests/summary_<model><tag>.csv per cost model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.backtest import data as bdata  # noqa: E402
from voldesk.backtest.engine import MULT, Backtester  # noqa: E402
from voldesk.backtest.library import LIBRARY  # noqa: E402
from voldesk.backtest.metrics import summarize, with_cost_model  # noqa: E402
from voldesk.strategies.costs import BASE, FRICTIONLESS, STRESS  # noqa: E402
from voldesk.strategies.margin import MarginLeg, requirement  # noqa: E402

COSTS = {"base": BASE, "stress": STRESS, "frictionless": FRICTIONLESS}


def margin_fn(legs: pd.DataFrame, day: pd.DataFrame) -> float:
    S = float(day["S"].iloc[0])
    ml = [MarginLeg(c, k, t, q, MULT, p) for c, k, t, q, p in
          zip(legs["cp"], legs["K"], legs["last_T"], legs["qty"], legs["last_mid"])]
    return requirement(ml, S, "SPX")["requirement"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1996)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    strats = [s for s in LIBRARY if not args.only or s.name in args.only.split(",")]
    days = bdata.spx_trading_days()
    t0 = time.time()
    bt = Backtester(strats, BASE, margin_fn=margin_fn)
    bt.run(bdata.spx_frames(args.first, args.last), days,
           progress=lambda y: print(f"{time.strftime('%H:%M:%S')} {y} done ({time.time() - t0:.0f}s)", flush=True))
    res = bt.results()
    out = config.CACHE / "backtests" / f"run{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    dest = config.RESULTS / "backtests"
    dest.mkdir(parents=True, exist_ok=True)
    for name, r in res.items():
        r["daily"].to_parquet(out / f"{name}_daily.parquet")
        r["trades"].to_csv(out / f"{name}_trades.csv", index=False)
    for cname, cm in COSTS.items():
        rows = []
        for name, r in res.items():
            d = with_cost_model(r["daily"], cm)
            s = summarize(d, r["trades"] if cname == "base" else None)
            s.update({"strategy": name, "group": r["strategy"].group, "hedge": r["strategy"].hedge,
                      "hedge_trades": r["hedges"], "description": r["strategy"].description})
            rows.append(s)
        summ = pd.DataFrame(rows).set_index("strategy")
        summ.to_csv(dest / f"summary_{cname}{args.tag}.csv", float_format="%.6g")
        print(f"\n== {cname}")
        print(summ[["group", "ann_return_notional", "ann_vol_notional", "sharpe", "max_drawdown_notional",
                    "cost_share_of_gross"]].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
