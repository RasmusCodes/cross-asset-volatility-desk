"""Recompute backtest summaries from the cached daily results (no engine rerun): three cost models, yearly returns,
stress-episode P&L, and the strategy correlation matrix.

    python scripts/summarize_backtests.py [--tag ""]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.backtest.library import BY_NAME  # noqa: E402
from voldesk.backtest.metrics import summarize, unit_returns, with_cost_model, yearly  # noqa: E402
from voldesk.strategies.costs import BASE, FRICTIONLESS, STRESS  # noqa: E402

COSTS = {"base": BASE, "stress": STRESS, "frictionless": FRICTIONLESS}
EPISODES = {"LTCM 1998": ("1998-07-17", "1998-10-08"), "Lehman 2008": ("2008-09-12", "2008-11-20"),
            "Flash crash 2010": ("2010-05-05", "2010-05-07"), "Downgrade 2011": ("2011-07-22", "2011-08-08"),
            "China 2015": ("2015-08-18", "2015-08-24"), "Volmageddon 2018": ("2018-02-01", "2018-02-08"),
            "COVID 2020": ("2020-02-19", "2020-03-23"), "Bear 2022": ("2022-01-03", "2022-06-16"),
            "Yen carry 2024": ("2024-07-16", "2024-08-05"), "Tariffs 2025": ("2025-04-02", "2025-04-08")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    src = config.CACHE / "backtests" / f"run{args.tag}"
    dest = config.RESULTS / "backtests"
    dest.mkdir(parents=True, exist_ok=True)
    daily = {f.name[: -len("_daily.parquet")]: pd.read_parquet(f) for f in sorted(src.glob("*_daily.parquet"))}
    trades = {n: pd.read_csv(src / f"{n}_trades.csv") for n in daily}
    for cname, cm in COSTS.items():
        rows = []
        for n, d in daily.items():
            dd = with_cost_model(d, cm)
            s = summarize(dd, trades[n] if cname == "base" else None)
            st = BY_NAME.get(n)
            s.update({"strategy": n, "group": st.group if st else "", "hedge": st.hedge if st else "",
                      "description": st.description if st else ""})
            rows.append(s)
        pd.DataFrame(rows).set_index("strategy").to_csv(dest / f"summary_{cname}{args.tag}.csv", float_format="%.6g")
    base = {n: with_cost_model(d, BASE) for n, d in daily.items()}
    pd.DataFrame({n: yearly(d) for n, d in base.items()}).to_csv(dest / f"yearly_base{args.tag}.csv", float_format="%.5g")
    R = pd.DataFrame({n: unit_returns(d) for n, d in base.items()})
    R.corr().to_csv(dest / f"correlation_base{args.tag}.csv", float_format="%.3f")
    ep = {}
    for name, (a, b) in EPISODES.items():
        ep[name] = R.loc[a:b].sum()
    pd.DataFrame(ep).T.to_csv(dest / f"episodes_base{args.tag}.csv", float_format="%.5g")
    print(pd.read_csv(dest / f"summary_base{args.tag}.csv").set_index("strategy")[
        ["ann_return_notional", "ann_vol_notional", "sharpe", "max_drawdown_notional", "worst_day_notional"]].round(3))


if __name__ == "__main__":
    main()
