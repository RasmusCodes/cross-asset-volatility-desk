"""Dispersion and correlation trades on the S&P 500's top-50 members, monthly, 1996-2025 (theoretical, variance-based;
see voldesk.dispersion.dispersion), plus correlation-forecast timing.

    python scripts/dispersion_study.py
Outputs: results/dispersion/cycles.csv, summary.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.dispersion.dispersion import dispersion_cycles  # noqa: E402
from voldesk.forecast import datasets as dsets  # noqa: E402

OUT = config.RESULTS / "dispersion"


def stats(x: pd.Series, per_year: float = 12.0) -> dict:
    x = x.dropna()
    return {"cycles": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "hit_rate": float((x > 0).mean()),
            "sharpe_ann": float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else np.nan,
            "t_stat": float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else np.nan,
            "worst": float(x.min()), "max_dd": float((x.cumsum() - x.cumsum().cummax()).min())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names_ret, weights, names_iv = dsets.names_panel()
    spx = dsets.spx_daily()
    idx_ret = spx["r"]
    idx_iv = dsets.atm_term()["iv30"]
    vix = dsets.vix_daily()["vix"]
    # entries: the SPX backtests' monthly roll dates (last trading day before each standard expiry), 1996-2025
    t = pd.read_csv(config.CACHE / "backtests" / "run" / "short_straddle_trades.csv", parse_dates=["entry_date"])
    entries = [d for d in t["entry_date"] if d <= pd.Timestamp("2025-07-31")]
    cyc = dispersion_cycles(idx_ret, names_ret, weights, idx_iv, names_iv, entries, horizon=21, vix=vix)
    corr_fc = pd.read_parquet(config.CACHE / "forecast" / "corr.parquet")
    cyc = cyc.join(corr_fc[["HAR_implied", "HAR"]].rename(columns={"HAR_implied": "fc_corr_har_implied",
                                                                   "HAR": "fc_corr_har"}), how="left")
    # published table: P&L, realized measures and derived correlations; the vendor's ATM vol levels stay out
    cyc.drop(columns=["index_iv", "basket_iv_rms"]).to_csv(OUT / "cycles.csv", float_format="%.6g")
    rows = []
    for label, col, cost_col, mult in [("dispersion (ATM strikes), gross", "pnl_dispersion", None, 1),
                                       ("dispersion (ATM strikes), net of base costs", "pnl_dispersion", "cost_dispersion", 1),
                                       ("dispersion (ATM strikes), net of 2x costs", "pnl_dispersion", "cost_dispersion", 2),
                                       ("dispersion (index strike = VIX^2), gross", "pnl_dispersion_vixstrike", None, 1),
                                       ("dispersion (index strike = VIX^2), net", "pnl_dispersion_vixstrike", "cost_dispersion", 1),
                                       ("short correlation swap, gross", "pnl_corr", None, 1),
                                       ("short correlation swap, net", "pnl_corr", "cost_corr", 1)]:
        x = cyc[col] - (mult * cyc[cost_col] if cost_col else 0.0)
        rows.append({"trade": label, "units": "variance points" if "dispersion" in label else "correlation points",
                     **stats(x)})
    # timing: trade the correlation swap only when implied exceeds the forecast realized correlation by > 0.10
    sig = cyc["rho_implied"] - cyc["fc_corr_har"]
    x = (cyc["pnl_corr"] - cyc["cost_corr"]).where(sig > 0.10, 0.0).where(sig.notna())
    rows.append({"trade": "short correlation swap, net, only if implied - HAR forecast > 0.10",
                 "units": "correlation points", **stats(x), "share_traded": float((sig > 0.10).mean())})
    x = (cyc["pnl_dispersion"] - cyc["cost_dispersion"]).where(sig > 0.10, 0.0).where(sig.notna())
    rows.append({"trade": "dispersion (ATM), net, only if implied - HAR forecast > 0.10", "units": "variance points",
                 **stats(x), "share_traded": float((sig > 0.10).mean())})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    print("\nimplied vs realized correlation (cycle means):", round(cyc["rho_implied"].mean(), 3),
          round(cyc["rho_realized"].mean(), 3))


if __name__ == "__main__":
    main()
