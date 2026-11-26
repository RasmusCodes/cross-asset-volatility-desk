"""One-off repair of daily backtest files written before the engine fix of 2026-11-22.

The engine's daily record stored the position's spot delta (index units) under the key "delta", overwriting the
delta-P&L attribution term after the residual had been computed. The residual was therefore computed with the true
delta P&L, so that term is recovered exactly: delta = mtm - (gamma + vega + vanna + volga + theta + rates + residual).
The overwritten value is kept as pos_delta. Files already in the new layout (with pos_delta) are left alone.

    python scripts/repair_backtest_cache.py [--tag ""]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402

OTHER = ["gamma", "vega", "vanna", "volga", "theta", "rates", "residual"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for f in sorted((config.CACHE / "backtests" / f"run{a.tag}").glob("*_daily.parquet")):
        d = pd.read_parquet(f)
        if "pos_delta" in d:
            continue
        d["pos_delta"] = d["delta"]
        d["delta"] = d["mtm"] - d[OTHER].sum(axis=1)
        flat = d["n_legs"].fillna(0) == 0
        d.loc[flat, "pos_delta"] = np.nan
        d.to_parquet(f)
        print(f"repaired {f.name}: delta P&L total {d['delta'].sum():,.0f}")


if __name__ == "__main__":
    main()
