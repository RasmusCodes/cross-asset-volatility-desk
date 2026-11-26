"""Year-by-year frames for the backtest engine from the cleaned store.

SPX: OptionMetrics (data/store/spx) through 2025-08-29, then Cboe (data/store/spx_cboe). Only the columns the engine
uses are read, and only standard-cycle rows plus the forwards of every expiry (the hedge needs the near forward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .. import config

COLS = ["date", "exdate", "optionid", "cp", "K", "bid", "ask", "mid", "am", "etype", "T", "S", "F", "D", "k", "iv_mid"]
OM_LAST = pd.Timestamp("2025-08-29")


def _read(name: str, year: int, start=None, end=None):
    f = config.STORE / name / f"{year}.parquet"
    ff = config.STORE / f"{name}_forwards" / f"{year}.parquet"
    if not f.exists():
        return None, None
    flt = ds.field("etype").isin(["standard_am", "standard_pm"])
    if start is not None:
        flt = flt & (ds.field("date") >= pd.Timestamp(start))
    if end is not None:
        flt = flt & (ds.field("date") <= pd.Timestamp(end))
    chain = ds.dataset(str(f)).to_table(columns=COLS, filter=flt).to_pandas()
    fw = ds.dataset(str(ff)).to_table(columns=["date", "exdate", "am", "F"]).to_pandas()
    for c in ("date", "exdate"):
        chain[c] = pd.to_datetime(chain[c])
        fw[c] = pd.to_datetime(fw[c])
    if start is not None:
        fw = fw[fw["date"] >= pd.Timestamp(start)]
    if end is not None:
        fw = fw[fw["date"] <= pd.Timestamp(end)]
    chain["cp"] = chain["cp"].astype(np.int8)
    return with_otm_iv(chain), fw


def with_otm_iv(chain: pd.DataFrame) -> pd.DataFrame:
    """Replace each contract's implied vol by the out-of-the-money option's at the same strike and expiry (the call
    if K >= F, the put otherwise), falling back to its own. For European options put-call parity makes the two equal;
    in practice deep in-the-money quotes are stale or below intrinsic (no implied vol at all), while the OTM side at
    the same strike is liquid. The contract's own value stays in iv_own."""
    c = chain
    c["iv_own"] = c["iv_mid"]
    otm = ((c["cp"] == 1) & (c["K"] >= c["F"])) | ((c["cp"] == -1) & (c["K"] < c["F"]))
    key = ["date", "exdate", "am", "K"]
    o = c.loc[otm & np.isfinite(c["iv_mid"]) & (c["iv_mid"] > 0), key + ["iv_mid"]].drop_duplicates(key)
    m = c[key].merge(o.rename(columns={"iv_mid": "iv_otm"}), on=key, how="left")["iv_otm"].to_numpy()
    c["iv_mid"] = np.where(np.isfinite(m), m, c["iv_own"].to_numpy())
    return c


def spx_frames(first: int = 1996, last: int = 2026):
    for y in range(first, last + 1):
        if y <= OM_LAST.year:
            c, f = _read("spx", y, end=OM_LAST)
            if c is not None and len(c):
                yield y, c, f
        if y >= OM_LAST.year:
            c, f = _read("spx_cboe", y, start=OM_LAST + pd.Timedelta(days=1))
            if c is not None and len(c):
                yield y + 0.5 if y == OM_LAST.year else y, c, f


def spx_trading_days() -> pd.DatetimeIndex:
    out = []
    for name in ("spx", "spx_cboe"):
        for f in sorted((config.STORE / name).glob("*.parquet")):
            t = ds.dataset(str(f)).to_table(columns=["date"]).column("date").unique().to_pandas()
            out.append(pd.to_datetime(t))
    return pd.DatetimeIndex(sorted(set(pd.concat(out))))
