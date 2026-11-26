"""Point-in-time single-name universe: the largest S&P 500 members by market cap at each month end.

Survivorship. Membership comes from CRSP's S&P 500 spells (crsp.msp500list), so every month uses the companies that
were in the index then, including those later delisted or acquired; market caps are CRSP's |prc| x shrout at that
month end. OptionMetrics ids come from the WRDS CRSP-OptionMetrics link (opcrsphist), valid on the month end, best
(lowest) score first. CRSP's legacy files end 2024-12-31: months after that reuse the December 2024 ranking, which the
data report states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_ends(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="ME")


def top_members(msp500: pd.DataFrame, msf: pd.DataFrame, link: pd.DataFrame, names: pd.DataFrame | None = None,
                n: int = 50, start: str = "1996-01-31", end: str = "2025-08-31") -> pd.DataFrame:
    """One row per (month_end, member) for the n largest members: permno, secid, mcap ($ thousands), rank, ticker."""
    msf = msf.copy()
    msf["date"] = pd.to_datetime(msf["date"])
    msf["mcap"] = msf["prc"].abs() * msf["shrout"]
    msf["month"] = msf["date"].dt.to_period("M")
    mem = msp500.copy()
    mem["start"], mem["ending"] = pd.to_datetime(mem["start"]), pd.to_datetime(mem["ending"])
    lk = link.copy()
    lk["sdate"], lk["edate"] = pd.to_datetime(lk["sdate"]), pd.to_datetime(lk["edate"])
    last_crsp = msf["date"].max()
    rows = []
    for me in month_ends(start, end):
        ref = min(me, last_crsp)
        m = ref.to_period("M")
        alive = mem[(mem["start"] <= ref) & (mem["ending"] >= ref)]["permno"].unique()
        cap = msf[(msf["month"] == m) & msf["permno"].isin(alive)].dropna(subset=["mcap"])
        cap = cap.sort_values("mcap", ascending=False).drop_duplicates("permno").head(n)
        cand = lk[lk["permno"].isin(cap["permno"]) & (lk["sdate"] <= ref) & (lk["edate"] >= ref)]
        best = cand.sort_values(["permno", "score"]).drop_duplicates("permno")[["permno", "secid", "score"]]
        out = cap[["permno", "mcap"]].merge(best, on="permno", how="left")
        out["rank"] = np.arange(1, len(out) + 1)
        out["month_end"] = me
        out["ranking_month"] = ref.to_period("M").to_timestamp("M")
        rows.append(out)
    res = pd.concat(rows, ignore_index=True)
    if names is not None and len(names):
        nm = names.copy()
        nm["namedt"], nm["nameendt"] = pd.to_datetime(nm["namedt"]), pd.to_datetime(nm["nameendt"])
        res = res.merge(nm[["permno", "namedt", "nameendt", "ticker", "comnam"]], on="permno", how="left")
        ok = (res["namedt"] <= res["ranking_month"]) & (res["nameendt"] >= res["ranking_month"])
        res = res[ok | res["namedt"].isna()].sort_values(["month_end", "rank", "nameendt"])
        res = res.drop_duplicates(["month_end", "permno"]).drop(columns=["namedt", "nameendt"])
    return res.reset_index(drop=True)
