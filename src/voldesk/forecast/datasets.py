"""Daily research panels for forecasting, assembled from the raw pulls (all point-in-time: values known at the close).

  spx_daily        SPX open/high/low/close: OptionMetrics secprd 1996-01 .. 2025-08-29, Cboe eqprice after
  vix_daily        Cboe VIX, VXO, VXN, VXD closes (cboe.cboe), 1990 .. 2026-10-30
  atm_term         OptionMetrics standardized ATM implied vol (mean of call and put) at 10/30/60/91/182/365 days
  skew_daily       25-delta minus 50-delta put vol at 30 and 91 days (standardized surface)
  vix_futures_cm   Cboe VIX futures (Datastream settlements): constant-maturity 30- and 60-day prices interpolated
                   linearly in days between the two contracts bracketing each maturity, front and second contract
  vvol_implied     30-day ATM implied vol of VIX options (OptionMetrics standardized options on the VIX), from 2006
  names_panel      daily returns of the point-in-time top-50 S&P 500 members and their month-end market-cap weights
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import catalog, store


def spx_daily() -> pd.DataFrame:
    om = store.secprd([catalog.SPX], columns=["secid", "date", "open", "high", "low", "close"])
    om = om.drop(columns="secid").set_index("date").sort_index()
    eq = store.read_raw("cboe/eqprice_panel.parquet")
    eq = eq[eq["eqid"] == catalog.CBOE_EQIDS["SPX"]].copy()
    eq["date"] = pd.to_datetime(eq["date_"])
    eq = eq.set_index("date").rename(columns={"open_": "open", "close_": "close"})[["open", "high", "low", "close"]]
    ext = eq[eq.index > om.index.max()]
    d = pd.concat([om, ext]).sort_index()
    d = d[~d.index.duplicated(keep="first")]
    for c in ("open", "high", "low"):
        bad = (d[c] <= 0) | d[c].isna()
        d.loc[bad, c] = np.nan
    d["r"] = np.log(d["close"]).diff()
    d["r2"] = d["r"] ** 2
    hl = np.log(d["high"] / d["low"])
    d["parkinson"] = (hl ** 2) / (4 * np.log(2))
    d.loc[(d["high"] <= d["low"]) | hl.isna(), "parkinson"] = np.nan
    d["source"] = np.where(d.index <= om.index.max(), "optionm", "cboe")
    return d


def vix_daily() -> pd.DataFrame:
    v = store.read_raw("cboe/vix_family.parquet")
    v["date"] = pd.to_datetime(v["date"])
    v = v.set_index("date").sort_index()
    v = v[~v.index.duplicated(keep="last")]          # the WRDS table repeats a few dates
    for c in v.columns:
        v.loc[v[c] <= 0, c] = np.nan
    return v


def atm_term(secid: int = catalog.SPX, tenors=(10, 30, 60, 91, 182, 365)) -> pd.DataFrame:
    s = store.read_dir("optionm/stdopd_panel", columns=["secid", "date", "days", "impl_volatility"])
    s = s[(s["secid"] == secid) & s["days"].isin(tenors)]
    s["date"] = pd.to_datetime(s["date"])
    out = s.groupby(["date", "days"])["impl_volatility"].mean().unstack()
    out.columns = [f"iv{int(c)}" for c in out.columns]
    return out.sort_index()


def skew_daily(secid: int = catalog.SPX) -> pd.DataFrame:
    v = store.read_dir("optionm/vsurfd_panel", columns=["secid", "date", "days", "delta", "impl_volatility"])
    v = v[(v["secid"] == secid) & v["delta"].isin([-25, -50, 25, 50])]
    v["date"] = pd.to_datetime(v["date"])
    p = v.pivot_table(index="date", columns=["days", "delta"], values="impl_volatility")
    out = pd.DataFrame(index=p.index)
    for d in (30, 91):
        if (d, -25) in p and (d, -50) in p:
            out[f"skew{d}"] = p[(d, -25)] - p[(d, -50)]
        if (d, 25) in p and (d, 50) in p:
            out[f"cwing{d}"] = p[(d, 25)] - p[(d, 50)]
    return out.sort_index()


def vix_futures_cm(maturities=(30, 60)) -> pd.DataFrame:
    info = store.read_raw("datastream/vix_contracts.parquet")
    val = store.read_raw("datastream/vix_settlements.parquet")
    val["date"] = pd.to_datetime(val["date_"])
    info = info[["futcode", "lasttrddate", "sttlmntdate"]].copy()
    info["expiry"] = pd.to_datetime(info["sttlmntdate"]).fillna(pd.to_datetime(info["lasttrddate"]))
    df = val.merge(info[["futcode", "expiry"]], on="futcode", how="inner")
    df = df[(df["settlement"] > 0) & df["expiry"].notna()]
    df["days"] = (df["expiry"] - df["date"]).dt.days
    df = df[df["days"] >= 1]
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("days")
        rec = {"date": d, "vx1": g["settlement"].iloc[0], "vx1_days": g["days"].iloc[0]}
        if len(g) > 1:
            rec["vx2"], rec["vx2_days"] = g["settlement"].iloc[1], g["days"].iloc[1]
        x, y = g["days"].to_numpy(float), g["settlement"].to_numpy(float)
        for m in maturities:
            rec[f"vx_cm{m}"] = float(np.interp(m, x, y)) if x.min() <= m <= x.max() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date").sort_index()


def vvol_implied() -> pd.Series:
    s = store.read_dir("optionm/stdopd_panel", columns=["secid", "date", "days", "impl_volatility"])
    s = s[(s["secid"] == catalog.INDEX_SECIDS["VIX"]) & (s["days"] == 30)]
    s["date"] = pd.to_datetime(s["date"])
    return s.groupby("date")["impl_volatility"].mean().sort_index().rename("vvol30")


def names_panel():
    """(returns: dates x secid, weights: month_end x secid (top-50 cap weights, normalized), atm30: dates x secid)."""
    uni = pd.read_parquet(store.config.STORE / "universe_top50.parquet")
    secids = sorted(int(s) for s in uni["secid"].dropna().unique())
    px = store.secprd(secids, columns=["secid", "date", "close", "return", "cfadj"])
    px["date"] = pd.to_datetime(px["date"])
    ret = px.pivot_table(index="date", columns="secid", values="return")
    ret = ret.where(ret > -0.99)
    w = uni.dropna(subset=["secid"]).copy()
    w["secid"] = w["secid"].astype(int)
    w["w"] = w["mcap"] / w.groupby("month_end")["mcap"].transform("sum")
    weights = w.pivot_table(index="month_end", columns="secid", values="w").fillna(0.0)
    s = store.read_dir("optionm/stdopd_names", columns=["secid", "date", "days", "impl_volatility"])
    s = s[s["days"] == 30]
    s["date"] = pd.to_datetime(s["date"])
    atm = s.groupby(["date", "secid"])["impl_volatility"].mean().unstack()
    return ret, weights, atm
