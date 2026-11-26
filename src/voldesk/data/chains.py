"""Clean option chains, parity forwards and our own implied volatilities.

Input: OptionMetrics opprcd rows (strike in thousandths, closing best bid/offer, OM's implied vol and Greeks from the
mid), the underlying close and a discount curve. Output columns (one row per contract and day):

  date, exdate, optionid, cp (+1/-1), K, bid, ask, mid, spread, volume, oi, am, etype, T, S, r, D, F, k = ln(K/F),
  iv_bid / iv_mid / iv_ask (ours, Black-76 with our F and D), iv_status (voldesk.pricing.IVStatus of the mid),
  v_iv, v_delta, v_gamma, v_vega, v_theta (the vendor's own values: OptionMetrics or Cboe, kept for comparison), and quality flags.

Forward per (date, expiry, settlement style): put-call parity, C - P = D (F - K), on strikes where both sides have a
two-sided quote and |ln(K/S)| <= max(3%, 0.4 sqrt(T)). Each pair gives F_K = K + (C - P) / D; the forward is the
weighted mean with weights 1 / (spread_C^2 + spread_P^2), after one pass that drops pairs more than 3 weighted standard
deviations away. D comes from the zero curve (rates.py). With fewer than 3 usable pairs the forward falls back to
S exp((r - q) T) with OptionMetrics' index dividend yield q (fwd_source 'yield'), or S exp(r T) without one.

Quality flags (True = problem): crossed (ask < bid), no_bid (bid <= 0), wide (spread > mid, i.e. above 100% of mid,
with mid > 0), below_intrinsic (mid < discounted forward intrinsic value - 1 tick), vendor_iv_missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..pricing.implied_vol import IVStatus, implied_vol_black
from . import calendar

TICK = 0.05


def base_frame(raw: pd.DataFrame, spot: pd.Series, curve) -> pd.DataFrame:
    """OptionMetrics rows -> standard columns with T, S, r, D (no forward yet)."""
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]).to_numpy(), "exdate": pd.to_datetime(raw["exdate"]).to_numpy(),
        "optionid": raw["optionid"].to_numpy(), "cp": np.where(raw["cp_flag"].to_numpy() == "C", 1, -1).astype(np.int8),
        "K": raw["strike_price"].to_numpy(float) / 1000.0, "bid": raw["best_bid"].to_numpy(float),
        "ask": raw["best_offer"].to_numpy(float), "volume": raw["volume"].to_numpy(), "oi": raw["open_interest"].to_numpy(),
        "am": raw["am_settlement"].fillna(0).astype(np.int8).to_numpy(),
        "v_iv": raw["impl_volatility"].to_numpy(float), "v_delta": raw["delta"].to_numpy(float),
        "v_gamma": raw["gamma"].to_numpy(float), "v_vega": raw["vega"].to_numpy(float),
        "v_theta": raw["theta"].to_numpy(float),
    })
    df["etype"] = calendar.expiry_type(raw["am_settlement"].to_numpy(), raw["expiry_indicator"].to_numpy())
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    df["T"] = calendar.time_to_expiry(df["date"], df["exdate"], df["am"])
    s = spot.copy()
    s.index = pd.to_datetime(s.index)
    df["S"] = df["date"].map(s).to_numpy(float)
    df = df[(df["T"] > 0) & np.isfinite(df["S"])].copy()
    r = np.full(len(df), np.nan)
    for d, idx in df.groupby("date").indices.items():
        r[idx] = curve.rate(d, df["T"].to_numpy()[idx])
    df["r"] = r
    df["D"] = np.exp(-df["r"] * df["T"])
    return df.reset_index(drop=True)


def parity_forwards(df: pd.DataFrame, div_yield: pd.Series | None = None, min_pairs: int = 3) -> pd.DataFrame:
    """One row per (date, exdate, am): F, n_pairs, fwd_sd (weighted sd of the pair forwards), fwd_source."""
    keys = ["date", "exdate", "am"]
    two = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]
    c = two[two["cp"] == 1][keys + ["K", "mid", "spread", "S", "T", "D"]]
    p = two[two["cp"] == -1][keys + ["K", "mid", "spread"]]
    m = c.merge(p, on=keys + ["K"], suffixes=("_c", "_p"))
    band = np.maximum(0.03, 0.4 * np.sqrt(m["T"].to_numpy()))
    m = m[np.abs(np.log(m["K"] / m["S"])) <= band].copy()
    m["Fk"] = m["K"] + (m["mid_c"] - m["mid_p"]) / m["D"]
    m["w"] = 1.0 / np.maximum(m["spread_c"] ** 2 + m["spread_p"] ** 2, 1e-8)

    def wstats(g: pd.DataFrame) -> pd.DataFrame:
        sw = g.groupby(keys)["w"].sum()
        mean = (g["Fk"] * g["w"]).groupby([g[k] for k in keys]).sum() / sw
        return mean.rename("F").to_frame().join(sw.rename("sw"))

    st = wstats(m)
    m = m.merge(st[["F"]].reset_index(), on=keys)
    var = ((m["Fk"] - m["F"]) ** 2 * m["w"]).groupby([m[k] for k in keys]).sum() / m.groupby(keys)["w"].sum()
    m = m.merge(np.sqrt(var).rename("sd0").reset_index(), on=keys)
    keep = np.abs(m["Fk"] - m["F"]) <= 3.0 * np.maximum(m["sd0"], 1e-8)
    m2 = m[keep].drop(columns=["F", "sd0"])
    st2 = wstats(m2)
    m2 = m2.merge(st2[["F"]].reset_index(), on=keys)
    var2 = ((m2["Fk"] - m2["F"]) ** 2 * m2["w"]).groupby([m2[k] for k in keys]).sum() / m2.groupby(keys)["w"].sum()
    out = st2[["F"]].join(np.sqrt(var2).rename("fwd_sd")).join(m2.groupby(keys).size().rename("n_pairs"))
    out = out.reset_index()
    out["fwd_source"] = "parity"
    out = out[out["n_pairs"] >= min_pairs]
    # fallback for slices without enough pairs
    allk = df.groupby(keys).agg(S=("S", "first"), T=("T", "first"), r=("r", "first")).reset_index()
    allk = allk.merge(out, on=keys, how="left")
    miss = allk["F"].isna()
    if miss.any():
        q = np.zeros(int(miss.sum()))
        src = np.full(int(miss.sum()), "rate_only", dtype=object)
        if div_yield is not None and len(div_yield):
            dy = div_yield.copy()
            dy.index = pd.to_datetime(dy.index)
            qq = allk.loc[miss, "date"].map(dy).to_numpy(float) / 100.0
            q = np.where(np.isfinite(qq), qq, 0.0)
            src = np.where(np.isfinite(qq), "yield", "rate_only")
        a = allk.loc[miss]
        allk.loc[miss, "F"] = a["S"] * np.exp((a["r"] - q) * a["T"])
        allk.loc[miss, "fwd_source"] = src
        allk.loc[miss, "n_pairs"] = 0
    return allk[keys + ["F", "fwd_sd", "n_pairs", "fwd_source"]]


def add_forwards_and_iv(df: pd.DataFrame, fwd: pd.DataFrame, compute_iv: bool = True) -> pd.DataFrame:
    out = df.merge(fwd, on=["date", "exdate", "am"], how="left")
    out["k"] = np.log(out["K"] / out["F"])
    intrinsic = out["D"] * np.maximum(out["cp"] * (out["F"] - out["K"]), 0.0)
    out["f_crossed"] = out["ask"] < out["bid"]
    out["f_no_bid"] = out["bid"] <= 0
    out["f_wide"] = (out["mid"] > 0) & (out["spread"] > out["mid"])
    out["f_below_intrinsic"] = out["mid"] < intrinsic - TICK
    out["f_vendor_iv_missing"] = ~(out["v_iv"] > 0)
    if compute_iv:
        args = (out["cp"].to_numpy(float), out["F"].to_numpy(), out["K"].to_numpy(), out["T"].to_numpy(),
                out["D"].to_numpy())
        rm = implied_vol_black(out["mid"].to_numpy(), *args, full_output=True)
        out["iv_mid"] = rm.sigma
        out["iv_status"] = rm.status
        ok_b = out["bid"].to_numpy() > 0
        ivb = np.full(len(out), np.nan)
        if ok_b.any():
            ivb[ok_b] = implied_vol_black(out["bid"].to_numpy()[ok_b], *(a[ok_b] for a in args))
        out["iv_bid"] = ivb
        out["iv_ask"] = implied_vol_black(out["ask"].to_numpy(), *args)
    return out


def otm_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-the-money side only (calls with K >= F, puts with K < F), two-sided, with valid mid/bid/ask IVs."""
    otm = ((df["cp"] == 1) & (df["K"] >= df["F"])) | ((df["cp"] == -1) & (df["K"] < df["F"]))
    good = (df["bid"] > 0) & (df["ask"] >= df["bid"]) & (df["iv_status"] == IVStatus.OK)
    good &= np.isfinite(df["iv_bid"]) & np.isfinite(df["iv_ask"]) & (df["iv_ask"] >= df["iv_bid"])
    return df[otm & good].copy()
