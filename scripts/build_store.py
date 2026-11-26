"""Build the cleaned option store from data/raw, month by month (low memory), with a quality summary per year.

    python scripts/build_store.py --source om   [--years 2008 2009]   # OptionMetrics SPX 1996-2025
    python scripts/build_store.py --source cboe                       # Cboe SPX 2025-07 .. 2026-11
    python scripts/build_store.py --source etf  [--years ...]         # SPY/QQQ/IWM month-end + near-the-money rows

Output: data/store/<name>/<year>.parquet (one row per contract and day: voldesk.data.chains columns),
data/store/<name>_forwards/<year>.parquet, results/data_quality/<name>_<year>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.data import catalog, chains, rates, store  # noqa: E402
from voldesk.pricing.implied_vol import IVStatus  # noqa: E402

KEEP = ["date", "exdate", "optionid", "cp", "K", "bid", "ask", "mid", "spread", "volume", "oi", "am", "etype", "T", "S",
        "r", "D", "F", "k", "fwd_sd", "n_pairs", "fwd_source", "iv_bid", "iv_mid", "iv_ask", "iv_status", "v_iv",
        "v_delta", "v_gamma", "v_vega", "v_theta", "f_crossed", "f_no_bid", "f_wide", "f_below_intrinsic",
        "f_vendor_iv_missing"]
FLAGS = ["f_crossed", "f_no_bid", "f_wide", "f_below_intrinsic", "f_vendor_iv_missing"]


def log(m: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def month_bounds(year: int):
    for m in range(1, 13):
        a = pd.Timestamp(year=year, month=m, day=1)
        yield a, a + pd.offsets.MonthBegin(1)


def quality(df: pd.DataFrame, om_fwd: pd.DataFrame | None) -> dict:
    n = len(df)
    q = {"rows": n, "days": int(df["date"].nunique()), "expiries": int(df[["exdate", "am"]].drop_duplicates().shape[0])}
    for f in FLAGS:
        q[f] = int(df[f].sum())
    st = df["iv_status"].value_counts().to_dict()
    q["iv_status"] = {IVStatus(int(k)).name: int(v) for k, v in st.items()}
    otm = (((df["cp"] == 1) & (df["K"] >= df["F"])) | ((df["cp"] == -1) & (df["K"] < df["F"])))
    two = (df["bid"] > 0) & (df["ask"] >= df["bid"])
    both = otm & two & np.isfinite(df["iv_mid"]) & (df["v_iv"] > 0)
    d = (df.loc[both, "iv_mid"] - df.loc[both, "v_iv"]) * 100
    q["iv_vs_vendor_volpts"] = {"n": int(both.sum()), "median": float(d.median()) if len(d) else None,
                                "p05": float(d.quantile(0.05)) if len(d) else None,
                                "p95": float(d.quantile(0.95)) if len(d) else None,
                                "mean_abs": float(d.abs().mean()) if len(d) else None}
    fw = df.drop_duplicates(["date", "exdate", "am"])
    q["forward_source"] = fw["fwd_source"].value_counts().to_dict()
    if om_fwd is not None and len(om_fwd):
        m = fw.merge(om_fwd, left_on=["date", "exdate"], right_on=["date", "expiration"], how="inner")
        m = m[m["amsettlement"].fillna(-1).astype(int) == m["am"].astype(int)]
        bp = (m["F"] / m["forwardprice"] - 1) * 1e4
        q["forward_vs_optionm_bp"] = {"n": int(len(bp)), "median": float(bp.median()) if len(bp) else None,
                                      "p05": float(bp.quantile(0.05)) if len(bp) else None,
                                      "p95": float(bp.quantile(0.95)) if len(bp) else None}
    return q


def write_parts(parts: list[pd.DataFrame], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    writer = None
    for p in parts:
        t = pa.Table.from_pandas(p, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd", compression_level=6)
        writer.write_table(t.cast(writer.schema))
    if writer is not None:
        writer.close()
        tmp.replace(dest)


def merge_quality(qs: list[dict]) -> dict:
    """Combine monthly quality dicts: counts add up; distribution summaries become row-weighted means of the monthly
    medians/quantiles (stated as such in the data report)."""
    if not qs:
        return {}
    out: dict = {"rows": sum(q["rows"] for q in qs), "days": sum(q["days"] for q in qs), "months": len(qs)}
    for f in FLAGS:
        out[f] = sum(q[f] for q in qs)
    st: dict = {}
    for q in qs:
        for k, v in q["iv_status"].items():
            st[k] = st.get(k, 0) + v
    out["iv_status"] = st
    fs: dict = {}
    for q in qs:
        for k, v in q["forward_source"].items():
            fs[k] = fs.get(k, 0) + int(v)
    out["forward_source"] = fs
    for key in ("iv_vs_vendor_volpts", "forward_vs_optionm_bp"):
        items = [q[key] for q in qs if key in q and q[key].get("n")]
        if items:
            n = sum(i["n"] for i in items)
            agg = {"n": n}
            for stat in ("median", "p05", "p95", "mean_abs"):
                vals = [(i[stat], i["n"]) for i in items if i.get(stat) is not None]
                if vals:
                    agg[stat + "_monthly_wmean"] = float(sum(v * w for v, w in vals) / sum(w for _, w in vals))
            out[key] = agg
    return out


def build_om_year(name: str, raw_dir: str, secids: dict, year: int, curve) -> dict:
    """OptionMetrics rows of one year (one or several underlyings) -> cleaned store, written month by month."""
    files = [str(f) for f in sorted((config.RAW / raw_dir).glob(f"{year}.parquet"))]
    if not files:
        return {}
    dset = ds.dataset(files, format="parquet")
    spots = {sid: store.close_series(sid) for sid in secids.values()}
    dys = {sid: store.div_yield(sid) for sid in secids.values()}
    om_fwd = store.read_dir("optionm/fwdprd", years=[year])
    if len(om_fwd):
        om_fwd["date"] = pd.to_datetime(om_fwd["date"])
        om_fwd["expiration"] = pd.to_datetime(om_fwd["expiration"])
    dest = store.store_dir(name) / f"{year}.parquet"
    fdest = store.store_dir(f"{name}_forwards") / f"{year}.parquet"
    tmp, ftmp = dest.with_suffix(".tmp"), fdest.with_suffix(".tmp")
    writer = fwriter = None
    monthly: dict = {t: [] for t in secids}
    for a, b in month_bounds(year):
        flt = (ds.field("date") >= a.date()) & (ds.field("date") < b.date())
        raw = dset.to_table(filter=flt).to_pandas()
        if raw.empty:
            continue
        for tick, sid in secids.items():
            r = raw[raw["secid"] == sid]
            if r.empty:
                continue
            base = chains.base_frame(r, spots[sid], curve)
            fwd = chains.parity_forwards(base, dys[sid] if len(dys[sid]) else None)
            out = chains.add_forwards_and_iv(base, fwd)
            out.insert(0, "underlying", tick)
            out = out[["underlying"] + KEEP]
            f = om_fwd[om_fwd["secid"] == sid] if len(om_fwd) else None
            monthly[tick].append(quality(out, f))
            t = pa.Table.from_pandas(out, preserve_index=False)
            ft = pa.Table.from_pandas(fwd.assign(underlying=tick), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd", compression_level=6)
                fwriter = pq.ParquetWriter(ftmp, ft.schema, compression="zstd", compression_level=6)
            writer.write_table(t.cast(writer.schema))
            fwriter.write_table(ft.cast(fwriter.schema))
            del base, fwd, out
        del raw
    if writer is None:
        return {}
    writer.close()
    fwriter.close()
    tmp.replace(dest)
    ftmp.replace(fdest)
    return {t: merge_quality(q) for t, q in monthly.items() if q}


def build_cboe(curve) -> dict:
    """Cboe SPX 2025-07 .. 2026-11 into the same schema (vendor IVs/Greeks from ivlisted)."""
    con = store.read_raw("cboe/optcontract_spx.parquet")
    con = con[con["opraroot"].isin(["SPX", "SPXW"])][["optid", "putcall", "strike", "expdate", "opraroot"]]
    eq = store.read_raw("cboe/eqprice_panel.parquet")
    spot = eq[eq["eqid"] == catalog.CBOE_EQIDS["SPX"]].set_index(pd.to_datetime(eq.loc[eq["eqid"] == catalog.CBOE_EQIDS["SPX"], "date_"]))["close_"]
    dy = store.div_yield(catalog.SPX)
    summaries = {}
    for y in (2025, 2026):
        pxd = ds.dataset(str(config.RAW / f"cboe/optprice_spx/{y}.parquet"), format="parquet")
        ivd = ds.dataset(str(config.RAW / f"cboe/ivlisted_spx/{y}.parquet"), format="parquet")
        dest = store.store_dir("spx_cboe") / f"{y}.parquet"
        fdest = store.store_dir("spx_cboe_forwards") / f"{y}.parquet"
        tmp, ftmp = dest.with_suffix(".tmp"), fdest.with_suffix(".tmp")
        writer = fwriter = None
        monthly = []
        n_rows = 0
        for a, b in month_bounds(y):
            flt = (ds.field("date_") >= a.date()) & (ds.field("date_") < b.date())
            px = pxd.to_table(filter=flt).to_pandas()
            if px.empty:
                continue
            iv = ivd.to_table(filter=flt, columns=["optid", "date_", "ivmid", "delta", "gamma", "vega",
                                                   "theta"]).to_pandas()
            df = px.merge(iv, on=["optid", "date_"], how="left").merge(con, on="optid", how="inner")
            del px, iv
            out, fwd = _cboe_month(df, spot, curve, dy)
            monthly.append(quality(out, None))
            n_rows += len(out)
            t = pa.Table.from_pandas(out, preserve_index=False)
            ft = pa.Table.from_pandas(fwd, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression="zstd", compression_level=6)
                fwriter = pq.ParquetWriter(ftmp, ft.schema, compression="zstd", compression_level=6)
            writer.write_table(t.cast(writer.schema))
            fwriter.write_table(ft.cast(fwriter.schema))
        if writer is not None:
            writer.close()
            fwriter.close()
            tmp.replace(dest)
            ftmp.replace(fdest)
        summaries[str(y)] = merge_quality(monthly)
        log(f"cboe {y}: {n_rows:,} rows")
    return summaries


def cboe_to_om_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cboe optprice + ivlisted + optcontract rows -> OptionMetrics-named columns. Settlement style from the root:
    SPX = AM-settled standard cycle, SPXW = PM-settled (third-Friday SPXW is 'standard_pm', other dates weekly)."""
    am = (df["opraroot"] == "SPX").astype(np.int8)
    ex = pd.to_datetime(df["expdate"])
    third_fri = (ex.dt.dayofweek == 4) & (ex.dt.day.between(15, 21))
    ind = np.where(am == 1, None, np.where(third_fri, None, "w"))
    return pd.DataFrame({"secid": catalog.SPX, "date": pd.to_datetime(df["date_"]), "exdate": ex,
                         "cp_flag": df["putcall"].str.upper().str[0], "strike_price": df["strike"] * 1000.0,
                         "best_bid": df["bid"], "best_offer": df["ask"], "volume": df["volume"],
                         "open_interest": df["openint"], "impl_volatility": df["ivmid"], "delta": df["delta"],
                         "gamma": df["gamma"], "vega": df["vega"], "theta": df["theta"], "optionid": df["optid"],
                         "am_settlement": am, "expiry_indicator": ind})


def _cboe_month(df: pd.DataFrame, spot: pd.Series, curve, dy: pd.Series):
    raw = cboe_to_om_columns(df)
    base = chains.base_frame(raw, spot, curve)
    fwd = chains.parity_forwards(base, dy if len(dy) else None)
    out = chains.add_forwards_and_iv(base, fwd)
    out.insert(0, "underlying", "SPX")
    return out[["underlying"] + KEEP], fwd.assign(underlying="SPX")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["om", "cboe", "etf"], required=True)
    ap.add_argument("--years", nargs="*", type=int)
    args = ap.parse_args()
    curve = rates.load_curve(config.RAW)
    out_dir = config.RESULTS / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "cboe":
        s = build_cboe(curve)
        (out_dir / "spx_cboe.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
        return
    years = args.years or list(range(catalog.FIRST_YEAR, catalog.LAST_YEAR + 1))
    for y in years:
        t0 = time.time()
        if args.source == "om":
            s = build_om_year("spx", "optionm/opprcd/spx", {"SPX": catalog.SPX}, y, curve)
            tag = "spx"
        else:
            s = {}
            for sub in ("etf_monthend", "etf_ntm"):
                s[sub] = build_om_year(sub, f"optionm/opprcd/{sub}", catalog.CHAIN_ETFS, y, curve)
            tag = "etf"
        if s:
            (out_dir / f"{tag}_{y}.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
            log(f"{tag} {y}: done in {time.time() - t0:.0f}s")
        else:
            log(f"{tag} {y}: no raw data yet")


if __name__ == "__main__":
    main()
