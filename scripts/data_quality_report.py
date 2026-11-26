"""Data-quality tables: yearly cleaning flags for SPX (OptionMetrics) and the Cboe extension, and a cross-vendor
comparison of the same SPX contracts in OptionMetrics and Cboe on 13 dates (1998-2025) plus the Jul-Aug 2025 overlap.

    python scripts/data_quality_report.py
Outputs: results/data_quality/summary_yearly.csv, cross_vendor.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.data import catalog, store  # noqa: E402

OUT = config.RESULTS / "data_quality"


def yearly() -> pd.DataFrame:
    rows = []
    for f in sorted(OUT.glob("spx_*.json")):
        if f.stem == "spx_cboe":
            continue
        q = json.loads(f.read_text(encoding="utf-8"))["SPX"]
        rows.append(_row(f.stem[4:], "OptionMetrics", q))
    c = json.loads((OUT / "spx_cboe.json").read_text(encoding="utf-8"))
    for y, q in c.items():
        rows.append(_row(f"{y} (Cboe)", "Cboe", q))
    return pd.DataFrame(rows).set_index("year")


def _row(year, vendor, q) -> dict:
    n = q["rows"]
    st = q.get("iv_status", {})
    ivv = q.get("iv_vs_vendor_volpts", {})
    fw = q.get("forward_vs_optionm_bp", {})
    fs = q.get("forward_source", {})
    return {"year": year, "vendor": vendor, "rows": n, "days": q["days"],
            "pct_crossed": 100 * q["f_crossed"] / n, "pct_no_bid": 100 * q["f_no_bid"] / n,
            "pct_wide": 100 * q["f_wide"] / n, "pct_below_intrinsic": 100 * q["f_below_intrinsic"] / n,
            "pct_vendor_iv_missing": 100 * q["f_vendor_iv_missing"] / n,
            "pct_iv_unsolvable": 100 * (n - st.get("OK", 0)) / n,
            "iv_minus_vendor_median_vp": ivv.get("median_monthly_wmean"),
            "iv_minus_vendor_meanabs_vp": ivv.get("mean_abs_monthly_wmean"),
            "fwd_minus_om_median_bp": fw.get("median_monthly_wmean"),
            "pct_forwards_from_parity": 100 * fs.get("parity", 0) / max(sum(fs.values()), 1)}


def cross_vendor() -> pd.DataFrame:
    con = store.read_raw("cboe/optcontract_spx.parquet")[["optid", "putcall", "strike", "expdate", "opraroot"]]
    con["exdate"] = pd.to_datetime(con["expdate"])
    con["cp"] = np.where(con["putcall"].str.upper().str[0] == "C", 1, -1)
    rows = []
    qa = [pd.Timestamp(d) for d in catalog.CBOE_QA_DATES] + [pd.Timestamp("2025-07-15"), pd.Timestamp("2025-08-15")]
    for d in qa:
        y = d.year
        if d >= pd.Timestamp("2025-07-01"):
            px = ds.dataset(str(config.RAW / f"cboe/optprice_spx/{y}.parquet")).to_table(filter=ds.field("date_") == d).to_pandas()
            iv = ds.dataset(str(config.RAW / f"cboe/ivlisted_spx/{y}.parquet")).to_table(filter=ds.field("date_") == d).to_pandas()
        else:
            px = store.read_raw(f"cboe/qa_optprice_spx/{y}.parquet")
            iv = store.read_raw(f"cboe/qa_ivlisted_spx/{y}.parquet")
            px, iv = px[pd.to_datetime(px["date_"]) == d], iv[pd.to_datetime(iv["date_"]) == d]
        cb = px.merge(iv[["optid", "date_", "ivmid"]], on=["optid", "date_"], how="left").merge(con, on="optid")
        om = ds.dataset(str(config.RAW / f"optionm/opprcd/spx/{y}.parquet")).to_table(
            filter=ds.field("date") == d, columns=["exdate", "cp_flag", "strike_price", "best_bid", "best_offer",
                                                  "impl_volatility", "am_settlement"]).to_pandas()
        om["exdate"] = pd.to_datetime(om["exdate"])
        # OptionMetrics' pre-2015 standard expiries are Saturdays; Cboe lists the Saturday too for those contracts
        om["cp"] = np.where(om["cp_flag"] == "C", 1, -1)
        om["strike"] = om["strike_price"] / 1000.0
        om["am"] = om["am_settlement"].fillna(0).astype(int)
        cb["am"] = (cb["opraroot"] != "SPXW").astype(int)
        m = om.merge(cb, on=["exdate", "cp", "strike", "am"], suffixes=("_om", "_cb"))
        two = (m["best_bid"] > 0) & (m["bid"] > 0)
        mm = m[two]
        mid_om = 0.5 * (mm["best_bid"] + mm["best_offer"])
        mid_cb = 0.5 * (mm["bid"] + mm["ask"])
        rel = (mid_om - mid_cb).abs() / mid_cb
        both_iv = (mm["impl_volatility"] > 0) & (mm["ivmid"] > 0)
        div = (mm.loc[both_iv, "impl_volatility"] - mm.loc[both_iv, "ivmid"]) * 100
        rows.append({"date": f"{d:%Y-%m-%d}", "om_rows": len(om), "cboe_rows": len(cb), "matched": len(m),
                     "matched_two_sided": int(two.sum()),
                     "share_same_bid": float((mm["best_bid"] - mm["bid"]).abs().le(0.005).mean()) if len(mm) else np.nan,
                     "share_same_ask": float((mm["best_offer"] - mm["ask"]).abs().le(0.005).mean()) if len(mm) else np.nan,
                     "mid_reldiff_median": float(rel.median()) if len(mm) else np.nan,
                     "mid_reldiff_p95": float(rel.quantile(0.95)) if len(mm) else np.nan,
                     "iv_diff_median_vp": float(div.median()) if len(div) else np.nan,
                     "iv_diff_meanabs_vp": float(div.abs().mean()) if len(div) else np.nan})
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    y = yearly()
    y.to_csv(OUT / "summary_yearly.csv", float_format="%.4g")
    pd.set_option("display.width", 250)
    print(y.round(3).to_string())
    c = cross_vendor()
    c.to_csv(OUT / "cross_vendor.csv", float_format="%.4g")
    print(c.round(4).to_string())


if __name__ == "__main__":
    main()
