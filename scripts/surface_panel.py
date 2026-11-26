"""Surface-model comparison panel: fit SVI, SSVI, eSSVI, SABR and Heston to SPX on many dates, checkpointed per date.

    python scripts/surface_panel.py --set monthend --workers 3     # last trading day of every month, 1996-01 .. 2026-10, + 2026-11-03
    python scripts/surface_panel.py --set stability --workers 3    # every day in four windows (2008, 2017, 2020, 2024)
    python scripts/surface_panel.py --set dates --dates 2025-08-29
    python scripts/surface_panel.py --aggregate                    # cache -> results/surfaces/*.csv

Slices: AM-settled standard-cycle expiries with 7 days to 3 years left (consistent across 1996-2026), OTM quotes
with forward |delta| >= 1% (voldesk.surface.slices). Per-date results go to data/cache/surface_panel/<set>/<date>.json
(parameters, per-slice fit statistics, arbitrage diagnostics, timings); a rerun skips dates already done.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402

STABILITY_WINDOWS = [("2017-03-01", "2017-04-28"), ("2020-02-03", "2020-04-30")]
OM_LAST = pd.Timestamp("2025-08-29")


def store_name(date: pd.Timestamp) -> str:
    return "spx" if date <= OM_LAST else "spx_cboe"


def trading_days() -> pd.DatetimeIndex:
    import pyarrow.dataset as ds
    out = []
    for name in ("spx", "spx_cboe"):
        for f in sorted((config.STORE / name).glob("*.parquet")):
            t = ds.dataset(str(f)).to_table(columns=["date"]).column("date").unique().to_pandas()
            out.append(pd.to_datetime(t))
    days = pd.DatetimeIndex(sorted(set(pd.concat(out)))) if out else pd.DatetimeIndex([])
    # the Cboe store overlaps OptionMetrics in Jul-Aug 2025; keep each date once (source chosen by store_name)
    return days


def date_set(name: str, explicit=None) -> list[pd.Timestamp]:
    days = trading_days()
    if name == "dates":
        return [pd.Timestamp(d) for d in explicit]
    if name == "monthend":
        s = pd.Series(days, index=days)
        me = s.groupby(days.to_period("M")).max().tolist()
        if days[-1] not in me:
            me.append(days[-1])
        return sorted(set(me))
    if name == "stability":
        return [d for a, b in STABILITY_WINDOWS for d in days if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
    raise ValueError(name)


def load_chain(date: pd.Timestamp) -> pd.DataFrame:
    import pyarrow.dataset as ds
    f = config.STORE / store_name(date) / f"{date.year}.parquet"
    return ds.dataset(str(f)).to_table(filter=ds.field("date") == date).to_pandas()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_etf_chain(date: pd.Timestamp, underlying: str) -> pd.DataFrame:
    import pyarrow.dataset as ds
    f = config.STORE / "etf_monthend" / f"{date.year}.parquet"
    flt = (ds.field("date") == date) & (ds.field("underlying") == underlying)
    return ds.dataset(str(f)).to_table(filter=flt).to_pandas()


def run_date(date_str: str, out_dir: str, underlying: str = "SPX") -> str:
    from voldesk.surface import compare
    from voldesk.surface.slices import slices_from_chain
    date = pd.Timestamp(date_str)
    stem = f"{date:%Y-%m-%d}" if underlying == "SPX" else f"{underlying}_{date:%Y-%m-%d}"
    dest = Path(out_dir) / f"{stem}.json"
    if dest.exists():
        return f"{stem} cached"
    t0 = time.time()
    if underlying == "SPX":
        ch = load_chain(date)
        etypes = ("standard_am",)
    else:
        ch = load_etf_chain(date, underlying)
        etypes = ("standard_pm",)        # ETF options: PM-settled (American) standard monthly cycle
    rec = {"date": f"{date:%Y-%m-%d}", "underlying": underlying,
           "source": store_name(date) if underlying == "SPX" else "etf_monthend", "rows": int(len(ch))}
    sl = slices_from_chain(ch, etypes=etypes) if len(ch) else []
    rec["slices"] = [{"T": s.T, "expiry": f"{pd.Timestamp(s.expiration):%Y-%m-%d}", "F": s.F, "D": s.D, "n": int(s.K.size),
                      "k_min": float(s.k.min()), "k_max": float(s.k.max()),
                      "atm_iv": float(np.interp(0.0, s.k, s.iv_mid)),
                      "spread_med": float(np.median(s.iv_spread)), "fwd_source": s.meta.get("fwd_source", "")}
                     for s in sl]
    rec["spot"] = float(ch["S"].iloc[0]) if len(ch) else None
    rec["models"] = {}
    if len(sl) >= 3:
        fits = compare.fit_all(sl)
        for m, f in fits.items():
            rec["models"][m] = {"params": f.params, "slices": f.slices, "seconds": f.seconds, "success": f.success,
                                "arbitrage": f.arbitrage}
    rec["seconds"] = time.time() - t0
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(rec)), encoding="utf-8")
    tmp.replace(dest)
    return f"{stem} {len(sl)} slices {rec['seconds']:.0f}s"


def aggregate(set_name: str) -> None:
    src = config.CACHE / "surface_panel" / set_name
    rows_s, rows_d = [], []
    for f in sorted(src.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for m, mr in r.get("models", {}).items():
            st = pd.DataFrame(mr.get("slices", []))
            if st.empty:
                continue
            for _, x in st.iterrows():
                rows_s.append({"date": r["date"], "model": m, **{k: x[k] for k in ("T", "expiry", "n", "rmse", "mae",
                                                                                 "max_abs", "inside", "spread_med")}})
            e2 = (st["rmse"] ** 2 * st["n"]).sum() / st["n"].sum()
            rows_d.append({"date": r["date"], "model": m, "n_slices": len(st), "n_quotes": int(st["n"].sum()),
                           "rmse_pooled": float(np.sqrt(e2)), "rmse_median_slice": float(st["rmse"].median()),
                           "inside_pooled": float((st["inside"] * st["n"]).sum() / st["n"].sum()),
                           "seconds": mr["seconds"], "success": mr["success"], **mr.get("arbitrage", {}),
                           **({f"p_{k}": v for k, v in mr["params"].items() if isinstance(v, (int, float))})})
    out = config.RESULTS / "surfaces"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_s).to_csv(out / f"{set_name}_slices.csv", index=False, float_format="%.6g")
    pd.DataFrame(rows_d).to_csv(out / f"{set_name}_dates.csv", index=False, float_format="%.6g")
    print(f"aggregated {len(rows_d)} date-model rows from {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="monthend", choices=["monthend", "stability", "dates"])
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--underlyings", nargs="*", default=["SPX"], help="SPX, or ETFs (SPY QQQ IWM) with --set dates")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.set)
        return
    etf = args.underlyings != ["SPX"]
    out_dir = config.CACHE / "surface_panel" / ("etf" if etf else args.set)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = [pd.Timestamp(d) for d in args.dates] if etf else date_set(args.set, args.dates)
    jobs = [(d, u) for d in dates for u in args.underlyings]
    stem = (lambda d, u: f"{d:%Y-%m-%d}" if u == "SPX" else f"{u}_{d:%Y-%m-%d}")
    todo = [(d, u) for d, u in jobs if not (out_dir / f"{stem(d, u)}.json").exists()]
    print(f"{time.strftime('%H:%M:%S')} {args.set}: {len(jobs)} jobs, {len(todo)} to fit", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_date, f"{d:%Y-%m-%d}", str(out_dir), u): d for d, u in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                msg = fu.result()
            except Exception as e:  # noqa: BLE001
                msg = f"{futs[fu]:%Y-%m-%d} FAILED {str(e)[:200]}"
            print(f"{time.strftime('%H:%M:%S')} [{i}/{len(todo)}] {msg}", flush=True)
    if not etf:
        aggregate(args.set)


if __name__ == "__main__":
    main()
