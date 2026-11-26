"""Surface models under perturbed and reduced inputs (SPX, 8 dates spread over 1998-2026).

    python scripts/surface_sensitivity.py [--workers 2]

Per date and model:
  noise          refit 6 times with each mid replaced by a uniform draw inside its bid-ask; report the standard
                 deviation (vol points) of the fitted vol at fixed points (30-day and 1-year maturity, k = -0.2, 0,
                 +0.05) and the average fit RMSE
  interpolation  leave every other expiry out (in turn), refit, score the held-out expiry's quotes (RMSE, vol pts)
  extrapolation  fit only quotes with forward |delta| >= 10%, score the wing quotes with 1% <= |delta| < 10%
Heston is included in the interpolation and extrapolation tests and refitted 3 times (not 6) in the noise test.
Results: data/cache/surface_sensitivity/<date>.json -> results/surfaces/sensitivity_*.csv
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
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402

DATES = ["1998-06-30", "2002-07-31", "2006-11-30", "2008-10-31", "2013-05-31", "2017-06-30", "2020-03-31", "2026-11-03"]
POINTS = [(30, -0.2), (30, 0.0), (30, 0.05), (365, -0.2), (365, 0.0)]
MODELS = ("svi", "ssvi", "essvi", "sabr", "heston")


def _iv_at(fit, T, k):
    try:
        return float(np.asarray(fit.surface.iv(np.array([k]), T)).ravel()[0])
    except Exception:  # noqa: BLE001
        return np.nan


def run(date_str: str) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from surface_panel import load_chain
    from voldesk.surface import compare
    from voldesk.surface.slices import slices_from_chain
    date = pd.Timestamp(date_str)
    dest = config.CACHE / "surface_sensitivity" / f"{date_str}.json"
    if dest.exists():
        return f"{date_str} cached"
    t0 = time.time()
    rng = np.random.default_rng(int(date.strftime("%Y%m%d")))
    sl = slices_from_chain(load_chain(date))
    rec = {"date": date_str, "n_slices": len(sl), "noise": {}, "interp": {}, "extrap": {}}
    from voldesk.pricing.implied_vol import implied_vol_black
    for m in MODELS:
        reps = 3 if m == "heston" else 6
        vals, rmses = [], []
        for _ in range(reps):
            pert = []
            for s in sl:
                u = rng.uniform(size=s.K.size)
                px = s.bid + u * (s.ask - s.bid)
                iv = implied_vol_black(px, s.cp, s.F, s.K, s.T, s.D)
                ok = np.isfinite(iv)
                pert.append(replace(s.subset(ok), iv_mid=iv[ok]))
            f = compare.FITTERS[m](pert)
            vals.append([_iv_at(f, T / 365, k) for T, k in POINTS])
            rmses.append(float(np.nanmedian([r["rmse"] for r in f.slices])))
        v = np.array(vals) * 100
        rec["noise"][m] = {"sd_volpts": np.nanstd(v, axis=0, ddof=1).tolist(), "rmse_med": float(np.mean(rmses))}
        # interpolation: leave every other expiry out (not the first or last)
        errs = []
        for j in range(1, len(sl) - 1, 2):
            train = [s for i, s in enumerate(sl) if i != j]
            f = compare.FITTERS[m](train)
            s = sl[j]
            iv = np.asarray(f.surface.iv(s.k, s.T), float)
            e = (iv - s.iv_mid) * 100
            errs.append(float(np.sqrt(np.nanmean(e ** 2))))
        rec["interp"][m] = {"rmse_heldout_med": float(np.median(errs)) if errs else np.nan, "n": len(errs)}
        # extrapolation: fit on |delta| >= 10%, score 1%-10% wings
        core = [s.subset(s.forward_delta() >= 0.10) for s in sl]
        core = [c for c in core if c.K.size >= 6]
        f = compare.FITTERS[m](core)
        e_w = []
        for s in sl:
            w = s.forward_delta() < 0.10
            if w.sum() == 0:
                continue
            iv = np.asarray(f.surface.iv(s.k[w], s.T), float)
            e_w.append((iv - s.iv_mid[w]) * 100)
        e_w = np.concatenate(e_w) if e_w else np.array([np.nan])
        rec["extrap"][m] = {"rmse_wings": float(np.sqrt(np.nanmean(e_w ** 2))), "bias_wings": float(np.nanmean(e_w)),
                            "n_wing_quotes": int(np.isfinite(e_w).sum())}
    rec["seconds"] = time.time() - t0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec), encoding="utf-8")
    return f"{date_str} done {rec['seconds']:.0f}s"


def aggregate() -> None:
    rows_n, rows_i, rows_e = [], [], []
    for f in sorted((config.CACHE / "surface_sensitivity").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for m in MODELS:
            n = r["noise"].get(m)
            if n:
                rows_n.append({"date": r["date"], "model": m, "rmse_med": n["rmse_med"],
                               **{f"sd_{T}d_k{k:+.2f}": v for (T, k), v in zip(POINTS, n["sd_volpts"])}})
            if m in r["interp"]:
                rows_i.append({"date": r["date"], "model": m, **r["interp"][m]})
            if m in r["extrap"]:
                rows_e.append({"date": r["date"], "model": m, **r["extrap"][m]})
    out = config.RESULTS / "surfaces"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_n).to_csv(out / "sensitivity_noise.csv", index=False, float_format="%.5g")
    pd.DataFrame(rows_i).to_csv(out / "sensitivity_interpolation.csv", index=False, float_format="%.5g")
    pd.DataFrame(rows_e).to_csv(out / "sensitivity_extrapolation.csv", index=False, float_format="%.5g")
    for name, rows in (("noise", rows_n), ("interp", rows_i), ("extrap", rows_e)):
        print(name)
        print(pd.DataFrame(rows).drop(columns="date").groupby("model").median(numeric_only=True).round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if not a.aggregate:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run, d): d for d in DATES}
            for fu in as_completed(futs):
                try:
                    print(time.strftime("%H:%M:%S"), fu.result(), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(time.strftime("%H:%M:%S"), futs[fu], "FAILED", str(e)[:300], flush=True)
    aggregate()


if __name__ == "__main__":
    main()
