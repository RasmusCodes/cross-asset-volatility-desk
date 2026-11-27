"""Local-stochastic volatility on two SPX dates: particle-method calibration to the arbitrage-free eSSVI surface,
vanilla repricing at the quoted strikes and maturities, and forward-start smiles and a cliquet against pure local vol
and pure Heston.

    python scripts/lsv_study.py --second-seed --particles 200000 --paths 400000   # the reported run (both dates)
    python scripts/lsv_study.py --ablation --particles 200000 --paths 400000      # estimator settings, Heston target
    python scripts/lsv_study.py --dates 2017-06-30 --particles 50000 --paths 100000    # quick look

Per date: the eSSVI slices and the Heston parameters fitted by scripts/surface_panel.py (data/cache/surface_panel/
monthend/<date>.json) and the quotes they were fitted to (data/store, same filters). Three Monte Carlo models share
one scheme (voldesk.surface.lsv) and one time grid:
  LV      Dupire local vol of the eSSVI surface
  LSV     the date's Heston variance process times a leverage function calibrated by the particle method to the
          same local vol (calibration pass, then an independent pricing pass with the leverage frozen)
  Heston  the date's Heston parameters alone (it does not reprice the surface: its own fit error applies)
Validation: the same particle calibration aimed at the Heston model's own local variance, with the same Heston
variance process, must return a leverage of 1 and reprice the Heston smile.

Outputs (results/lsv/, derived statistics only, no quotes):
  vanilla_repricing.csv   per date, model and maturity: errors against the eSSVI surface (the calibration target)
                          and against the quoted mids, Monte Carlo standard errors, z-scores
  vanilla_summary.csv     the same pooled over all quotes of a date
  forward_smile.csv       3-month implied vols starting in 6 months, and spot-start 3-month vols, per model
  cliquet.csv             12 monthly resets, payoff max(0, sum of returns clipped to +-1%), per model
  leverage_summary.csv    leverage quantiles over time (per date)
  heston_target_check.csv validation run (repricing errors against Heston quadrature prices; leverage quantiles in
                          leverage_summary.csv)
  heston_target_ablation.csv (--ablation) the validation run under the paper's estimator settings and ours
  run_info.json           settings, seeds and timings
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from surface_panel import load_chain  # noqa: E402

from voldesk import config  # noqa: E402
from voldesk.surface import heston as hs  # noqa: E402
from voldesk.surface import localvol, lsv, svi  # noqa: E402
from voldesk.surface.slices import slices_from_chain  # noqa: E402

OUT = config.RESULTS / "lsv"
FWD_T1, FWD_T2 = 0.5, 0.75                     # 3-month options starting in 6 months
FWD_M = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10])
CLIQ_N, CLIQ_CAP = 12, 0.01                    # monthly resets over one year, returns clipped to +-1%


def load_date(date: str):
    r = json.loads((config.CACHE / "surface_panel" / "monthend" / f"{date}.json").read_text(encoding="utf-8"))
    es = r["models"]["essvi"]["params"]["slices"]
    surf = svi.SVISurface([s["T"] for s in es], [svi.ESSVISlice(s["theta"], s["rho"], s["psi"]) for s in es])
    hp = r["models"]["heston"]["params"]
    p = hs.HestonParams(hp["v0"], hp["kappa"], hp["theta"], hp["sigma"], hp["rho"])
    sl = slices_from_chain(load_chain(pd.Timestamp(date)), etypes=("standard_am",))
    cached = [round(s["T"], 10) for s in r["slices"]]
    if [round(s.T, 10) for s in sl] != cached:
        raise RuntimeError(f"{date}: slices differ from the cached fit; rerun scripts/surface_panel.py")
    return surf, p, sl


def vanilla_rows(date, model, X, slices, surf, extra=None):
    rows = []
    for s in slices:
        k = s.k
        iv_t = surf.iv(k, s.T)                                  # eSSVI: the calibration target
        tgt = lsv.black_otm(k, s.T, iv_t)
        pr, se = lsv.otm_prices(X[s.T], k)
        iv, se_iv = lsv.otm_iv(pr, k, s.T, se)
        z = (pr - tgt) / se
        e_t = (iv - iv_t) * 100
        e_m = (iv - s.iv_mid) * 100
        ok = np.isfinite(e_t) & np.isfinite(z)
        row = {"date": date, "model": model, "T": round(s.T, 6), "n": int(k.size), "n_valid": int(ok.sum()),
               "rmse_vs_essvi": float(np.sqrt(np.mean(e_t[ok] ** 2))),
               "bias_vs_essvi": float(np.mean(e_t[ok])),
               "maxabs_vs_essvi": float(np.max(np.abs(e_t[ok]))),
               "se_median": float(np.median(se_iv[ok] * 100)),
               "share_absz_le2": float(np.mean(np.abs(z[ok]) <= 2)),
               "share_absz_le3": float(np.mean(np.abs(z[ok]) <= 3)),
               "max_absz": float(np.max(np.abs(z[ok]))),
               "mean_z": float(np.mean(z[ok])),
               "rmse_vs_mid": float(np.sqrt(np.mean(e_m[ok] ** 2))),
               "essvi_rmse_vs_mid": float(np.sqrt(np.mean(((iv_t - s.iv_mid) * 100) ** 2)))}
        if extra is not None:
            row.update(extra(s, iv, pr, se))
        rows.append(row)
    return rows


def forward_smile_rows(date, model, X):
    k = np.log(FWD_M)
    tau = FWD_T2 - FWD_T1
    Y = X[FWD_T2] - X[FWD_T1]
    pr, se = lsv.otm_prices(Y, k)
    iv, se_iv = lsv.otm_iv(pr, k, tau, se)
    pr0, se0 = lsv.otm_prices(X[tau], k)
    iv0, se_iv0 = lsv.otm_iv(pr0, k, tau, se0)
    return [{"date": date, "model": model, "moneyness": float(m), "fwd_iv": float(a) * 100, "fwd_iv_se": float(b) * 100,
             "spot_iv": float(c) * 100, "spot_iv_se": float(d) * 100}
            for m, a, b, c, d in zip(FWD_M, iv, se_iv, iv0, se_iv0)]


def cliquet_row(date, model, X):
    t = [i / CLIQ_N for i in range(1, CLIQ_N + 1)]
    prev = np.zeros_like(X[t[0]])
    acc = np.zeros_like(prev)
    up = np.zeros_like(prev)
    for ti in t:
        R = np.exp(X[ti] - prev) - 1.0
        acc += np.clip(R, -CLIQ_CAP, CLIQ_CAP)
        up += R > 0
        prev = X[ti]
    pay = np.maximum(acc, 0.0)
    return {"date": date, "model": model, "price_pct": float(pay.mean() * 100),
            "se_pct": float(pay.std(ddof=1) / np.sqrt(pay.size) * 100),
            "mean_up_months": float(up.mean()), "p_payoff_positive": float(np.mean(pay > 0))}


def leverage_rows(date, cal, label):
    d = cal.diagnostics
    rows = []
    for T in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        if T > d["t"][-1]:
            continue
        i = int(np.argmin(np.abs(d["t"] - T)))
        rows.append({"date": date, "run": label, "t": round(float(d["t"][i]), 4), "h": float(d["h"][i]),
                     "share_supported": float(d["share_supported"][i]), "L_p01": float(d["L_p01"][i]),
                     "L_p50": float(d["L_p50"][i]), "L_p99": float(d["L_p99"][i])})
    return rows


def heston_target_check(date, p, args, *, local_linear=True, t_min=None, dt=None, label="used"):
    """Particle calibration aimed at the Heston model's own local variance with the same Heston variance: the leverage
    should be 1, and the LSV smile the Heston smile (quadrature prices). Strikes at -2.5 .. +1.5 ATM standard
    deviations, maturities 0.05, 0.25 and 1 year."""
    Ts = [0.05, 0.25, 1.0]
    t0 = time.time()
    cal = lsv.calibrate_lsv(lambda x, t: lsv.heston_local_variance(p, x, t), p, 1.0, nodes=Ts,
                            n_particles=args.particles,
                            dt=dt if dt is not None else lsv.graded_dt(args.dt_short, args.dt_long, 0.1),
                            seed=args.seed, t_min=args.t_min if t_min is None else t_min, local_linear=local_linear)
    X = lsv.simulate_lsv(cal, Ts, n_paths=args.paths, seed=args.seed + 100)
    rows = []
    for T in Ts:
        atm = float(hs.heston_iv(1.0, np.ones(1), T, 1.0, p)[0])
        z_sd = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        k = z_sd * atm * np.sqrt(T)
        pr, se = lsv.otm_prices(X[T], k)
        iv, se_iv = lsv.otm_iv(pr, k, T, se)
        ivc = hs.heston_iv(1.0, np.exp(k), T, 1.0, p, method="quad")
        z = (pr - lsv.black_otm(k, T, ivc)) / se
        rows.append({"date": date, "variant": label, "T": T, "n_strikes": int(k.size),
                     "rmse_volpts": float(np.sqrt(np.mean(((iv - ivc) * 100) ** 2))),
                     "maxabs_volpts": float(np.max(np.abs(iv - ivc)) * 100), "se_median": float(np.median(se_iv) * 100),
                     "max_absz": float(np.max(np.abs(z)))})
    return rows, leverage_rows(date, cal, f"heston_target {label}"), time.time() - t0


def ablation(args) -> None:
    """The Heston-target check under the paper's estimator settings and ours, on the first date: Nadaraya-Watson vs
    local-linear regression, bandwidth floor t_min 0.25 vs 0.05, daily vs graded steps. Writes
    heston_target_ablation.csv."""
    date = args.dates[0]
    _, p, _ = load_date(date)
    graded = lsv.graded_dt(args.dt_short, args.dt_long, 0.1)
    variants = [("Nadaraya-Watson, t_min 0.25, daily steps (paper)", False, 0.25, 1 / 365),
                ("local linear, t_min 0.25, daily steps", True, 0.25, 1 / 365),
                ("local linear, t_min 0.25, graded steps", True, 0.25, graded),
                ("local linear, t_min 0.05, graded steps (used)", True, 0.05, graded)]
    rows, lev = [], []
    for label, ll, tm, dt in variants:
        r, lr, sec = heston_target_check(date, p, args, local_linear=ll, t_min=tm, dt=dt, label=label)
        rows += r
        lev += lr
        print(f"{label}: {sec:.0f}s  RMSE by T {[round(x['rmse_volpts'], 3) for x in r]}  max|z| "
              f"{[round(x['max_absz'], 1) for x in r]}", flush=True)
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out / "heston_target_ablation.csv", index=False, float_format="%.6g")
    pd.DataFrame(lev).to_csv(out / "heston_target_ablation_leverage.csv", index=False, float_format="%.6g")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=["2017-06-30", "2020-03-31"])
    ap.add_argument("--particles", type=int, default=100_000)
    ap.add_argument("--paths", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-min", type=float, default=0.05, dest="t_min")
    ap.add_argument("--skip-heston-check", action="store_true")
    ap.add_argument("--dt-short", type=float, default=1 / 2920, dest="dt_short", help="step size up to t = 0.1")
    ap.add_argument("--dt-long", type=float, default=1 / 1460, dest="dt_long", help="step size from t = 0.4")
    ap.add_argument("--second-seed", action="store_true", help="recalibrate LSV with another particle seed (rows 'LSV#2')")
    ap.add_argument("--ablation", action="store_true", help="only the estimator ablation on the Heston target")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.ablation:
        ablation(args)
        return
    dt = lsv.graded_dt(args.dt_short, args.dt_long, 0.1)
    settings = {**vars(args), "out": Path(os.path.relpath(out, ROOT)).as_posix()}   # no local absolute paths
    van, fwd, cliq, lev, chk, info = [], [], [], [], [], {"settings": settings, "dates": {}}
    for date in args.dates:
        surf, p, slices = load_date(date)
        local_var = (lambda s: (lambda x, t: localvol.local_variance(s, x, t)))(surf)
        atm_vol = (lambda s: (lambda t: float(s.iv(np.zeros(1), max(t, 1e-6))[0])))(surf)
        T_q = [s.T for s in slices]
        nodes = sorted(set(T_q + [FWD_T1, FWD_T2, FWD_T2 - FWD_T1] + [i / CLIQ_N for i in range(1, CLIQ_N + 1)]))
        T_max = max(nodes)
        tim = {}
        t0 = time.time()
        cal = lsv.calibrate_lsv(local_var, p, T_max, nodes=nodes, n_particles=args.particles, dt=dt, seed=args.seed,
                                atm_vol=atm_vol, t_min=args.t_min)
        tim["calibration_s"] = time.time() - t0
        lev += leverage_rows(date, cal, "essvi")
        print(f"{date}: calibrated {len(cal.leverage.times) - 1} steps in {tim['calibration_s']:.0f}s", flush=True)
        t0 = time.time()
        X_lsv = lsv.simulate_lsv(cal, nodes, n_paths=args.paths, seed=args.seed + 11)
        tim["lsv_pricing_s"] = time.time() - t0
        t0 = time.time()
        X_lv = lsv.simulate_local_vol(local_var, nodes, n_paths=args.paths, dt=dt, seed=args.seed + 12, T_max=T_max)
        tim["lv_s"] = time.time() - t0
        t0 = time.time()
        X_h = lsv.simulate_heston(p, nodes, n_paths=args.paths, dt=dt, seed=args.seed + 13, T_max=T_max)
        tim["heston_s"] = time.time() - t0

        def heston_extra(s, iv, pr, se, p=p):
            ivc = hs.heston_iv(1.0, np.exp(s.k), s.T, 1.0, p, method="quad")
            e = (iv - ivc) * 100
            z = (pr - lsv.black_otm(s.k, s.T, ivc)) / se
            ok = np.isfinite(e) & np.isfinite(z)
            return {"rmse_vs_heston_cos": float(np.sqrt(np.mean(e[ok] ** 2))), "max_absz_vs_heston_cos": float(np.max(np.abs(z[ok]))),
                    "heston_cos_rmse_vs_mid": float(np.sqrt(np.nanmean(((ivc - s.iv_mid) * 100) ** 2)))}
        van += vanilla_rows(date, "LSV", X_lsv, slices, surf)
        van += vanilla_rows(date, "LV", X_lv, slices, surf)
        van += vanilla_rows(date, "Heston", X_h, slices, surf, heston_extra)
        runs = [("LV", X_lv), ("LSV", X_lsv), ("Heston", X_h)]
        del X_lv, X_h
        if args.second_seed:                       # calibration noise: new particles, new pricing paths
            t0 = time.time()
            cal2 = lsv.calibrate_lsv(local_var, p, T_max, nodes=nodes, n_particles=args.particles, dt=dt,
                                     seed=args.seed + 1, atm_vol=atm_vol, t_min=args.t_min)
            X2 = lsv.simulate_lsv(cal2, nodes, n_paths=args.paths, seed=args.seed + 21)
            tim["lsv_seed2_s"] = time.time() - t0
            van += vanilla_rows(date, "LSV#2", X2, slices, surf)
            lev += leverage_rows(date, cal2, "essvi#2")
            runs.append(("LSV#2", X2))
        for name, X in runs:
            fwd += forward_smile_rows(date, name, X)
            cliq.append(cliquet_row(date, name, X))
        del runs, X_lsv
        if not args.skip_heston_check:
            rows, lrows, sec = heston_target_check(date, p, args)
            chk += rows
            lev += lrows
            tim["heston_check_s"] = sec
        info["dates"][date] = {"heston": p.__dict__, "n_slices": len(slices), "n_quotes": int(sum(s.K.size for s in slices)),
                               "steps": int(len(cal.leverage.times) - 1), "T_max": T_max, "timings": tim}
        d = pd.DataFrame([r for r in van if r["date"] == date])
        print(d.groupby("model")[["rmse_vs_essvi", "max_absz", "share_absz_le2", "rmse_vs_mid"]].median().round(3).to_string())
        print(pd.DataFrame([c for c in cliq if c["date"] == date]).round(4).to_string(index=False), flush=True)
    pd.DataFrame(van).to_csv(out / "vanilla_repricing.csv", index=False, float_format="%.6g")
    pd.DataFrame(fwd).to_csv(out / "forward_smile.csv", index=False, float_format="%.6g")
    pd.DataFrame(cliq).to_csv(out / "cliquet.csv", index=False, float_format="%.6g")
    pd.DataFrame(lev).to_csv(out / "leverage_summary.csv", index=False, float_format="%.6g")
    if chk:
        pd.DataFrame(chk).to_csv(out / "heston_target_check.csv", index=False, float_format="%.6g")
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    summarize(out)
    print("written", out)


def summarize(out: Path) -> pd.DataFrame:
    """Pool the per-maturity rows of vanilla_repricing.csv over all quotes of a date (exact for RMSE, bias, z shares
    and max |z|; the standard error is the quote-weighted mean of the per-maturity medians)."""
    v = pd.read_csv(out / "vanilla_repricing.csv")
    rows = []
    for (date, model), g in v.groupby(["date", "model"], sort=False):
        n = g["n_valid"].to_numpy(float)
        w = n / n.sum()
        rows.append({"date": date, "model": model, "maturities": len(g), "quotes": int(n.sum()),
                     "rmse_vs_essvi": float(np.sqrt(np.sum(w * g["rmse_vs_essvi"] ** 2))),
                     "bias_vs_essvi": float(np.sum(w * g["bias_vs_essvi"])),
                     "maxabs_vs_essvi": float(g["maxabs_vs_essvi"].max()),
                     "se_typical": float(np.sum(w * g["se_median"])),
                     "share_absz_le2": float(np.sum(w * g["share_absz_le2"])),
                     "share_absz_le3": float(np.sum(w * g["share_absz_le3"])),
                     "max_absz": float(g["max_absz"].max()),
                     "maturities_mean_absz_gt1": int((g["mean_z"].abs() > 1).sum()),
                     "rmse_vs_mid": float(np.sqrt(np.sum(w * g["rmse_vs_mid"] ** 2))),
                     "essvi_rmse_vs_mid": float(np.sqrt(np.sum(w * g["essvi_rmse_vs_mid"] ** 2)))})
    s = pd.DataFrame(rows)
    s.to_csv(out / "vanilla_summary.csv", index=False, float_format="%.6g")
    print(s.round(3).to_string(index=False))
    return s


if __name__ == "__main__":
    if "--summarize-only" in sys.argv:
        summarize(Path(sys.argv[sys.argv.index("--summarize-only") + 1]) if len(sys.argv) > sys.argv.index("--summarize-only") + 1 else OUT)
    else:
        main()
