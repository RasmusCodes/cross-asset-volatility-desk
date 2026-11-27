"""Rough volatility on SPX: the ATM-skew power law across 371 month-end surfaces, Heston's skew term structure on the
same dates, and a rough Bergomi calibration against Heston on the short-dated quotes of two dates.

    python scripts/rough_study.py                       # all parts (about an hour and a half)
    python scripts/rough_study.py --part rbergomi       # the calibration on two dates
    python scripts/rough_study.py --part convergence    # time-step convergence at the shortest maturity
    python scripts/rough_study.py --part xi0            # variance-swap vs ATM level of the forward variance
    python scripts/rough_study.py --part coarse         # the same calibration on a uniform 1,460-a-year grid
Outputs (results/rough/): atm_skew_power_law.csv; rbergomi_slices.csv (per date, model and maturity: fit errors on
the quotes); rbergomi_fits.csv (per date and model: parameters, pooled errors, Monte Carlo and time-grid checks);
rbergomi_convergence.csv (vols of the calibrated model at the shortest maturity on nested time grids);
rbergomi_xi0_gap.csv (variance-swap minus ATM vol per slice); rbergomi_coarse.csv (uniform-grid calibration).

Rough Bergomi set-up (voldesk.surface.rough): the forward variance curve xi0(t) is read off the date's arbitrage-free
eSSVI surface, not fitted, and only (H, eta, rho) are calibrated. Two readings of the surface's term structure:
  vs   variance-swap total variance of each eSSVI slice (log-contract replication); xi0 = E[v_t] in the model, so
       this is the model-consistent input;
  atm  eSSVI ATM total variance theta(T), as a cross-check. With a negative skew the ATM implied variance lies below
       the variance-swap variance, so this curve understates the variance level.
Calibration: weighted least squares on the quotes of all slices with 7 days to ~4 months, the residual weights of
the Heston fit (1 / bid-ask width, equal weight per slice), 2 x 16,384 antithetic paths with common random numbers,
the mixing estimator. Time grid: at most 1/1,460 year per step and at most T/400 in the interval ending at each
maturity T, so the three-week smile gets about 400 steps (~8,700 a year). Reported errors come from an independent
evaluation: 8 replications of 2 x 16,384 paths with fresh seeds, prices at every quoted strike.
Time-step checks: (1) paired: the calibrated model on the grid and on one twice as fine from the same paths (the
coarse scheme on every second node), so the difference is discretisation, not Monte Carlo noise; (2) at the
shortest maturity, nested grids from 1/32 to all of 1,600 steps.
Heston is refitted to the same quotes (voldesk.surface.heston.fit_heston, five parameters) and also shown as fitted
to the whole surface in the panel; SVI, eSSVI and SABR errors on the same quotes come from the panel fits.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

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
from voldesk import config  # noqa: E402
from voldesk.surface import heston as hs  # noqa: E402
from voldesk.surface import svi  # noqa: E402
from voldesk.surface.rough import (RBergomiMC, calibrate_rbergomi, forward_variance_curve,  # noqa: E402
                                   power_law_exponent, svi_atm_skew, variance_swap_total_variance)

OUT = config.RESULTS / "rough"
T_SHORT = 0.35                 # "short-dated": the slices with up to ~4 months left (four per date here)
EVAL_REPS, EVAL_PATHS = 8, 2 ** 14
STEPS = 1460                   # at most 1/STEPS years per time step ...
MIN_STEPS = 400                # ... and at most T/MIN_STEPS in the interval ending at maturity T


def heston_atm_skew(p: dict, T: float, h: float = 1e-3) -> float:
    pp = hs.HestonParams(p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
    iv = hs.heston_iv(1.0, np.exp(np.array([-h, h])), T, 1.0, pp)
    return float(abs(iv[1] - iv[0]) / (2 * h))


def skew_part() -> None:
    rows = []
    for f in sorted((config.CACHE / "surface_panel" / "monthend").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        m = r.get("models", {})
        if "svi" not in m or not m["svi"].get("params"):
            continue
        sl = m["svi"]["params"]["slices"]
        T = np.array([s["T"] for s in sl])
        psi = np.array([svi_atm_skew(s["a"], s["b"], s["rho"], s["m"], s["s"], s["T"]) for s in sl])
        fit = power_law_exponent(T, psi)
        rec = {"date": r["date"], "svi_slope": fit["slope"], "H_implied": fit["H"], "n": fit["n"], "r2": fit["r2"],
               "psi_1m": float(np.interp(30 / 365, T, psi)) if len(T) else np.nan}
        if "heston" in m and m["heston"].get("params", {}).get("v0") is not None:
            Th = T[(T >= 7 / 365) & (T <= 0.5)]
            if len(Th) >= 3:
                ph = np.array([heston_atm_skew(m["heston"]["params"], t) for t in Th])
                hf = power_law_exponent(Th, ph)
                rec.update({"heston_slope": hf["slope"]})
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "atm_skew_power_law.csv", index=False, float_format="%.5g")
    print(d[["svi_slope", "H_implied", "r2", "heston_slope"]].describe().round(3).to_string())


# ------------------------------------------------------------------------------------------------ rough Bergomi
def quote_errors(model_iv: dict, slices) -> tuple[list, dict]:
    rows, e_all = [], []
    for s in slices:
        e = (np.asarray(model_iv[s.T]) - s.iv_mid) * 100
        ok = np.isfinite(e)
        inside = (model_iv[s.T] >= s.iv_bid) & (model_iv[s.T] <= s.iv_ask)
        rows.append({"T": round(s.T, 6), "n": int(s.K.size), "n_failed": int((~ok).sum()),
                     "rmse": float(np.sqrt(np.mean(e[ok] ** 2))), "bias": float(np.mean(e[ok])),
                     "inside": float(np.mean(inside[ok]))})
        e_all.append(e[ok])
    e_all = np.concatenate(e_all)
    pooled = {"rmse_pooled": float(np.sqrt(np.mean(e_all ** 2))),
              "rmse_mean_slice": float(np.sqrt(np.mean([r["rmse"] ** 2 for r in rows])))}
    return rows, pooled


def rbergomi_eval(slices, xi0, H, eta, rho, *, reps=EVAL_REPS, n_paths=EVAL_PATHS, seed0=1000, min_steps=MIN_STEPS):
    """Model vols at every quoted strike from `reps` independent replications (fresh seeds) on the calibration grid;
    price standard errors from the spread across replications, converted to vol by the Black vega."""
    from voldesk.surface import lsv as _lsv
    acc = {s.T: [] for s in slices}
    for r in range(reps):
        eng = RBergomiMC([s.T for s in slices], xi0, n_paths=n_paths, steps_per_year=STEPS, min_steps=min_steps,
                         seed=seed0 + r)
        pr = eng.prices(H, eta, rho, {s.T: s.k for s in slices})
        for s in slices:
            acc[s.T].append(pr[s.T])
        del eng
    iv, se_iv = {}, {}
    for s in slices:
        P = np.array(acc[s.T])
        iv[s.T], se_iv[s.T] = _lsv.otm_iv(P.mean(axis=0), s.k, s.T, P.std(axis=0, ddof=1) / np.sqrt(reps))
    return iv, se_iv


def grid_check(slices, xi0, H, eta, rho, *, reps: int = 2, n_paths: int = 2 ** 13, seed0: int = 77) -> dict:
    """Paired time-grid check: the calibration grid against one twice as fine, both computed from the same simulated
    paths (the coarse scheme uses every second node), at the calibrated parameters. Returns |vol difference| over
    the quotes (median, 95th percentile, max) and the pooled RMSE against the mids on each grid."""
    from voldesk.surface import lsv as _lsv
    acc = {"coarse": {s.T: [] for s in slices}, "fine": {s.T: [] for s in slices}}
    kq = {s.T: s.k for s in slices}
    for r in range(reps):
        eng = RBergomiMC([s.T for s in slices], xi0, n_paths=n_paths, steps_per_year=STEPS, min_steps=MIN_STEPS,
                         seed=seed0 + r, refine=2)
        for key, coarse in (("coarse", True), ("fine", False)):
            pr = eng.prices(H, eta, rho, kq, coarse=coarse)
            for s in slices:
                acc[key][s.T].append(pr[s.T])
        del eng
    iv = {key: {s.T: _lsv.otm_iv(np.mean(acc[key][s.T], axis=0), s.k, s.T) for s in slices} for key in acc}
    d = np.abs(np.concatenate([(iv["coarse"][s.T] - iv["fine"][s.T]) * 100 for s in slices]))
    d = d[np.isfinite(d)]
    return {"grid_dvol_median": float(np.median(d)), "grid_dvol_p95": float(np.percentile(d, 95)),
            "grid_dvol_max": float(np.max(d)),
            "grid_rmse_coarse": quote_errors(iv["coarse"], slices)[1]["rmse_pooled"],
            "grid_rmse_fine": quote_errors(iv["fine"], slices)[1]["rmse_pooled"]}


def load_short(date: str):
    """(cached panel record, short-dated slices, eSSVI maturities, variance-swap and ATM total variances)."""
    from surface_panel import load_chain
    from voldesk.surface.slices import slices_from_chain
    r = json.loads((config.CACHE / "surface_panel" / "monthend" / f"{date}.json").read_text(encoding="utf-8"))
    all_sl = slices_from_chain(load_chain(pd.Timestamp(date)), etypes=("standard_am",))
    if [round(s.T, 10) for s in all_sl] != [round(s["T"], 10) for s in r["slices"]]:
        raise RuntimeError(f"{date}: slices differ from the cached fit; rerun scripts/surface_panel.py")
    es = r["models"]["essvi"]["params"]["slices"]
    Ts = np.array([s["T"] for s in es])
    w_vs = np.array([variance_swap_total_variance(svi.ESSVISlice(s["theta"], s["rho"], s["psi"]).w) for s in es])
    w_atm = np.array([s["theta"] for s in es])
    return r, [s for s in all_sl if s.T <= T_SHORT], Ts, w_vs, w_atm


def rbergomi_part(dates) -> None:
    slice_rows, fit_rows = [], []
    for date in dates:
        r, short, Ts, w_vs, w_atm = load_short(date)
        curves = {"vs": forward_variance_curve(Ts, w_vs), "atm": forward_variance_curve(Ts, w_atm)}
        print(f"{date}: {len(short)} short slices, {sum(s.K.size for s in short)} quotes; VS vol / ATM vol at the "
              f"first slices: {np.round(np.sqrt(w_vs[:len(short)] / Ts[:len(short)]) * 100, 2)} / "
              f"{np.round(np.sqrt(w_atm[:len(short)] / Ts[:len(short)]) * 100, 2)}", flush=True)
        models = {}
        for name, xi0 in curves.items():
            t0 = time.time()
            fit, eng = calibrate_rbergomi(short, xi0, steps_per_year=STEPS, min_steps=MIN_STEPS, n_polish=2)
            steps = int(eng.t.size)
            del eng
            sec = time.time() - t0
            iv, se_iv = rbergomi_eval(short, xi0, fit.H, fit.eta, fit.rho)
            se_all = np.concatenate([se_iv[s.T] * 100 for s in short])
            g = grid_check(short, xi0, fit.H, fit.eta, fit.rho)
            models[f"rBergomi ({name} xi0)"] = (iv, {"H": fit.H, "eta": fit.eta, "rho": fit.rho, "nfev": fit.nfev,
                                                    "steps": steps, "seconds": sec,
                                                    "mc_se_median": float(np.nanmedian(se_all)),
                                                    "mc_se_p95": float(np.nanpercentile(se_all, 95)), **g})
            print(f"  rBergomi {name}: H {fit.H:.3f} eta {fit.eta:.2f} rho {fit.rho:.3f} ({sec:.0f}s, {fit.nfev} evals, "
                  f"{steps} steps); grid check { {k: round(v, 3) for k, v in g.items()} }", flush=True)
        t0 = time.time()
        hf = hs.fit_heston(short)
        sec = time.time() - t0
        p = hf.params
        models["Heston (refit to these quotes)"] = ({s.T: hs.heston_iv(s.F, s.K, s.T, s.D, p, method="quad") for s in short},
                                                    {"v0": p.v0, "kappa": p.kappa, "theta": p.theta, "sigma": p.sigma,
                                                     "rho": p.rho, "feller": p.feller_ratio, "seconds": sec})
        hp = r["models"]["heston"]["params"]
        pp = hs.HestonParams(hp["v0"], hp["kappa"], hp["theta"], hp["sigma"], hp["rho"])
        models["Heston (panel, whole surface)"] = ({s.T: hs.heston_iv(s.F, s.K, s.T, s.D, pp, method="quad") for s in short},
                                                   {"v0": pp.v0, "kappa": pp.kappa, "theta": pp.theta,
                                                    "sigma": pp.sigma, "rho": pp.rho, "feller": pp.feller_ratio})
        for name, (iv, par) in models.items():
            rows, pooled = quote_errors(iv, short)
            for row in rows:
                slice_rows.append({"date": date, "model": name, **row})
            fit_rows.append({"date": date, "model": name, "n_slices": len(short),
                             "n_quotes": int(sum(s.K.size for s in short)), **pooled, **par})
            print(f"  {name:34s} RMSE pooled {pooled['rmse_pooled']:.3f}  per slice "
                  f"{[round(x['rmse'], 2) for x in rows]}", flush=True)
        # panel surface fits on the same quotes (whole-surface fits, for reference)
        for m in ("svi", "essvi", "sabr"):
            label = f"{m.upper() if m != 'essvi' else 'eSSVI'} (panel)"
            st = [x for x in r["models"][m]["slices"] if x["T"] <= T_SHORT]
            for x in st:
                slice_rows.append({"date": date, "model": label, "T": round(x["T"], 6), "n": x["n"],
                                   "n_failed": x.get("n_failed", 0), "rmse": x["rmse"], "bias": np.nan,
                                   "inside": x["inside"]})
            e2 = sum(x["rmse"] ** 2 * x["n"] for x in st) / sum(x["n"] for x in st)
            fit_rows.append({"date": date, "model": label, "n_slices": len(st), "n_quotes": int(sum(x["n"] for x in st)),
                             "rmse_pooled": float(np.sqrt(e2)),
                             "rmse_mean_slice": float(np.sqrt(np.mean([x["rmse"] ** 2 for x in st])))})
    pd.DataFrame(slice_rows).to_csv(OUT / "rbergomi_slices.csv", index=False, float_format="%.5g")
    pd.DataFrame(fit_rows).to_csv(OUT / "rbergomi_fits.csv", index=False, float_format="%.5g")
    print("written", OUT / "rbergomi_slices.csv", OUT / "rbergomi_fits.csv")


def convergence_part(dates, n_max: int = 1600, seeds: int = 4, n_paths: int = 2 ** 12) -> None:
    """How fast the left-point scheme converges at the shortest maturity: the date's calibrated rough Bergomi (vs xi0)
    on [0, T1] with n_max uniform steps, and the same paths on every 2nd, 4th, ... node (nested grids down to
    n_max / 32). Strikes at -2, -1, 0, 0.5 and 1 variance-swap standard deviations. Writes rbergomi_convergence.csv."""
    from voldesk.pricing.bsm import black_price
    from voldesk.pricing.implied_vol import implied_vol_black
    from voldesk.surface.rough import rbergomi_cov
    fits = pd.read_csv(OUT / "rbergomi_fits.csv")
    rows = []
    for date in dates:
        _, short, Ts, w_vs, _ = load_short(date)
        f = fits[(fits["date"] == date) & (fits["model"] == "rBergomi (vs xi0)")].iloc[0]
        H, eta, rho = float(f["H"]), float(f["eta"]), float(f["rho"])
        T = float(short[0].T)
        xi = float(forward_variance_curve(Ts, w_vs)(0.0))          # xi0 is flat on [0, T1]
        t = np.linspace(T / n_max, T, n_max)
        L = np.linalg.cholesky(rbergomi_cov(t, H, 1.0) + 1e-13 * np.eye(2 * n_max))
        z = np.array([-2.0, -1.0, 0.0, 0.5, 1.0])
        k = z * np.sqrt(xi * T)
        cp = np.where(k >= 0, 1.0, -1.0)
        levels = [n_max // 2 ** j for j in range(6)][::-1]
        acc = {n: [] for n in levels}
        tt = np.concatenate([[0.0], t])
        for sd in range(seeds):
            Z = np.random.default_rng(2000 + sd).standard_normal((n_paths, 2 * n_max))
            G = np.concatenate([Z, -Z]) @ L.T
            V0 = np.concatenate([np.zeros((G.shape[0], 1)), G[:, :n_max]], axis=1)
            W0 = np.concatenate([np.zeros((G.shape[0], 1)), G[:, n_max:]], axis=1)
            v = xi * np.exp(eta * V0 - 0.5 * eta ** 2 * tt ** (2 * H))
            for n in levels:
                idx = np.arange(0, n_max + 1, n_max // n)
                I = np.sum(np.sqrt(v[:, idx[:-1]]) * np.diff(W0[:, idx], axis=1), axis=1)
                Q = np.sum(v[:, idx[:-1]] * (T / n), axis=1)
                F = np.exp(rho * I - 0.5 * rho * rho * Q)
                sdc = np.sqrt(np.maximum((1 - rho * rho) * Q, 0.0))
                acc[n].append(black_price(cp[None, :], F[:, None], np.exp(k)[None, :], 1.0, sdc[:, None], 1.0)
                              .mean(axis=0))
            del G, V0, W0, v
        for n in levels:
            iv = implied_vol_black(np.mean(acc[n], axis=0), cp, 1.0, np.exp(k), T, 1.0) * 100
            for zz, x in zip(z, iv):
                rows.append({"date": date, "T": round(T, 6), "steps": n, "steps_per_year": round(n / T), "z": zz,
                             "iv": float(x)})
        print(date, pd.DataFrame([r for r in rows if r["date"] == date]).pivot(index="steps", columns="z", values="iv")
              .round(3).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rbergomi_convergence.csv", index=False, float_format="%.6g")


def xi0_part(dates) -> None:
    """Variance-swap vol minus ATM implied vol of the eSSVI slices used (vol points): how far the two readings of the
    forward variance level are apart. Writes rbergomi_xi0_gap.csv (differences only, no vol levels)."""
    rows = []
    for date in dates:
        _, short, Ts, w_vs, w_atm = load_short(date)
        for T, a, b in zip(Ts[:len(short)], w_vs[:len(short)], w_atm[:len(short)]):
            rows.append({"date": date, "T": round(float(T), 6),
                         "vs_minus_atm_volpts": float((np.sqrt(a / T) - np.sqrt(b / T)) * 100)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "rbergomi_xi0_gap.csv", index=False, float_format="%.4g")
    print(d.round(2).to_string(index=False))


def coarse_part(dates) -> None:
    """The same calibration (variance-swap xi0) on a uniform grid of 1,460 steps a year, without the concentration at
    short maturities, evaluated on that grid: what a too-coarse grid does to the fit. Writes rbergomi_coarse.csv."""
    rows = []
    for date in dates:
        _, short, Ts, w_vs, _ = load_short(date)
        xi0 = forward_variance_curve(Ts, w_vs)
        t0 = time.time()
        fit, eng = calibrate_rbergomi(short, xi0, steps_per_year=STEPS, min_steps=0, n_polish=2)
        steps = int(eng.t.size)
        del eng
        iv, _ = rbergomi_eval(short, xi0, fit.H, fit.eta, fit.rho, min_steps=0)
        sl_rows, pooled = quote_errors(iv, short)
        rows.append({"date": date, "model": "rBergomi (vs xi0), uniform 1,460 steps a year", "H": fit.H, "eta": fit.eta,
                     "rho": fit.rho, "steps": steps, "first_expiry_steps": int(np.ceil(short[0].T * STEPS - 1e-9)),
                     **pooled, **{f"rmse_slice{i + 1}": x["rmse"] for i, x in enumerate(sl_rows)},
                     "seconds": time.time() - t0})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rbergomi_coarse.csv", index=False, float_format="%.5g")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["skew", "rbergomi", "convergence", "xi0", "coarse", "all"], default="all")
    ap.add_argument("--dates", nargs="*", default=["2017-06-30", "2020-03-31"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part in ("skew", "all"):
        skew_part()
    if args.part in ("rbergomi", "all"):
        rbergomi_part(args.dates)
    if args.part in ("convergence", "all"):
        convergence_part(args.dates)
    if args.part in ("xi0", "all"):
        xi0_part(args.dates)
    if args.part in ("coarse", "all"):
        coarse_part(args.dates)


if __name__ == "__main__":
    main()
