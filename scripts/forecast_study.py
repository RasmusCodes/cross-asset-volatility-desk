"""Forecasting study: realized vol, vol of vol, implied vol, correlation and regime transitions, all walk-forward
(expanding windows, refits every 21 trading days unless stated, training rows only where the target is already
observed at the refit date). Economic value is in scripts/forecast_economics.py.

    python scripts/forecast_study.py --section rv|vov|iv|corr|regime|all

Outputs: results/forecast/<section>_*.csv (loss tables, tests); forecasts in data/cache/forecast/<section>.parquet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402
from voldesk.forecast import correlation as cr  # noqa: E402
from voldesk.forecast import datasets as dsets  # noqa: E402
from voldesk.forecast import evaluate as ev  # noqa: E402
from voldesk.forecast import levels as lv  # noqa: E402
from voldesk.forecast import models as fm  # noqa: E402
from voldesk.forecast import regimes as rg  # noqa: E402
from voldesk.forecast.realized import forward_sum  # noqa: E402

OUT = config.RESULTS / "forecast"
CACHE = config.CACHE / "forecast"
EVERY = 21


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def save(df: pd.DataFrame, name: str, index=True):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=index, float_format="%.6g")


# ================================================================================================ helpers
def linear_fp(X: np.ndarray, y: np.ndarray, log_model: bool, floor_series: np.ndarray | None = None):
    def fp(train, pred):
        Xt, yt = X[train], y[train]
        ok = np.all(np.isfinite(Xt), axis=1) & np.isfinite(yt) & (yt > 0)
        if log_model:
            ok &= np.all(Xt > 0, axis=1)
        floor = None
        if floor_series is not None:
            fs = floor_series[train]
            floor = float(np.nanmin(fs[fs > 0])) if np.any(fs > 0) else None
        fit = fm.fit_linear(Xt[ok], yt[ok], log=log_model, floor=floor)
        Xp = X[pred]
        out = np.full(len(Xp), np.nan)
        okp = np.all(np.isfinite(Xp), axis=1) & (np.all(Xp > 0, axis=1) if log_model else True)
        if okp.any():
            out[okp] = fm.predict_linear(fit, Xp[okp])[0]
        return out
    return fp


def garch_fp(r: pd.Series, h: int, asym: bool):
    def fp(train, pred):
        fit = fm.fit_garch(r.iloc[train].dropna().to_numpy(), asym=asym)
        s_next = fm.garch_next(fit, r.iloc[: pred.stop]).to_numpy()[pred]
        return fm.garch_h_day(fit, s_next, h) / h
    return fp


def loss_table(fc: pd.DataFrame, target: pd.Series, h: int, start, base: str) -> pd.DataFrame:
    m = fc.index >= pd.Timestamp(start)
    F = fc[m]
    x = target[m]
    ok = x.notna() & F.notna().all(axis=1) & (F > 0).all(axis=1)
    F, x = F[ok], x[ok]
    L = pd.DataFrame({c: ev.qlike(F[c], x) for c in F.columns}, index=F.index)
    M = pd.DataFrame({c: ev.mse(F[c], x) for c in F.columns}, index=F.index)
    pos = x > 0
    P = pd.DataFrame({c: (x[pos] / F.loc[pos, c]) - np.log(x[pos] / F.loc[pos, c]) - 1 for c in F.columns})
    rows = []
    for c in F.columns:
        dm = ev.dm_test(L[base], L[c], max(h, 5)) if c != base else {"t": np.nan, "p": np.nan}
        mz = ev.mincer_zarnowitz(F[c], x)
        rows.append({"model": c, "h": h, "n": int(len(x)), "qlike": float(L[c].mean()),
                     "qlike_patton_vs_base_pct": 100 * (float(P[c].mean()) / float(P[base].mean()) - 1),
                     "mse_ratio": float(M[c].mean() / M[base].mean()), "dm_t_vs_base": dm["t"], "dm_p": dm["p"],
                     "mz_slope": mz["slope"], "mz_r2": mz["r2"]})
    t = pd.DataFrame(rows).set_index("model")
    mcs = ev.model_confidence_set(L, size=0.10, block=max(h, 22), reps=1000)
    return t.join(mcs)


# ================================================================================================ sections
def section_rv(start="2000-01-03"):
    d = dsets.spx_daily()
    vix = dsets.vix_daily()["vix"].reindex(d.index).ffill(limit=5)
    d = d[d.index >= "1996-01-04"]
    vix = vix.reindex(d.index)
    r, r2, pk = d["r"], d["r2"], d["parkinson"]
    iv2 = (vix / 100.0) ** 2 / 252.0
    rows, all_fc = [], {}
    for h in (1, 5, 22):
        y = forward_sum(r2, h) / h
        mean22 = r2.rolling(22, min_periods=22).mean().to_numpy()
        har_r2 = fm.har_components(r2)
        har_pk = fm.har_components(pk)
        fc = pd.DataFrame(index=d.index)
        fc["EWMA"] = fm.ewma_next(r2)
        fc["GARCH"] = ev.walk_forward(d.index, start, EVERY * 6, h, garch_fp(r, h, False))
        fc["GJR"] = ev.walk_forward(d.index, start, EVERY * 6, h, garch_fp(r, h, True))
        fc["HAR"] = ev.walk_forward(d.index, start, EVERY, h, linear_fp(har_r2.to_numpy(), y.to_numpy(), False, mean22))
        fc["logHAR_pk"] = ev.walk_forward(d.index, start, EVERY, h, linear_fp(har_pk.to_numpy(), y.to_numpy(), True))
        X = np.column_stack([har_r2.to_numpy(), iv2.to_numpy()])
        fc["HAR_VIX"] = ev.walk_forward(d.index, start, EVERY, h, linear_fp(X, y.to_numpy(), False, mean22))
        X = np.column_stack([har_pk.to_numpy(), iv2.to_numpy()])
        fc["logHAR_pk_VIX"] = ev.walk_forward(d.index, start, EVERY, h, linear_fp(X, y.to_numpy(), True))
        fc["VIX_affine"] = ev.walk_forward(d.index, start, EVERY, h,
                                           linear_fp(iv2.to_numpy()[:, None], y.to_numpy(), False, mean22))
        fc["VIX_raw"] = iv2
        t = loss_table(fc, y, h, start, "EWMA")
        rows.append(t.reset_index())
        all_fc[h] = fc.assign(target=y)
        log(f"rv h={h}: best QLIKE {t['qlike'].idxmin()} ({t['qlike_patton_vs_base_pct'].min():+.1f}% vs EWMA)")
    save(pd.concat(rows), "rv_losses", index=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat({h: f for h, f in all_fc.items()}, names=["h", "date"]).to_parquet(CACHE / "rv.parquet")


def section_vov(start="2008-01-02"):
    v = dsets.vix_daily()
    vv = dsets.vvol_implied()
    v = v[v.index >= "2006-03-01"]
    lr = np.log(v["vix"]).diff()
    lr2 = lr ** 2
    rng = (np.log(v["vixh"] / v["vixl"]) ** 2) / (4 * np.log(2))
    rng[(v["vixh"] <= v["vixl"]) | rng.isna()] = np.nan
    iv2 = (vv.reindex(v.index).ffill(limit=5)) ** 2 / 252.0
    rows, all_fc = [], {}
    for h in (5, 22):
        y = forward_sum(lr2, h) / h
        mean22 = lr2.rolling(22, min_periods=22).mean().to_numpy()
        H = fm.har_components(lr2)
        Hr = fm.har_components(rng)
        fc = pd.DataFrame(index=v.index)
        fc["EWMA"] = fm.ewma_next(lr2)
        fc["HAR"] = ev.walk_forward(v.index, start, EVERY, h, linear_fp(H.to_numpy(), y.to_numpy(), False, mean22))
        fc["logHAR_range"] = ev.walk_forward(v.index, start, EVERY, h, linear_fp(Hr.to_numpy(), y.to_numpy(), True))
        X = np.column_stack([Hr.to_numpy(), iv2.to_numpy()])
        fc["logHAR_range_IVV"] = ev.walk_forward(v.index, start, EVERY, h, linear_fp(X, y.to_numpy(), True))
        fc["IVV_affine"] = ev.walk_forward(v.index, start, EVERY, h,
                                           linear_fp(iv2.to_numpy()[:, None], y.to_numpy(), False, mean22))
        fc["IVV_raw"] = iv2
        t = loss_table(fc, y, h, start, "EWMA")
        rows.append(t.reset_index())
        all_fc[h] = fc.assign(target=y)
        log(f"vov h={h}: best {t['qlike'].idxmin()} ({t['qlike_patton_vs_base_pct'].min():+.1f}% vs EWMA)")
    save(pd.concat(rows), "vov_losses", index=False)
    pd.concat({h: f for h, f in all_fc.items()}, names=["h", "date"]).to_parquet(CACHE / "vov.parquet")
    # descriptive: implied vs realized vol of vol (annualized, %)
    real22 = np.sqrt(forward_sum(lr2, 22) / 22 * 252) * 100
    desc = pd.DataFrame({"implied_vvol30_pct": vv.reindex(v.index) * 100, "realized_next22_pct": real22}).dropna()
    save(desc.describe().T, "vov_implied_vs_realized")


def section_iv(start="2000-01-03"):
    d = dsets.spx_daily()
    vix = dsets.vix_daily()["vix"]
    atm = dsets.atm_term()
    fut = dsets.vix_futures_cm(maturities=(2, 8, 32))
    idx = vix.index[(vix.index >= "1996-01-04")]
    vix = vix.reindex(idx)
    rv22 = (d["r2"].rolling(22, min_periods=22).mean() * 252).reindex(idx)
    ret22 = np.log(d["close"]).diff(22).reindex(idx)
    slope = (atm["iv91"] - atm["iv30"]).reindex(idx).ffill(limit=5)
    vrp = (vix / 100) ** 2 - rv22
    rows, all_fc = [], {}
    cal = {1: 2, 5: 8, 22: 32}
    for h in (1, 5, 22):
        y_future = vix.shift(-h)
        Hl = lv.har_frame(np.log(vix), "lv")
        fc = pd.DataFrame(index=idx)
        fc["RW"] = vix
        fc["AR1_log"] = ev.walk_forward(idx, start, EVERY, h, lv.ols_forecaster(np.log(vix).to_numpy()[:, None],
                                                                                 y_future.to_numpy(), True))
        fc["HAR_log"] = ev.walk_forward(idx, start, EVERY, h, lv.ols_forecaster(Hl.to_numpy(), y_future.to_numpy(), True))
        X = np.column_stack([Hl.to_numpy(), vrp.to_numpy(), ret22.to_numpy(), slope.fillna(0).to_numpy()])
        fc["HARX_log"] = ev.walk_forward(idx, start, EVERY, h, lv.ols_forecaster(X, y_future.to_numpy(), True))
        f_h = fut[f"vx_cm{cal[h]}"].reindex(idx)
        fc["VIX_futures"] = f_h
        Xf = np.column_stack([np.log(f_h.to_numpy()), np.log(vix.to_numpy())])
        fc["futures_adj"] = ev.walk_forward(idx, "2007-01-03", EVERY, h, lv.ols_forecaster(Xf, y_future.to_numpy(), True))
        all_fc[h] = fc.assign(target=y_future)
        for st, cols in ((start, ["RW", "AR1_log", "HAR_log", "HARX_log"]),
                         ("2007-01-03", ["RW", "AR1_log", "HAR_log", "HARX_log", "VIX_futures", "futures_adj"])):
            t, _meta = lv.evaluate_levels(fc[cols], y_future, vix, h, st, base="RW")
            t = t.reset_index()
            t["h"], t["sample_start"] = h, st
            rows.append(t)
        log(f"iv h={h}: done")
    save(pd.concat(rows), "iv_vix_levels", index=False)
    pd.concat({h: f for h, f in all_fc.items()}, names=["h", "date"]).to_parquet(CACHE / "iv.parquet")


def section_corr(start="2000-01-03"):
    ret, weights, atm_names = dsets.names_panel()
    spx_iv = dsets.atm_term()["iv30"]
    ret = ret[ret.index >= "1996-01-04"]
    log("corr: computing realized correlations")
    rc22 = cr.realized_corr(ret, weights, 22)
    rc66 = cr.realized_corr(ret, weights, 66)
    rc5 = cr.realized_corr(ret, weights, 5)
    target = cr.forward_corr(ret, weights, 22)
    ic = cr.implied_corr(spx_iv, atm_names, weights).reindex(ret.index)
    P = pd.DataFrame({"rc5": rc5, "rc22": rc22, "rc66": rc66, "implied": ic, "target": target})
    P.to_parquet(CACHE / "corr_inputs.parquet")
    fc = pd.DataFrame(index=P.index)
    fc["RW_rc22"] = P["rc22"]
    fc["rc66"] = P["rc66"]
    X = P[["rc5", "rc22", "rc66"]].to_numpy()
    fc["HAR"] = ev.walk_forward(P.index, start, EVERY, 22, lv.ols_forecaster(X, P["target"].to_numpy(), False))
    fc["implied_raw"] = P["implied"]
    fc["implied_adj"] = ev.walk_forward(P.index, start, EVERY, 22,
                                        lv.ols_forecaster(P[["implied"]].to_numpy(), P["target"].to_numpy(), False))
    X = P[["rc5", "rc22", "rc66", "implied"]].to_numpy()
    fc["HAR_implied"] = ev.walk_forward(P.index, start, EVERY, 22, lv.ols_forecaster(X, P["target"].to_numpy(), False))
    t, _meta = lv.evaluate_levels(fc, P["target"], P["rc22"], 22, start, base="RW_rc22")
    t = t.reset_index()
    save(t, "corr_forecasts", index=False)
    crp = (P["implied"] - P["target"]).dropna()
    save(pd.DataFrame({"implied_minus_realized_next22": crp.describe(),
                       "implied": P["implied"].dropna().describe(), "realized_next22": P["target"].dropna().describe()}),
         "corr_premium")
    fc.assign(target=P["target"]).to_parquet(CACHE / "corr.parquet")
    log("corr: done")


def section_regime(start="2000-01-03", threshold=0.25):
    d = dsets.spx_daily()
    d = d[d.index >= "1996-01-04"]
    vix = dsets.vix_daily()["vix"].reindex(d.index).ffill(limit=5)
    x = 100 * d["r"]
    fwd = np.sqrt(forward_sum(d["r2"], 22) / 22 * 252)
    y = (fwd > threshold).astype(float).where(fwd.notna())
    rv22 = np.sqrt(d["r2"].rolling(22).mean() * 252)
    rv5 = np.sqrt(d["r2"].rolling(5).mean() * 252)
    # HMM filtered probabilities, refitted once a year on data up to the refit date (no look-ahead)
    hp = pd.Series(np.nan, index=d.index)
    for yr in range(pd.Timestamp(start).year, d.index.max().year + 1):
        a = pd.Timestamp(f"{yr}-01-01")
        tr = x[x.index < a].dropna()
        m = rg.fit_hmm2(tr.to_numpy())
        seg = x[x.index < pd.Timestamp(f"{yr + 1}-01-01")]
        p = rg.filtered_high_prob(seg.to_numpy(), m)
        mask = seg.index >= a
        hp.loc[seg.index[mask]] = p[mask]
    feats = pd.DataFrame({"lvix": np.log(vix), "lrv22": np.log(rv22), "lrv5": np.log(rv5), "hmm": hp}, index=d.index)
    probs = pd.DataFrame(index=d.index)
    specs = {"base_rate": [], "logit_vix": ["lvix"], "logit_rv": ["lrv22", "lrv5"], "logit_hmm": ["hmm"],
             "logit_vix_hmm": ["lvix", "hmm"], "logit_all": ["lvix", "lrv22", "lrv5", "hmm"]}
    pts = ev.refit_points(d.index, start, EVERY)
    for name, cols in specs.items():
        out = np.full(len(d), np.nan)
        for j, k in enumerate(pts):
            end = pts[j + 1] if j + 1 < len(pts) else len(d)
            tr = slice(0, k - 22 + 1)
            yt = y.iloc[tr]
            if not cols:
                out[k:end] = float(np.nanmean(yt))
                continue
            Xt = feats[cols].iloc[tr]
            ok = yt.notna() & Xt.notna().all(axis=1)
            Xp = feats[cols].iloc[k:end]
            okp = Xp.notna().all(axis=1).to_numpy()
            pr = np.full(end - k, np.nan)
            if ok.sum() > 200 and okp.any():
                pr[okp] = rg.logistic_fit_predict(Xt[ok].to_numpy(), yt[ok].to_numpy(), Xp[okp].to_numpy())
            out[k:end] = pr
        probs[name] = out
    m = (probs.index >= start) & y.notna() & probs.notna().all(axis=1)
    calm = m & (rv22 < threshold)
    rows = []
    for subset, mm in (("all days", m), ("calm days (trailing 22d vol < 25%)", calm)):
        for c in probs.columns:
            rows.append({"subset": subset, "model": c, "n": int(mm.sum()), "event_rate": float(y[mm].mean()),
                         "brier": rg.brier(probs.loc[mm, c], y[mm]), "log_loss": rg.log_loss(probs.loc[mm, c], y[mm]),
                         "auc": rg.auc(probs.loc[mm, c], y[mm])})
    save(pd.DataFrame(rows), "regime_scores", index=False)
    probs.assign(y=y, fwd_vol=fwd).to_parquet(CACHE / "regime.parquet")
    log("regime: done")


SECTIONS = {"rv": section_rv, "vov": section_vov, "iv": section_iv, "corr": section_corr, "regime": section_regime}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=list(SECTIONS) + ["all"])
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, fn in SECTIONS.items():
        if a.section in (name, "all"):
            t0 = time.time()
            fn()
            log(f"{name}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
