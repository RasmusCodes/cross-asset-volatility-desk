"""Fit every surface model to the same slices and measure them the same way.

Models
  svi     raw SVI per slice under butterfly constraints, calendar order imposed where cheap (svi.fit_svi_surface)
  ssvi    power-law SSVI, one joint fit, arbitrage-free by its sufficient conditions (svi.fit_ssvi)
  essvi   extended SSVI, slice by slice, arbitrage-free by construction (svi.fit_essvi_surface)
  sabr    SABR per slice with beta = 1 (Hagan's lognormal formula), slices joined linearly in total variance
  heston  one joint Heston fit (COS pricing during calibration)

Measurements (identical for every model)
  fit         per slice: RMSE and mean absolute error of model minus mid vol (vol points), share of quotes whose
              model vol lies inside [bid vol, ask vol], largest miss
  butterfly   Gatheral-Jacquier g(k, T) on a grid (k in [-1.5, 0.6]); g < 0 means a negative density. Reported
              separately inside the quoted moneyness range of each maturity and outside it (extrapolation)
  calendar    total variance falling between consecutive grid maturities at fixed k (dw < -1e-8)
  local vol   Dupire local variance dw/dT / g: the share of the grid where it is positive and finite
  wings       |dw/dk| at k = -1.5 and +0.6 against Roger Lee's moment bound of 2
Derivatives come from each surface's analytical expressions where it has them (SVI, SSVI, eSSVI slices) and from
central differences of its total variance otherwise (SABR, Heston), with the same grid for all.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import heston as hs
from . import sabr as sb
from . import svi

MODELS = ("svi", "ssvi", "essvi", "sabr", "heston")
K_GRID = np.linspace(-1.5, 0.6, 211)


# ------------------------------------------------------------------------------------------------ surfaces
class SliceInterpSurface:
    """Per-slice total-variance functions joined linearly in T at fixed k (flat vol before the first slice and after
    the last), the interpolation SVISurface uses; for models fitted slice by slice without analytical derivatives."""

    def __init__(self, Ts, wfuncs):
        o = np.argsort(Ts)
        self.Ts = np.asarray(Ts, float)[o]
        self.wf = [wfuncs[i] for i in o]

    def w(self, k, T):
        k = np.asarray(k, float)
        T = float(T)
        if T <= self.Ts[0]:
            return self.wf[0](k) * T / self.Ts[0]
        if T >= self.Ts[-1]:
            return self.wf[-1](k) * T / self.Ts[-1]
        j = int(np.searchsorted(self.Ts, T))
        i = j - 1
        lam = (T - self.Ts[i]) / (self.Ts[j] - self.Ts[i])
        return (1 - lam) * self.wf[i](k) + lam * self.wf[j](k)

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)


class HestonSurface:
    """Black implied vols of a Heston model in log forward moneyness (independent of F and D)."""

    def __init__(self, p: hs.HestonParams):
        self.p = p

    def iv(self, k, T):
        k = np.atleast_1d(np.asarray(k, float))
        return hs.heston_iv(1.0, np.exp(k), float(T), 1.0, self.p)

    def w(self, k, T):
        return self.iv(k, T) ** 2 * float(T)


@dataclass
class ModelFit:
    model: str
    params: dict
    slices: list
    seconds: float
    success: bool
    surface: object = field(repr=False, default=None)
    arbitrage: dict = field(default_factory=dict)


def _fit_stats(model_iv: np.ndarray, s) -> dict:
    e = (np.asarray(model_iv, float) - s.iv_mid) * 100
    ok = np.isfinite(e)
    inside = (model_iv >= s.iv_bid - 1e-12) & (model_iv <= s.iv_ask + 1e-12)
    return {"T": float(s.T), "expiry": str(np.datetime_as_string(np.datetime64(s.expiration), unit="D")),
            "n": int(s.K.size), "rmse": float(np.sqrt(np.mean(e[ok] ** 2))) if ok.any() else np.nan,
            "mae": float(np.mean(np.abs(e[ok]))) if ok.any() else np.nan,
            "max_abs": float(np.max(np.abs(e[ok]))) if ok.any() else np.nan,
            "inside": float(np.mean(inside[ok])) if ok.any() else np.nan,
            "spread_med": float(np.median(s.iv_spread) * 100), "n_failed": int((~ok).sum())}


# ------------------------------------------------------------------------------------------------ fitters
def fit_svi_model(slices) -> ModelFit:
    t0 = time.time()
    surf, fits = svi.fit_svi_surface(slices)
    rows = []
    for s, f in zip(sorted(slices, key=lambda x: x.T), fits):
        st = _fit_stats(np.sqrt(np.maximum(f.params.w(s.k), 0) / s.T), s)
        st.update({"a": f.params.a, "b": f.params.b, "rho": f.params.rho, "m": f.params.m, "s": f.params.s,
                   "success": f.success, "min_g_fit": f.min_g})
        rows.append(st)
    params = {"slices": [{k: r[k] for k in ("T", "expiry", "a", "b", "rho", "m", "s")} for r in rows]}
    return ModelFit("svi", params, rows, time.time() - t0, all(r["success"] for r in rows), surf)


def fit_essvi_model(slices) -> ModelFit:
    t0 = time.time()
    surf, fits = svi.fit_essvi_surface(slices)
    rows = []
    for s, f in zip(sorted(slices, key=lambda x: x.T), fits):
        st = _fit_stats(np.sqrt(np.maximum(f.slice.w(s.k), 0) / s.T), s)
        st.update({"theta": f.slice.theta, "rho": f.slice.rho, "psi": f.slice.psi, "success": f.success})
        rows.append(st)
    params = {"slices": [{k: r[k] for k in ("T", "expiry", "theta", "rho", "psi")} for r in rows]}
    return ModelFit("essvi", params, rows, time.time() - t0, all(r["success"] for r in rows), surf)


def fit_ssvi_model(slices) -> ModelFit:
    t0 = time.time()
    res = svi.fit_ssvi(slices)
    surf = res.surface
    rows = [_fit_stats(surf.iv(s.k, s.T), s) for s in sorted(slices, key=lambda x: x.T)]
    params = {"rho": surf.p.rho, "eta": surf.p.eta, "gamma": surf.p.gamma, "Ts": surf.Ts.tolist(),
              "thetas": surf.thetas.tolist()}
    return ModelFit("ssvi", params, rows, time.time() - t0, res.success, surf)


def fit_sabr_model(slices, beta: float = 1.0) -> ModelFit:
    t0 = time.time()
    rows, Ts, wfs, ok = [], [], [], True
    for s in sorted(slices, key=lambda x: x.T):
        f = sb.fit_sabr_slice(s.K, s.F, s.T, s.iv_mid, iv_bid=s.iv_bid, iv_ask=s.iv_ask, beta=beta)
        st = _fit_stats(f.iv(s.K), s)
        st.update({"alpha": f.alpha, "beta": beta, "rho": f.rho, "nu": f.nu, "success": f.success})
        rows.append(st)
        ok &= f.success
        Ts.append(s.T)
        wfs.append(lambda k, f=f, F=s.F, T=s.T: np.maximum(f.iv(F * np.exp(np.asarray(k, float))), 0.0) ** 2 * T)
    params = {"slices": [{k: r[k] for k in ("T", "expiry", "alpha", "beta", "rho", "nu")} for r in rows]}
    return ModelFit("sabr", params, rows, time.time() - t0, ok, SliceInterpSurface(Ts, wfs))


def fit_heston_model(slices, quad_check: bool = True) -> ModelFit:
    t0 = time.time()
    res = hs.fit_heston(slices)
    p = res.params
    rows = []
    for s in sorted(slices, key=lambda x: x.T):
        iv = hs.heston_iv(s.F, s.K, s.T, s.D, p, method="quad" if quad_check else "cos")
        rows.append(_fit_stats(iv, s))
    params = {"v0": p.v0, "kappa": p.kappa, "theta": p.theta, "sigma": p.sigma, "rho": p.rho,
              "feller": p.feller_ratio}
    return ModelFit("heston", params, rows, time.time() - t0, res.success, HestonSurface(p))


FITTERS = {"svi": fit_svi_model, "ssvi": fit_ssvi_model, "essvi": fit_essvi_model, "sabr": fit_sabr_model,
           "heston": fit_heston_model}


# ------------------------------------------------------------------------------------------------ diagnostics
def _derivs(surface, k, T, h: float = 1e-3):
    if hasattr(surface, "dw_dk") and hasattr(surface, "d2w_dk2"):
        return surface.w(k, T), surface.dw_dk(k, T), surface.d2w_dk2(k, T)
    w0 = surface.w(k, T)
    wp, wm = surface.w(k + h, T), surface.w(k - h, T)
    return w0, (wp - wm) / (2 * h), (wp - 2 * w0 + wm) / (h * h)


def arbitrage_metrics(surface, slices, k_grid=K_GRID, n_between: int = 2) -> dict:
    """Butterfly, calendar, local-vol and wing diagnostics on a common (k, T) grid (see module docstring)."""
    sl = sorted(slices, key=lambda x: x.T)
    Tn = np.array([s.T for s in sl])
    T_grid = [Tn[0]]
    for a, b in zip(Tn[:-1], Tn[1:]):
        T_grid += list(np.linspace(a, b, n_between + 2)[1:])
    T_grid = np.array(T_grid)
    lo = np.interp(T_grid, Tn, [s.k.min() for s in sl])
    hi = np.interp(T_grid, Tn, [s.k.max() for s in sl])
    g_in, g_out, lv_ok, n_lv = [], [], 0, 0
    W = []
    for T, a, b in zip(T_grid, lo, hi):
        w, d1, d2 = _derivs(surface, k_grid, T)
        w = np.maximum(w, 1e-12)
        g = svi.g_function(k_grid, w, d1, d2)
        inside = (k_grid >= a) & (k_grid <= b)
        g_in.append(g[inside])
        g_out.append(g[~inside])
        W.append(w)
    W = np.array(W)
    dW = np.diff(W, axis=0)
    dT = np.diff(T_grid)[:, None]
    cal_viol = dW < -1e-8
    both_quoted = np.array([(k_grid >= max(lo[i], lo[i + 1])) & (k_grid <= min(hi[i], hi[i + 1]))
                            for i in range(len(T_grid) - 1)])
    for i in range(len(T_grid) - 1):
        w, d1, d2 = _derivs(surface, k_grid, T_grid[i])
        g = svi.g_function(k_grid, np.maximum(w, 1e-12), d1, d2)
        lv = (dW[i] / dT[i]) / g
        n_lv += lv.size
        lv_ok += int(np.sum(np.isfinite(lv) & (lv > 0)))
    gi = np.concatenate(g_in) if g_in else np.array([np.nan])
    go = np.concatenate(g_out) if g_out else np.array([np.nan])
    wl = _derivs(surface, np.array([-1.5, 0.6]), float(Tn[-1]))[1]
    return {"min_g_quoted": float(np.nanmin(gi)) if gi.size else np.nan,
            "share_g_neg_quoted": float(np.mean(gi < -1e-8)) if gi.size else np.nan,
            "min_g_extrap": float(np.nanmin(go)) if go.size else np.nan,
            "share_g_neg_extrap": float(np.mean(go < -1e-8)) if go.size else np.nan,
            "share_calendar_viol": float(cal_viol.mean()), "worst_calendar_dw": float(dW.min()),
            "share_calendar_viol_quoted": float(cal_viol[both_quoted].mean()) if both_quoted.any() else np.nan,
            "share_calendar_viol_extrap": float(cal_viol[~both_quoted].mean()) if (~both_quoted).any() else np.nan,
            "share_localvar_valid": lv_ok / max(n_lv, 1),
            "wing_slope_left": float(abs(wl[0])), "wing_slope_right": float(abs(wl[1]))}


def fit_all(slices, models=MODELS, arbitrage: bool = True) -> dict[str, ModelFit]:
    out = {}
    for m in models:
        try:
            f = FITTERS[m](slices)
            if arbitrage and f.surface is not None:
                f.arbitrage = arbitrage_metrics(f.surface, slices)
        except Exception as e:  # noqa: BLE001 - a failed model is recorded, not fatal
            f = ModelFit(m, {"error": str(e)[:300]}, [], 0.0, False)
        out[m] = f
    return out
