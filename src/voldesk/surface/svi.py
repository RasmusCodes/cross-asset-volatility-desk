"""SVI and SSVI total-variance parameterisations and their no-arbitrage conditions.

Notation: k = ln(K/F_T) (log forward moneyness), w(k, T) = sigma_BS(k, T)^2 T (total implied variance).

Raw SVI slice (Gatheral 2004)
    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2)),   parameters (a, b, rho, m, s).
    Validity: b >= 0, |rho| < 1, s > 0, a + b s sqrt(1 - rho^2) >= 0 (w >= 0 everywhere).
    Lee's moment bound: wing slopes b (1 +- rho) <= 2.

Butterfly condition (Gatheral & Jacquier 2014, Lemma 2.2): a slice is free of butterfly arbitrage iff
    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2 >= 0 for all k
(plus the call price vanishing as k -> inf, implied by wing slopes < 2). The risk-neutral density of
k is p(k) = g(k) exp(-d_-^2/2) / sqrt(2 pi w), d_- = -k/sqrt(w) - sqrt(w)/2.
Calendar condition: w(k, T) non-decreasing in T for every k.

SSVI surface (Gatheral & Jacquier 2014)
    w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),  phi = phi(theta),
    with theta_T the ATM total variance (non-decreasing in T) and the power law
    phi(theta) = eta theta^-gamma (1 + theta)^(gamma - 1).
    With gamma in (0, 1/2] and eta (1 + |rho|) <= 2 the paper's sufficient conditions hold for every
    theta: theta phi (1 + |rho|) < 2 < 4, theta phi^2 (1 + |rho|) <= 4, and
    d(theta phi)/d theta = (1 - gamma) phi / (1 + theta) <= (1 + sqrt(1 - rho^2)) phi / rho^2.
    The fit enforces those bounds; the tests also check g >= 0 and dw/dT >= 0 numerically.

Surfaces: ``SVISurface`` interpolates fitted raw-SVI slices linearly in total variance at fixed k
(this preserves calendar monotonicity of the slices); before the first slice w scales with T
(constant implied vol), after the last slice the implied vol is held flat. ``SSVISurface`` interpolates
theta linearly in T with the same extrapolation. Both expose w, dw/dk, d2w/dk2 and dw/dT, which is what
the Dupire local-volatility formula needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize

SQRT2PI = np.sqrt(2 * np.pi)


# ------------------------------------------------------------------------------------------------
# Raw SVI
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def w(self, k):
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.s * self.s))

    def dw(self, k):
        x = np.asarray(k, dtype=float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.s * self.s))

    def d2w(self, k):
        x = np.asarray(k, dtype=float) - self.m
        return self.b * self.s ** 2 / (x * x + self.s * self.s) ** 1.5

    def min_w(self) -> float:
        return self.a + self.b * self.s * np.sqrt(1 - self.rho ** 2)

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.s])


def g_function(k, w, dw, d2w):
    """Gatheral-Jacquier butterfly function; the slice is arbitrage-free where g >= 0."""
    k, w, dw, d2w = (np.asarray(v, dtype=float) for v in (k, w, dw, d2w))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - k * dw / (2 * w)) ** 2 - dw ** 2 / 4 * (1 / w + 0.25) + d2w / 2


def density(k, w, dw, d2w):
    """Risk-neutral density of k = ln(K/F) implied by a total-variance smile."""
    w = np.asarray(w, dtype=float)
    dm = -np.asarray(k) / np.sqrt(w) - np.sqrt(w) / 2
    return g_function(k, w, dw, d2w) * np.exp(-0.5 * dm * dm) / (SQRT2PI * np.sqrt(w))


@dataclass
class SVIFit:
    params: SVIParams
    T: float
    rmse_vol: float               # unweighted RMS of (model - mid) in vol
    within_bid_ask: float         # share of quotes whose model vol lies inside [bid, ask]
    min_g: float                  # min of g on the arbitrage-check grid
    calendar_violation: float     # max (w_prev - w) on the grid (<= 0 means none), nan if no previous slice
    n: int
    success: bool


def _inner_ls(k, w_mkt, omega, m, s, a_bound):
    """Quasi-explicit inner step (Zeliade 2009): for fixed (m, s), raw SVI is linear,
    w = a + u (R + y)/2 + v (R - y)/2 with y = (k - m)/s, R = sqrt(y^2 + 1), u = c + d, v = c - d,
    c = b s, d = rho b s. |rho| <= 1 and Lee's bound b (1 + |rho|) <= 2 become 0 <= u, v <= 2 s, a box,
    so a bounded linear least squares gives the exact optimum of (a, b, rho) for this (m, s)."""
    y = (k - m) / s
    R = np.sqrt(y * y + 1)
    A = np.column_stack([np.ones_like(k), (R + y) / 2, (R - y) / 2]) * omega[:, None]
    rhs = w_mkt * omega
    x, *_ = np.linalg.lstsq(A, rhs, rcond=None)                 # unconstrained first: often feasible
    if not (-a_bound <= x[0] <= a_bound and 0.0 <= x[1] <= 2 * s and 0.0 <= x[2] <= 2 * s):
        x = lsq_linear(A, rhs, bounds=([-a_bound, 0.0, 0.0], [a_bound, 2 * s, 2 * s]), method="bvls").x
    cost = float(np.sum((A @ x - rhs) ** 2))
    a, u, v = x
    c, d = (u + v) / 2, (u - v) / 2
    b = c / s
    rho = d / c if c > 0 else 0.0
    return cost, SVIParams(float(a), float(b), float(np.clip(rho, -0.999, 0.999)), float(m), float(s))


def quasi_explicit_fit(k, w_mkt, omega, n_grid: int = 15, m_range=None):
    """Global search over (m, s) with the exact inner solve, refined by Nelder-Mead on (m, log s).

    `m_range` = (lo, hi) restricts the vertex m. With the vertex inside the quoted range the fit cannot
    use only the left branch of the hyperbola with a vertex beyond the last strike, a shape that often
    fits best but has a steep, arbitrageable extrapolated wing.
    """
    span = max(float(k.max() - k.min()), 1e-3)
    a_bound = 2.0 * float(np.max(w_mkt)) + 1e-4
    m_lo, m_hi = m_range if m_range is not None else (k.min() - 0.5 * span, k.max() + 0.5 * span)
    best = (np.inf, None)
    for m in np.linspace(m_lo, m_hi, n_grid):
        for s in np.geomspace(0.005 * span, 2.0 * span, n_grid):
            obj, p = _inner_ls(k, w_mkt, omega, m, s, a_bound)
            if obj < best[0]:
                best = (obj, p)
    p0 = best[1]

    def f(x):
        return _inner_ls(k, w_mkt, omega, float(np.clip(x[0], m_lo, m_hi)), np.exp(x[1]), a_bound)[0]

    res = minimize(f, [p0.m, np.log(p0.s)], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-12 * max(best[0], 1e-300), "maxiter": 600})
    return _inner_ls(k, w_mkt, omega, float(np.clip(res.x[0], m_lo, m_hi)), np.exp(res.x[1]), a_bound)[1]


def default_constraint_grid(k) -> np.ndarray:
    """Dense around the quotes (range extended by max(50%, 0.1) each side) plus a sparse far grid on
    [-2, 1], so the no-arbitrage conditions are imposed well beyond the data, not just between quotes."""
    span = float(k.max() - k.min())
    margin = max(0.5 * span, 0.1)
    return np.unique(np.concatenate([np.linspace(k.min() - margin, k.max() + margin, 121), np.linspace(-2.0, 1.0, 61)]))


def fit_svi_slice(k, iv_mid, T, *, iv_bid=None, iv_ask=None, prev: SVIParams | None = None,
                  k_grid=None, calendar_grid=None, penalty: float = 1e4, max_nfev: int = 1500,
                  max_iter: int = 300) -> SVIFit:
    """Fit a raw SVI slice to mid implied vols under static no-arbitrage constraints.

    (1) Quasi-explicit global fit in total variance (Zeliade 2009), with weights that make w-errors
        approximate the vol-space objective; it imposes |rho| <= 1 and Lee's bound exactly.
    (2) Constrained polish in vol space by SLSQP: minimise sum(((model vol - mid vol) * w_i)^2) with
        w_i proportional to 1/max(bid-ask width, 0.001) (equal weights without bid/ask), subject to
        g(k_j) >= 0 on `k_grid` (butterfly; default ``default_constraint_grid``, which reaches far into
        the wings), w_min >= 0, b (1 +- rho) <= 2 (Lee) and, when `prev` (the previous, shorter slice)
        is given, w(k_j) >= w_prev(k_j) on `calendar_grid` (default: `k_grid`). Starts: stage 1, and
        stage 1 with s halved / doubled. The best feasible result wins; if no start ends feasible, a
        penalised least-squares polish (weight `penalty`) is the fallback and `success` is False.

    Why the calendar grid is separate: imposing w >= w_prev deep in extrapolated wings forces a slice
    above the previous slice's (often steep) extrapolation, ruins its fit near the quotes and cascades
    down the maturities. ``fit_svi_surface`` therefore imposes calendar order where both slices have
    quotes, and the surface diagnostics report any crossing that remains in the extrapolated wings.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv_mid, dtype=float)
    w_mkt = iv * iv * T
    if iv_bid is not None and iv_ask is not None:
        wt = 1.0 / np.maximum(np.asarray(iv_ask) - np.asarray(iv_bid), 1e-3)
    else:
        wt = np.ones_like(k)
    wt = wt / np.sqrt(np.mean(wt ** 2))
    if k_grid is None:
        k_grid = default_constraint_grid(k)
    k_grid = np.asarray(k_grid, dtype=float)
    cal_grid = np.asarray(calendar_grid if calendar_grid is not None else k_grid, dtype=float)
    w_prev = prev.w(cal_grid) if prev is not None else None
    scale_w = max(float(np.median(w_mkt)), 1e-6)

    def model_vol(x, kk):
        return np.sqrt(np.maximum(SVIParams(*x).w(kk), 1e-12) / T)

    def objective(x):
        r = (model_vol(x, k) - iv) * wt
        return 0.5 * float(r @ r)

    def c_butterfly(x):
        p = SVIParams(*x)
        return g_function(k_grid, np.maximum(p.w(k_grid), 1e-12), p.dw(k_grid), p.d2w(k_grid))

    def c_misc(x):
        p = SVIParams(*x)
        return np.array([p.min_w() / scale_w, 2.0 - p.b * (1 + p.rho), 2.0 - p.b * (1 - p.rho)])

    constraints = [{"type": "ineq", "fun": c_butterfly}, {"type": "ineq", "fun": c_misc}]
    if w_prev is not None:
        constraints.append({"type": "ineq", "fun": lambda x: (SVIParams(*x).w(cal_grid) - w_prev) / scale_w})

    def feasible(x):                                        # 1e-6 tolerance; the exact min g is reported
        ok = c_butterfly(x).min() >= -1e-6 and c_misc(x).min() >= -1e-8
        if w_prev is not None:
            ok = ok and float(np.min((SVIParams(*x).w(cal_grid) - w_prev) / scale_w)) >= -1e-6
        return ok

    def resid(x):
        p = SVIParams(*x)
        wk = np.maximum(p.w(k), 1e-12)
        r_fit = (np.sqrt(wk / T) - iv) * wt
        wg = p.w(k_grid)
        g = g_function(k_grid, np.maximum(wg, 1e-12), p.dw(k_grid), p.d2w(k_grid))
        pen = [penalty * np.maximum(-g, 0.0) / np.sqrt(k_grid.size),
               [penalty * max(-p.min_w(), 0.0) / scale_w, penalty * max(p.b * (1 + abs(p.rho)) - 2.0, 0.0)]]
        if w_prev is not None:
            pen.append(penalty * np.maximum(w_prev - p.w(cal_grid), 0.0) / scale_w / np.sqrt(cal_grid.size))
        return np.concatenate([r_fit, *[np.atleast_1d(v) for v in pen]])

    span = max(k.max() - k.min(), 1e-3)
    lo = [-2.0 * scale_w - 1.0, 0.0, -0.999, k.min() - 2 * span, 1e-5]
    hi = [2.0 * scale_w + 1.0, 2.0, 0.999, k.max() + 2 * span, 5.0 * span + 1.0]
    omega = wt / (2.0 * np.maximum(iv, 1e-4) * T)          # w-space weights ~ vol-space objective
    qe = quasi_explicit_fit(k, w_mkt, omega)
    qe_in = quasi_explicit_fit(k, w_mkt, omega, m_range=(float(k.min()), float(k.max())))
    starts = []
    for base in (qe, qe_in):
        starts += [base.as_array()] + [np.array([base.a, base.b, base.rho, base.m, base.s * fs]) for fs in (0.5, 2.0)]
    if prev is not None:
        starts.append(prev.as_array())                      # feasible for the calendar constraint
    def violation(x):
        v = max(0.0, -float(c_butterfly(x).min()), -float(c_misc(x).min()))
        if w_prev is not None:
            v = max(v, -float(np.min((SVIParams(*x).w(cal_grid) - w_prev) / scale_w)))
        return v

    # Box bounds matter: without them SLSQP can step to w < 0 or s -> 0, where the constraints are NaN,
    # and stops with a singular subproblem. (Normalising the objective made it less reliable, not more.)
    # A run that stops near-feasible (iteration limit) is restarted from where it stopped, up to twice.
    # Any result more than 25x worse than the unconstrained quasi-explicit objective is rejected: SLSQP
    # can collapse to a flat smile (b = 0, g = 1), which is feasible but meaningless.
    obj_cap = 25.0 * max(objective(qe.as_array()), 1e-14)
    candidates = []                                         # every SLSQP result: (x, objective, violation, success)
    for x0 in starts:
        x = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        for _ in range(3):
            res = minimize(objective, x, method="SLSQP", bounds=list(zip(lo, hi)),
                           constraints=constraints, options={"maxiter": max_iter, "ftol": 1e-12})
            if not np.all(np.isfinite(res.x)):
                break
            x = res.x
            val = objective(x)
            if np.isfinite(val) and val <= obj_cap:
                candidates.append((x, val, violation(x), bool(res.success)))
            if feasible(x) or violation(x) > 1e-3:
                break
    feas = [c for c in candidates if feasible(c[0])]
    near = [c for c in candidates if c[2] <= 1e-4]
    if feas:                                                # best fit among feasible results
        best_x, _, _, success = min(feas, key=lambda c: c[1])
    elif near:                                              # else best fit among nearly feasible ones
        best_x, _, _, _ = min(near, key=lambda c: c[1])
        success = False
    elif candidates:                                        # else the least violation
        best_x, _, _, _ = min(candidates, key=lambda c: c[2])
        success = False
    else:                                                   # fallback: penalised least squares from stage 1
        sol = least_squares(resid, np.clip(qe.as_array(), np.array(lo) + 1e-9, np.array(hi) - 1e-9),
                            bounds=(lo, hi), method="trf", max_nfev=max_nfev, x_scale="jac")
        best_x, success = sol.x, False
    p = SVIParams(*best_x)
    model = np.sqrt(np.maximum(p.w(k), 0) / T)
    wg = p.w(k_grid)
    g = g_function(k_grid, np.maximum(wg, 1e-12), p.dw(k_grid), p.d2w(k_grid))
    inside = np.nan
    if iv_bid is not None and iv_ask is not None:
        inside = float(np.mean((model >= np.asarray(iv_bid) - 1e-12) & (model <= np.asarray(iv_ask) + 1e-12)))
    return SVIFit(params=p, T=T, rmse_vol=float(np.sqrt(np.mean((model - iv) ** 2))), within_bid_ask=inside,
                  min_g=float(np.min(g)),
                  calendar_violation=float(np.max(w_prev - p.w(cal_grid))) if w_prev is not None else np.nan,
                  n=int(k.size), success=success)


class SVISurface:
    """Fitted raw-SVI slices interpolated linearly in total variance at fixed k (see module docstring)."""

    def __init__(self, Ts, params: list[SVIParams]):
        order = np.argsort(Ts)
        self.Ts = np.asarray(Ts, dtype=float)[order]
        self.params = [params[i] for i in order]

    def _bracket(self, T):
        T = float(T)
        if T <= self.Ts[0]:
            return 0, 0, T / self.Ts[0], "pre"
        if T >= self.Ts[-1]:
            n = len(self.Ts) - 1
            return n, n, T / self.Ts[-1], "post"
        j = int(np.searchsorted(self.Ts, T))
        i = j - 1
        lam = (T - self.Ts[i]) / (self.Ts[j] - self.Ts[i])
        return i, j, lam, "in"

    def _combine(self, attr, k, T):
        i, j, lam, kind = self._bracket(T)
        if kind in ("pre", "post"):
            return lam * getattr(self.params[i], attr)(k)
        return (1 - lam) * getattr(self.params[i], attr)(k) + lam * getattr(self.params[j], attr)(k)

    def w(self, k, T):
        return self._combine("w", k, T)

    def dw_dk(self, k, T):
        return self._combine("dw", k, T)

    def d2w_dk2(self, k, T):
        return self._combine("d2w", k, T)

    def dw_dT(self, k, T):
        i, j, lam, kind = self._bracket(T)
        if kind in ("pre", "post"):
            return self.params[i].w(k) / self.Ts[i]
        return (self.params[j].w(k) - self.params[i].w(k)) / (self.Ts[j] - self.Ts[i])

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)


def fit_svi_surface(slices, *, calendar: bool = True, **kw):
    """Fit raw SVI to each slice in order of maturity (butterfly constraints on each slice's wide grid).

    With `calendar`, a slice whose butterfly-only fit crosses below the previous slice on the k range
    quoted in both (61 points) is refitted with w >= w_prev imposed there. The calendar-constrained fit
    is kept only if it is feasible and costs little: RMSE <= max(1.5 x the butterfly-only RMSE,
    that RMSE + half the slice's median bid-ask width in vol). Otherwise the butterfly-only fit is kept
    and its calendar crossing stays in `calendar_violation`. This stops one hard slice from forcing every
    later slice above it. Returns (SVISurface, list[SVIFit]).
    """
    fits, prev, prev_k = [], None, None
    for s in sorted(slices, key=lambda x: x.T):
        f = fit_svi_slice(s.k, s.iv_mid, s.T, iv_bid=s.iv_bid, iv_ask=s.iv_ask, **kw)
        if calendar and prev is not None:
            lo, hi = max(s.k.min(), prev_k.min()), min(s.k.max(), prev_k.max())
            if hi > lo:
                cal_grid = np.linspace(lo, hi, 61)
                crossing = float(np.max(prev.w(cal_grid) - f.params.w(cal_grid)))
                f.calendar_violation = crossing
                if crossing > 1e-10:
                    fc = fit_svi_slice(s.k, s.iv_mid, s.T, iv_bid=s.iv_bid, iv_ask=s.iv_ask, prev=prev,
                                       calendar_grid=cal_grid, **kw)
                    budget = max(1.5 * f.rmse_vol, f.rmse_vol + 0.5 * float(np.median(s.iv_spread)))
                    if fc.success and fc.calendar_violation <= 1e-8 and fc.rmse_vol <= budget:
                        f = fc
        fits.append(f)
        prev, prev_k = f.params, s.k
    return SVISurface([f.T for f in fits], [f.params for f in fits]), fits


# ------------------------------------------------------------------------------------------------
# SSVI
# ------------------------------------------------------------------------------------------------
def phi_power(theta, eta, gamma):
    theta = np.asarray(theta, dtype=float)
    return eta * theta ** (-gamma) * (1 + theta) ** (gamma - 1)


def dphi_power(theta, eta, gamma):
    theta = np.asarray(theta, dtype=float)
    return phi_power(theta, eta, gamma) * (-gamma / theta + (gamma - 1) / (1 + theta))


def ssvi_w(k, theta, rho, phi):
    k = np.asarray(k, dtype=float)
    p = phi * k + rho
    return theta / 2 * (1 + rho * phi * k + np.sqrt(p * p + 1 - rho * rho))


def ssvi_dw_dk(k, theta, rho, phi):
    p = phi * np.asarray(k, dtype=float) + rho
    R = np.sqrt(p * p + 1 - rho * rho)
    return theta * phi / 2 * (rho + p / R)


def ssvi_d2w_dk2(k, theta, rho, phi):
    p = phi * np.asarray(k, dtype=float) + rho
    R = np.sqrt(p * p + 1 - rho * rho)
    return theta * phi ** 2 / 2 * (1 - rho * rho) / R ** 3


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float

    def phi(self, theta):
        return phi_power(theta, self.eta, self.gamma)


class SSVISurface:
    """Power-law SSVI with theta(T) piecewise linear through the fitted ATM total variances."""

    def __init__(self, Ts, thetas, params: SSVIParams):
        order = np.argsort(Ts)
        self.Ts = np.asarray(Ts, dtype=float)[order]
        self.thetas = np.asarray(thetas, dtype=float)[order]
        self.p = params

    def theta(self, T):
        T = float(T)
        if T <= self.Ts[0]:
            return self.thetas[0] * T / self.Ts[0]
        if T >= self.Ts[-1]:
            return self.thetas[-1] * T / self.Ts[-1]
        return float(np.interp(T, self.Ts, self.thetas))

    def dtheta_dT(self, T):
        T = float(T)
        if T <= self.Ts[0]:
            return self.thetas[0] / self.Ts[0]
        if T >= self.Ts[-1]:
            return self.thetas[-1] / self.Ts[-1]
        j = int(np.searchsorted(self.Ts, T, side="right"))
        return (self.thetas[j] - self.thetas[j - 1]) / (self.Ts[j] - self.Ts[j - 1])

    def w(self, k, T):
        th = self.theta(T)
        return ssvi_w(k, th, self.p.rho, self.p.phi(th))

    def dw_dk(self, k, T):
        th = self.theta(T)
        return ssvi_dw_dk(k, th, self.p.rho, self.p.phi(th))

    def d2w_dk2(self, k, T):
        th = self.theta(T)
        return ssvi_d2w_dk2(k, th, self.p.rho, self.p.phi(th))

    def dw_dtheta(self, k, theta):
        k = np.asarray(k, dtype=float)
        ph = self.p.phi(theta)
        p = ph * k + self.p.rho
        R = np.sqrt(p * p + 1 - self.p.rho ** 2)
        w = theta / 2 * (1 + self.p.rho * ph * k + R)
        return w / theta + theta * k / 2 * (self.p.rho + p / R) * dphi_power(theta, self.p.eta, self.p.gamma)

    def dw_dT(self, k, T):
        return self.dw_dtheta(k, self.theta(T)) * self.dtheta_dT(T)

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)


# ------------------------------------------------------------------------------------------------
# Extended SSVI (eSSVI): one (theta, rho, psi) per slice
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ESSVISlice:
    """eSSVI slice (Hendriks & Martini 2019): SSVI with its own correlation per maturity.

    w(k) = theta/2 (1 + rho psi k/theta + sqrt((psi k/theta + rho)^2 + 1 - rho^2)), with psi = theta phi.
    theta = w(0) is the ATM total variance. The asymptotic wing slopes are psi (1 +- rho) / 2.
    Gatheral-Jacquier's sufficient no-butterfly conditions read psi (1 + |rho|) <= 4 and
    psi^2 (1 + |rho|) <= 4 theta.
    """
    theta: float
    rho: float
    psi: float

    def w(self, k):
        return ssvi_w(k, self.theta, self.rho, self.psi / self.theta)

    def dw(self, k):
        return ssvi_dw_dk(k, self.theta, self.rho, self.psi / self.theta)

    def d2w(self, k):
        return ssvi_d2w_dk2(k, self.theta, self.rho, self.psi / self.theta)

    def butterfly_margin(self) -> float:
        """min(4 - psi (1 + |rho|), 4 - psi^2 (1 + |rho|) / theta): >= 0 is sufficient for g >= 0."""
        a = 1 + abs(self.rho)
        return float(min(4 - self.psi * a, 4 - self.psi ** 2 * a / self.theta))


@dataclass
class ESSVIFit:
    slice: ESSVISlice
    T: float
    rmse_vol: float
    within_bid_ask: float
    min_g: float
    calendar_min: float            # min over the calendar grid of (w - w_prev); nan for the first slice
    success: bool


def fit_essvi_surface(slices, *, k_cal=None, max_iter: int = 300):
    """Fit eSSVI slice by slice in order of maturity, free of static arbitrage by construction.

    Per slice, SLSQP minimises the bid-ask-weighted squared vol errors over (theta, rho, psi) subject to
      butterfly:  psi (1 +- rho) <= 4 and psi^2 (1 +- rho) <= 4 theta (sufficient, closed form);
      calendar:   theta >= theta_prev, psi (1 +- rho) >= psi_prev (1 +- rho_prev) (ATM level and both
                  asymptotic wing slopes non-decreasing), and w(k) >= w_prev(k) on `k_cal`
                  (default [-3, 2]), which also covers the intermediate region.
    Returns (SVISurface over the eSSVI slices, list[ESSVIFit]). The surface interpolates total variance
    linearly in T, which preserves the calendar order of the slices.
    """
    k_cal = np.linspace(-3.0, 2.0, 251) if k_cal is None else np.asarray(k_cal, dtype=float)
    fits, prev = [], None
    for s in sorted(slices, key=lambda x: x.T):
        k, iv, T = s.k, s.iv_mid, s.T
        wt = 1.0 / np.maximum(s.iv_spread, 1e-3)
        wt = wt / np.sqrt(np.mean(wt ** 2))
        w_mkt = iv * iv * T
        scale_w = max(float(np.median(w_mkt)), 1e-6)
        w_prev = prev.w(k_cal) if prev is not None else None
        th0 = max(atm_total_variance(s), (prev.theta if prev else 0.0) * (1 + 1e-6), 1e-8)
        # The optimiser works on (theta / th0, rho, psi): theta is a total variance (1e-4 for a few-day
        # slice) while rho and psi are O(0.1-1); unscaled, SLSQP's finite-difference gradients and
        # quasi-Newton updates stall short of the optimum.
        def unscale(x):
            return np.array([x[0] * th0, x[1], x[2]])

        def objective(x):
            r = (np.sqrt(np.maximum(ESSVISlice(*unscale(x)).w(k), 1e-12) / T) - iv) * wt
            return 0.5 * float(r @ r)

        def c_butterfly(x):
            th, rho, psi = unscale(x)
            return np.array([4 - psi * (1 + rho), 4 - psi * (1 - rho), (4 * th - psi ** 2 * (1 + rho)) / th0,
                             (4 * th - psi ** 2 * (1 - rho)) / th0])

        cons = [{"type": "ineq", "fun": c_butterfly}]
        if prev is not None:
            pr = prev

            def c_order(x, pr=pr):
                th, rho, psi = unscale(x)
                return np.array([(th - pr.theta) / scale_w, psi * (1 + rho) - pr.psi * (1 + pr.rho),
                                 psi * (1 - rho) - pr.psi * (1 - pr.rho)])

            cons += [{"type": "ineq", "fun": c_order},
                     {"type": "ineq", "fun": lambda x, wp=w_prev: (ESSVISlice(*unscale(x)).w(k_cal) - wp) / scale_w}]

        def feasible(x):
            ok = c_butterfly(x).min() >= -1e-10
            if prev is not None:
                ok = ok and c_order(x).min() >= -1e-10 and float(np.min((ESSVISlice(*unscale(x)).w(k_cal) - w_prev) / scale_w)) >= -1e-8
            return ok

        order = np.argsort(k)
        near = np.abs(k[order]) <= max(0.25 * (k.max() - k.min()), 1e-3)
        slope = np.polyfit(k[order][near], w_mkt[order][near], 1)[0] if near.sum() >= 3 else -0.5 * th0
        starts = []
        for rho0 in ([prev.rho] if prev else []) + [-0.9, -0.7, -0.5]:
            psi0 = abs(slope / rho0) if rho0 != 0 else 0.1
            lo_psi = 0.0 if prev is None else max(prev.psi * (1 + prev.rho) / (1 + rho0), prev.psi * (1 - prev.rho) / (1 - rho0))
            psi0 = float(np.clip(max(psi0, lo_psi * 1.01), 1e-4, 0.95 * min(4 / (1 + abs(rho0)), np.sqrt(4 * th0 / (1 + abs(rho0))))))
            starts.append([1.0, rho0, psi0])                 # theta in units of th0
        th_lo = max(prev.theta, 1e-10) if prev else 1e-10
        bounds = [(th_lo / th0, (4.0 * float(w_mkt.max()) + 1e-4) / th0), (-0.999, 0.999), (1e-6, 4.0)]
        best, best_val, success = None, np.inf, False
        for x0 in starts:
            x = np.array(x0, dtype=float)
            for _ in range(3):
                res = minimize(objective, x, method="SLSQP", bounds=bounds, constraints=cons,
                               options={"maxiter": max_iter, "ftol": 1e-12})
                if not np.all(np.isfinite(res.x)):
                    break
                x = res.x
                if feasible(x):
                    break
            if np.all(np.isfinite(x)) and feasible(x) and objective(x) < best_val:
                best, best_val, success = x, objective(x), bool(res.success)
        if best is None:                      # no feasible start: keep the previous slice's shape, scaled up
            base = prev if prev is not None else ESSVISlice(th0, -0.7, min(0.5, np.sqrt(th0)))
            best, success = np.array([max(th0, base.theta) / th0, base.rho, base.psi]), False
        sl = ESSVISlice(*unscale(best))
        model = np.sqrt(np.maximum(sl.w(k), 0) / T)
        kg = default_constraint_grid(k)
        g = g_function(kg, sl.w(kg), sl.dw(kg), sl.d2w(kg))
        fits.append(ESSVIFit(sl, T, float(np.sqrt(np.mean((model - iv) ** 2))),
                             float(np.mean((model >= s.iv_bid) & (model <= s.iv_ask))), float(g.min()),
                             float(np.min(sl.w(k_cal) - w_prev)) if w_prev is not None else np.nan, success))
        prev = sl
    return SVISurface([f.T for f in fits], [f.slice for f in fits]), fits


@dataclass
class SSVIFit:
    surface: SSVISurface
    rmse_vol: float
    within_bid_ask: float
    per_slice_rmse: np.ndarray
    success: bool


def atm_total_variance(s) -> float:
    """ATM-forward total variance of a slice by linear interpolation of mid total variance at k = 0."""
    k, w = s.k, s.w_mid
    order = np.argsort(k)
    return float(np.interp(0.0, k[order], w[order]))


def fit_ssvi(slices, *, max_nfev: int = 3000) -> SSVIFit:
    """Fit power-law SSVI to all slices jointly.

    Free parameters: rho, zeta = eta (1 + |rho|)/2 in (0, 1], gamma in [0.01, 0.5], and one ATM total
    variance per slice parameterised as cumulative positive increments, so theta is strictly increasing
    (no calendar arbitrage by construction). Residuals: (model - mid vol) / max(bid-ask, 0.001), with
    every slice weighted equally (each slice's residuals scaled by 1/sqrt(n_slice)).
    """
    sl = sorted(slices, key=lambda x: x.T)
    Ts = np.array([s.T for s in sl])
    th0 = np.maximum.accumulate(np.array([max(atm_total_variance(s), 1e-6) for s in sl]))
    inc0 = np.diff(np.concatenate([[0.0], th0]))
    inc0 = np.maximum(inc0, 1e-7)
    data = []
    for s in sl:
        wt = 1.0 / np.maximum(s.iv_spread, 1e-3)
        wt = wt / np.sqrt(np.mean(wt ** 2)) / np.sqrt(s.k.size)
        data.append((s.k, s.iv_mid, s.T, wt))

    def unpack(x):
        rho, zeta, gamma = x[:3]
        eta = 2 * zeta / (1 + abs(rho))
        thetas = np.cumsum(np.exp(x[3:]))
        return SSVIParams(rho, eta, gamma), thetas

    def resid(x):
        p, thetas = unpack(x)
        out = []
        for (k, iv, T, wt), th in zip(data, thetas):
            w = ssvi_w(k, th, p.rho, p.phi(th))
            out.append((np.sqrt(np.maximum(w, 1e-12) / T) - iv) * wt)
        return np.concatenate(out)

    x0 = np.concatenate([[-0.7, 0.5, 0.4], np.log(inc0)])
    lo = np.concatenate([[-0.999, 1e-4, 0.01], np.full(len(sl), -25.0)])
    hi = np.concatenate([[0.999, 1.0, 0.5], np.full(len(sl), 2.0)])
    best = None
    for rho0, zeta0, g0 in ((-0.7, 0.5, 0.4), (-0.9, 0.9, 0.25), (-0.5, 0.2, 0.5)):
        x0[:3] = (rho0, zeta0, g0)
        sol = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), method="trf",
                            max_nfev=max_nfev, x_scale="jac")
        if best is None or sol.cost < best.cost:
            best = sol
    p, thetas = unpack(best.x)
    surf = SSVISurface(Ts, thetas, p)
    errs, inside, per = [], [], []
    for s, th in zip(sl, thetas):
        model = np.sqrt(np.maximum(ssvi_w(s.k, th, p.rho, p.phi(th)), 0) / s.T)
        e = model - s.iv_mid
        errs.append(e)
        per.append(np.sqrt(np.mean(e ** 2)))
        inside.append((model >= s.iv_bid) & (model <= s.iv_ask))
    errs = np.concatenate(errs)
    return SSVIFit(surface=surf, rmse_vol=float(np.sqrt(np.mean(errs ** 2))),
                   within_bid_ask=float(np.mean(np.concatenate(inside))), per_slice_rmse=np.array(per),
                   success=bool(best.success))
