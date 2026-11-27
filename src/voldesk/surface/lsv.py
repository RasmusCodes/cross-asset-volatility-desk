"""Local-stochastic volatility (LSV): Heston-type variance times a leverage function, calibrated by the particle method.

Model. In the forward-normalised log price X_t = ln(S_t / F_t) (deterministic rates and dividends)
    dX_t = -1/2 L(t, X_t)^2 V_t dt + L(t, X_t) sqrt(V_t) dW_t,
    dV_t = kappa (theta - V_t) dt + sigma sqrt(V_t) dB_t,          d<W, B>_t = rho dt,
with (v0, kappa, theta, sigma, rho) taken from a Heston fit. By Gyongy's (1986) mimicking theorem the model
reproduces every vanilla of a Dupire local-volatility surface sigma_D exactly when
    L(t, x)^2 E[V_t | X_t = x] = sigma_D(t, x)^2.
The conditional expectation depends on L itself, so the equation is solved forward in time.

Particle method (Guyon & Henry-Labordere 2012). Simulate N particles (X, V); at each time step estimate
E[V_t | X_t = x] by kernel regression across the particles, set L(t, x) = sigma_D(t, x) / sqrt(E[V_t | X_t = x])
and move the particles one step with that L. Details:
  * kernel: quartic, K(u) = (1 - u^2)^2 on |u| <= 1 (its normalisation cancels in the ratio); bandwidth
    h_t = kappa_h sigma_ATM(t) sqrt(max(t, t_min)) N^(-1/5), with the paper's kappa_h = 1.5 and t_min = 0.25 as
    defaults. The floor t_min sets the bandwidth for the first months; for smiles only a few weeks long it
    over-smooths (scripts/lsv_study.py uses t_min = 0.05; its --ablation run compares the settings on the Heston
    test described below);
  * local-linear regression by default instead of the paper's Nadaraya-Watson: it removes Nadaraya-Watson's
    design bias h^2 m' f'/f, which is largest in the tails of the particle cloud (`local_linear=False` restores the
    paper's estimator);
  * evaluated by linear binning (Fan & Marron 1994) on a grid of spacing h / bins_per_h and discrete convolutions,
    so a step costs O(N) rather than O(N x grid points);
  * where the kernel weight is thin (fewer than `min_weight` particle-equivalents) the leverage is held flat at the
    nearest well-estimated value (linear interpolation, flat extrapolation, as in the paper);
  * the leverage is stored on each step's grid. Prices come from a second, independent simulation with the leverage
    frozen, so the pricing paths are i.i.d. and their standard errors are exact.

Discretisation, shared by LSV, pure local vol and pure Heston so that their differences are model differences:
  * variance: full-truncation Euler (Lord, Koekkoek & van Dijk 2010), V+ = max(V, 0) in drift and diffusion;
  * log price: X_{n+1} = X_n - 1/2 L^2 V+ dt + L sqrt(V+ dt) Z. The step is conditionally lognormal, so
    E[exp(X_{n+1}) | F_n] = exp(X_n) holds exactly: the simulated forward is a martingale with no drift correction;
  * the local variance is taken at the mid-step time (the convention of voldesk.surface.localvol.forward_pde).
Pure local vol is the same scheme with V = 1 and L = sigma_D; pure Heston has L = 1. With sigma = 0 the variance is
deterministic, the regression is exact and LSV collapses to local vol (tested).

Validation target. `heston_local_variance` gives the Dupire local variance of a Heston model by Fourier-COS
expansions. For a Heston "market" and the same Heston variance process the leverage must come out as L = 1, which
tests the regression with a known answer (tests/test_lsv_rough.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pricing.bsm import black_price
from ..pricing.implied_vol import implied_vol_black
from .heston import HestonParams


# ------------------------------------------------------------------------------------------------ grids, regression
def time_grid(T_max: float, nodes=(), dt=1 / 365) -> np.ndarray:
    """Times 0 = t_0 < ... < t_M = T_max with every node in (0, T_max] on the grid. `dt` is the largest step, or a
    function dt(t) giving the largest step at time t (e.g. finer steps near t = 0, where short maturities live)."""
    step = dt if callable(dt) else (lambda t, d=float(dt): d)
    pts = np.unique(np.concatenate([[0.0, float(T_max)],
                                    [float(t) for t in np.atleast_1d(nodes) if 0.0 < float(t) <= T_max]]))
    out = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        t = a
        while t < b - 1e-12:
            m = max(int(np.ceil((b - t) / float(step(t)) - 1e-9)), 1)
            nxt = t + (b - t) / m
            out.append(nxt if m > 1 else b)
            t = out[-1]
    return np.array(out)


def graded_dt(dt_short: float = 1 / 1460, dt_long: float = 1 / 365, t_short: float = 0.25):
    """Step size dt_short up to t_short, rising linearly to dt_long at 4 t_short and dt_long after."""
    def f(t):
        if t <= t_short:
            return dt_short
        return min(dt_long, dt_short + (dt_long - dt_short) * (t - t_short) / (3 * t_short))
    return f


def kernel_regression(x: np.ndarray, y: np.ndarray, h: float, bins_per_h: int = 10, local_linear: bool = True):
    """Kernel estimate of E[y | x] with the quartic kernel K(u) = (1 - u^2)^2, by linear binning.

    local_linear=False: Nadaraya-Watson, m(g) = sum K y / sum K (Guyon & Henry-Labordere's choice).
    local_linear=True: local linear regression (Fan 1992), the intercept of a kernel-weighted line fitted around g;
        it removes the design-bias term h^2 m'(g) f'(g)/f(g) of Nadaraya-Watson, which is large in the tails of the
        particle distribution where f'/f is large. Falls back to Nadaraya-Watson where the local design is
        degenerate.
    Returns (grid, m, weight) on a regular grid covering [min x - h, max x + h]; weight(g) = sum_i K((x_i - g)/h),
    the number of particle-equivalents behind m(g).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    delta = h / bins_per_h
    lo = float(x.min()) - h
    nb = int(np.ceil((float(x.max()) + h - lo) / delta)) + 2
    p = (x - lo) / delta
    j = np.floor(p).astype(np.int64)
    f = p - j
    c = np.bincount(j, 1.0 - f, nb) + np.bincount(j + 1, f, nb)
    s = np.bincount(j, (1.0 - f) * y, nb) + np.bincount(j + 1, f * y, nb)
    u = np.arange(-bins_per_h, bins_per_h + 1) / bins_per_h
    ker = (1.0 - u * u) ** 2
    S0 = np.convolve(c, ker, mode="same")
    T0 = np.convolve(s, ker, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(S0 > 0, T0 / S0, np.nan)
        if local_linear:
            # np.convolve flips the kernel: sum_b c_b K(u_{g-b}) u_{g-b}, with u_{g-b} = (g - b)/h. The moments
            # need (b - g)/h = -u, hence the minus signs on the odd kernels.
            S1 = -np.convolve(c, ker * u, mode="same")
            S2 = np.convolve(c, ker * u * u, mode="same")
            T1 = -np.convolve(s, ker * u, mode="same")
            det = S0 * S2 - S1 * S1
            ll = (S2 * T0 - S1 * T1) / det
            good = (S0 > 0) & (det > 1e-6 * S0 * np.maximum(S2, 1e-300)) & np.isfinite(ll)
            m = np.where(good, ll, m)
    return lo + delta * np.arange(nb), m, S0


# ------------------------------------------------------------------------------------------------ containers
@dataclass
class Leverage:
    """Leverage function on the simulation grid: step n (from times[n] to times[n + 1]) uses
    L = np.interp(x, grids[n], values[n]) (linear inside, flat outside)."""
    times: np.ndarray
    grids: list
    values: list

    def at(self, n: int, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.grids[n], self.values[n])


@dataclass
class LSVCalibration:
    heston: HestonParams
    leverage: Leverage
    n_particles: int
    dt: object                                         # largest step, or a function of t (see time_grid)
    diagnostics: dict = field(default_factory=dict)   # per step arrays: t, h, share_supported, L quantiles


# ------------------------------------------------------------------------------------------------ simulation
def _normals(rng, n_paths: int):
    return rng.standard_normal(n_paths), rng.standard_normal(n_paths)


def _step(X, V, L, dt, p: HestonParams | None, z1, z2):
    """One step of the shared scheme (see module docstring). V is None for pure local vol (V = 1)."""
    if p is None:
        return X - 0.5 * L * L * dt + L * np.sqrt(dt) * z1, None
    Vp = np.maximum(V, 0.0)
    sv = np.sqrt(Vp * dt)
    Xn = X - 0.5 * L * L * Vp * dt + L * sv * z1
    Vn = V + p.kappa * (p.theta - Vp) * dt + p.sigma * sv * (p.rho * z1 + np.sqrt(1.0 - p.rho ** 2) * z2)
    return Xn, Vn


def _check_obs(times, T_obs):
    T_obs = np.atleast_1d(np.asarray(T_obs, float))
    idx = {}
    for T in T_obs:
        j = int(np.argmin(np.abs(times - T)))
        if abs(times[j] - T) > 1e-12:
            raise ValueError(f"observation time {T} is not on the simulation grid (pass it in `nodes`)")
        idx[j] = float(T)
    return idx


def calibrate_lsv(local_var, heston: HestonParams, T_max: float, *, nodes=(), n_particles: int = 100_000,
                  dt=1 / 365, seed: int = 0, kappa_h: float = 1.5, t_min: float = 0.25, atm_vol=None,
                  bins_per_h: int = 10, min_weight: float = 10.0, local_linear: bool = True,
                  record=()) -> LSVCalibration:
    """Particle-method calibration of the leverage function up to T_max.

    local_var(x, t): Dupire local variance at log forward moneyness x (array) and time t (scalar), e.g.
        ``lambda x, t: voldesk.surface.localvol.local_variance(essvi_surface, x, t)``.
    heston: parameters of the variance process (v0, kappa, theta, sigma, rho).
    nodes: times that must lie on the simulation grid (the maturities to be priced later).
    atm_vol(t): ATM implied vol used in the bandwidth (default sqrt(local_var(0, t))).
    record: grid times at which the calibration particles' X are kept in diagnostics["X"] (in-sample check only).
    """
    times = time_grid(T_max, nodes, dt)
    rec = _check_obs(times, record) if len(np.atleast_1d(record)) else {}
    rng = np.random.default_rng(seed)
    N = int(n_particles)
    X = np.zeros(N)
    V = np.full(N, float(heston.v0))
    nfac = N ** -0.2
    grids, values = [], []
    diag = {k: [] for k in ("t", "h", "share_supported", "L_p01", "L_p50", "L_p99", "EV_min")}
    kept = {}
    for n in range(times.size - 1):
        t, dtn = times[n], times[n + 1] - times[n]
        tm = t + 0.5 * dtn
        Vp = np.maximum(V, 0.0)
        sig = float(atm_vol(tm)) if atm_vol is not None else float(np.sqrt(local_var(np.zeros(1), tm)[0]))
        h = kappa_h * sig * np.sqrt(max(t, t_min)) * nfac
        g, ev, wgt = kernel_regression(X, Vp, h, bins_per_h, local_linear)
        ok = (wgt >= min_weight) & np.isfinite(ev) & (ev > 1e-12)
        if not ok.any():                                  # cannot happen with N >= min_weight; kept as a guard
            ok = np.isfinite(ev) & (ev > 1e-12)
        g, ev = g[ok], ev[ok]
        lev = np.sqrt(np.maximum(np.asarray(local_var(g, tm), float), 0.0) / ev)
        grids.append(g)
        values.append(lev)
        L = np.interp(X, g, lev)
        inside = (X >= g[0]) & (X <= g[-1])
        q = np.quantile(L, [0.01, 0.5, 0.99])
        for key, val in zip(diag, (t, h, float(inside.mean()), q[0], q[1], q[2], float(ev.min()))):
            diag[key].append(val)
        z1, z2 = _normals(rng, N)
        X, V = _step(X, V, L, dtn, heston, z1, z2)
        if (n + 1) in rec:
            kept[rec[n + 1]] = X.copy()
    out = {k: np.array(v) for k, v in diag.items()}
    if kept:
        out["X"] = kept
    return LSVCalibration(heston, Leverage(times, grids, values), N, dt, out)


def _simulate(times, T_obs, n_paths, seed, p, lev_fn, batch: int = 250_000):
    """Shared path generator: returns {T: X_T} for every T in T_obs (all on `times`). lev_fn(n, t_mid, X) -> L."""
    obs = _check_obs(times, T_obs)
    rng = np.random.default_rng(seed)
    out = {T: [] for T in obs.values()}
    left = int(n_paths)
    while left > 0:
        N = min(batch, left)
        left -= N
        X = np.zeros(N)
        V = np.full(N, float(p.v0)) if p is not None else None
        for n in range(times.size - 1):
            dtn = times[n + 1] - times[n]
            L = lev_fn(n, times[n] + 0.5 * dtn, X)
            z1, z2 = _normals(rng, N) if p is not None else (rng.standard_normal(N), None)
            X, V = _step(X, V, L, dtn, p, z1, z2)
            if (n + 1) in obs:
                out[obs[n + 1]].append(X.copy())
    return {T: np.concatenate(v) for T, v in out.items()}


def simulate_lsv(cal: LSVCalibration, T_obs, *, n_paths: int = 200_000, seed: int = 1) -> dict:
    """Independent pricing pass with the calibrated leverage frozen. T_obs must be calibration nodes."""
    lev = cal.leverage
    return _simulate(lev.times, T_obs, n_paths, seed, cal.heston, lambda n, tm, X: lev.at(n, X))


def simulate_local_vol(local_var, T_obs, *, n_paths: int = 200_000, dt=1 / 365, seed: int = 1,
                       T_max: float | None = None, nodes=()) -> dict:
    """Pure Dupire local vol by the same scheme (V = 1, L = sigma_D)."""
    T_obs = np.atleast_1d(np.asarray(T_obs, float))
    times = time_grid(T_max if T_max is not None else float(T_obs.max()), np.concatenate([T_obs, np.atleast_1d(nodes)]), dt)

    def lev(n, tm, X):
        return np.sqrt(np.maximum(np.asarray(local_var(X, tm), float), 0.0))
    return _simulate(times, T_obs, n_paths, seed, None, lev)


def simulate_heston(heston: HestonParams, T_obs, *, n_paths: int = 200_000, dt=1 / 365, seed: int = 1,
                    T_max: float | None = None, nodes=()) -> dict:
    """Pure Heston by the same scheme (L = 1)."""
    T_obs = np.atleast_1d(np.asarray(T_obs, float))
    times = time_grid(T_max if T_max is not None else float(T_obs.max()), np.concatenate([T_obs, np.atleast_1d(nodes)]), dt)
    return _simulate(times, T_obs, n_paths, seed, heston, lambda n, tm, X: 1.0)


# ------------------------------------------------------------------------------------------------ pricing helpers
def otm_prices(XT: np.ndarray, k, *, control: bool = True):
    """Monte Carlo prices of out-of-the-money options on exp(X_T) (forward 1, undiscounted, strike e^k: calls for
    k >= 0, puts for k < 0) with standard errors, for i.i.d. samples XT. With `control`, the forward exp(X_T) (mean
    exactly 1 in the scheme) is a control variate with the optimal coefficient per strike. All strikes at once from
    prefix sums over the sorted sample: O(N log N + strikes log N)."""
    S = np.sort(np.exp(np.asarray(XT, float)))
    N = S.size
    k = np.atleast_1d(np.asarray(k, float))
    K = np.exp(k)
    zero = np.zeros(1)
    c1 = np.concatenate([zero, np.cumsum(S)])            # sums of S over the i smallest
    c2 = np.concatenate([zero, np.cumsum(S * S)])
    i = np.searchsorted(S, K, side="right")              # count of S <= K
    call = k >= 0
    # sums over the region where the option pays: S > K (calls) or S <= K (puts)
    n_in = np.where(call, N - i, i)
    s1 = np.where(call, c1[-1] - c1[i], c1[i])
    s2 = np.where(call, c2[-1] - c2[i], c2[i])
    sgn = np.where(call, 1.0, -1.0)
    m1 = sgn * (s1 - K * n_in) / N                       # E[payoff]
    m2 = (s2 - 2 * K * s1 + K * K * n_in) / N            # E[payoff^2]
    var_pay = np.maximum(m2 - m1 * m1, 0.0)
    if not control:
        return m1, np.sqrt(var_pay / N)
    mS, vS = c1[-1] / N, max(c2[-1] / N - (c1[-1] / N) ** 2, 1e-300)
    cov = sgn * (s2 - K * s1) / N - m1 * mS             # Cov(payoff, S)
    b = cov / vS
    price = m1 - b * (mS - 1.0)
    var = np.maximum(var_pay - cov * cov / vS, 0.0)
    return price, np.sqrt(var / N)


def otm_iv(price, k, T, se=None):
    """Black implied vols (forward 1, undiscounted) of OTM prices at log-strikes k, and the standard errors in vol
    (price standard error / Black vega) when `se` is given."""
    k = np.atleast_1d(np.asarray(k, float))
    K = np.exp(k)
    cp = np.where(k >= 0, 1.0, -1.0)
    iv = implied_vol_black(price, cp, 1.0, K, T, 1.0)
    if se is None:
        return iv
    s = iv * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = -k / s - 0.5 * s
        vega = K * np.exp(-0.5 * d2 * d2) / np.sqrt(2 * np.pi) * np.sqrt(T)
        return iv, np.asarray(se, float) / vega


def black_otm(k, T, iv):
    """Black OTM prices (forward 1, undiscounted) at log-strikes k for implied vols iv."""
    k = np.atleast_1d(np.asarray(k, float))
    return black_price(np.where(k >= 0, 1.0, -1.0), 1.0, np.exp(k), T, iv, 1.0)


# ------------------------------------------------------------------------------------------------ Heston local vol
def _heston_cf_and_dT(u, T, p: HestonParams):
    """Characteristic function of X_T = ln(S_T/F_T) (voldesk.surface.heston.heston_cf) and its T-derivative from the
    Riccati equations D' = -(u^2 + iu)/2 - (kappa - rho sigma iu) D + sigma^2 D^2 / 2, C' = kappa theta D."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    a = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(a * a + p.sigma ** 2 * (iu + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = p.kappa * p.theta / p.sigma ** 2 * ((a - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (a - d) / p.sigma ** 2 * (1 - e) / (1 - g * e)
    phi = np.exp(C + D * p.v0)
    dD = -0.5 * (u * u + iu) - a * D + 0.5 * p.sigma ** 2 * D * D
    return phi, phi * (p.kappa * p.theta * D + dD * p.v0)


def heston_local_variance(p: HestonParams, x, T: float, *, n: int = 2048, L: float = 12.0,
                          rel_density_floor: float = 1e-9):
    """Dupire local variance of a Heston model at log forward moneyness x (array), maturity T:
        sigma_loc^2(x, T) = 2 dc/dT (x, T) / (e^x p_T(x)),   c(x, T) = E[(e^{X_T} - e^x)^+],
    (the forward equation dc/dT = 1/2 sigma_loc^2 (c_xx - c_x) with c_xx - c_x = e^x p_T(x)), which for Heston equals
    E[V_T | X_T = x] (Gyongy). The density p_T and dc/dT come from COS expansions (Fang & Oosterlee 2008) of the
    characteristic function and of its analytic T-derivative on the truncation range of heston_price_cos. Where the
    density is below `rel_density_floor` times its peak the ratio is numerically meaningless and the value is held
    flat at the nearest reliable point."""
    x = np.atleast_1d(np.asarray(x, float))
    T = max(float(T), 1e-8)
    ekt = np.exp(-p.kappa * T)
    mean_var = p.theta * T + (p.v0 - p.theta) * (1 - ekt) / p.kappa
    sd = np.sqrt(max(mean_var, 1e-12)) * (1.0 + p.sigma)
    a, b = -0.5 * mean_var - L * sd, -0.5 * mean_var + L * sd
    j = np.arange(n)
    u = j * np.pi / (b - a)
    phi, dphi = _heston_cf_and_dT(u, T, p)
    shift = np.exp(-1j * u * a)
    Fk = 2.0 / (b - a) * np.real(phi * shift)
    Gk = 2.0 / (b - a) * np.real(dphi * shift)
    Fk[0] *= 0.5
    Gk[0] *= 0.5
    xc = np.clip(x, a, b)
    arg = np.outer(xc - a, u)
    dens = np.cos(arg) @ Fk
    ex = np.exp(xc)[:, None]
    uu = u[None, :]
    chi = (np.cos(u * (b - a)) * np.exp(b) - np.cos(arg) * ex + uu * np.sin(u * (b - a)) * np.exp(b)
           - uu * np.sin(arg) * ex) / (1.0 + uu * uu)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(uu == 0, b - xc[:, None], (np.sin(u * (b - a)) - np.sin(arg)) / np.where(uu == 0, 1.0, uu))
    dcdT = (chi - ex * psi) @ Gk
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = 2.0 * dcdT / (np.exp(xc) * dens)
    good = np.isfinite(lv) & (dens > rel_density_floor * max(float(dens.max()), 1e-300)) & (lv > 0)
    if not good.any():
        return np.full(x.shape, np.nan)
    if not good.all():
        order = np.argsort(xc)
        xs, ls, gs = xc[order], lv[order], good[order]
        filled = np.interp(xs, xs[gs], ls[gs])
        lv = np.empty_like(filled)
        lv[order] = filled
    return lv
