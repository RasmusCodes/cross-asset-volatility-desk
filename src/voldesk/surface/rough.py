"""Rough volatility (advanced extension): the term structure of the at-the-money skew, and a rough Bergomi pricer.

ATM skew. psi(T) = |d sigma_BS / dk| at k = 0. Rough volatility models with Hurst exponent H imply
psi(T) ~ c T^(H - 1/2) for short maturities (Alos, Leon & Vives 2007; Fukasawa 2011; Bayer, Friz & Gatheral 2016),
while one-factor Markovian models such as Heston give a skew that flattens to a constant as T -> 0. Regressing
log psi on log T across a date's maturities therefore gives an implied H. From a raw SVI slice,
    sigma(k) = sqrt(w(k)/T),  d sigma/dk at 0 = w'(0) / (2 sqrt(w(0) T)).

Rough Bergomi (Bayer, Friz & Gatheral 2016), forward variance curve xi0(t):
    v_t = xi0(t) exp(eta sqrt(2H) int_0^t (t-s)^(H-1/2) dW_s - eta^2 t^(2H)/2),  dS/S = sqrt(v) dZ,  d<W,Z> = rho dt.
xi0(t) = E[v_t] is the forward variance curve: int_0^T xi0 is the variance-swap total variance to T. The process is
simulated exactly on a time grid by a Cholesky factorization of the joint covariance of the Volterra process and a
Brownian motion (the method of the original paper; O(n^3) once per H, fine for a few hundred steps).

`rbergomi_prices` is the original flat-xi0 pricer (simulates Z, prices payoffs). `RBergomiMC` is the calibration
engine: a forward-variance curve, the conditional ("mixing") estimator and common random numbers.
  * Mixing estimator (Romano & Touzi 1997; used for rough Bergomi by McCrickerd & Pakkanen 2018): write
    Z = rho W + sqrt(1 - rho^2) W_perp. Given the path of W, log(S_T/F_T) is Gaussian with mean rho I - Q/2 and
    variance (1 - rho^2) Q, where I = int sqrt(v) dW and Q = int v dt. An option price is therefore the average over
    paths of Black prices with forward exp(rho I - rho^2 Q/2) and total variance (1 - rho^2) Q: W_perp is integrated
    out exactly, which removes most of the Monte Carlo noise. On the grid I and Q are left-point sums; v_{t_i} is
    independent of the next increment of W, so exp(rho I - rho^2 Q/2) has mean exactly 1 (the forward is matched).
  * Common random numbers: the standard normals are drawn once, so the estimated smile is a smooth deterministic
    function of (H, eta, rho) and a least-squares optimiser with finite-difference Jacobians works. Only H enters
    the Cholesky factor; eta and rho act on the stored paths.
  * Forward variance: `variance_swap_total_variance` integrates an arbitrage-free smile's OTM prices (the log-
    contract replication); `forward_variance_curve` turns total variances at the slice maturities into a
    piecewise-constant xi0(t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import hyp2f1

from ..pricing.bsm import black_price
from ..pricing.implied_vol import implied_vol_black


def svi_atm_skew(a, b, rho, m, s, T) -> float:
    w0 = a + b * (rho * (-m) + np.sqrt(m * m + s * s))
    dw0 = b * (rho + (-m) / np.sqrt(m * m + s * s))
    return float(abs(dw0) / (2.0 * np.sqrt(max(w0, 1e-12) * T)))


def power_law_exponent(T, psi, T_max: float = 0.5, T_min: float = 7 / 365) -> dict:
    T, psi = np.asarray(T, float), np.asarray(psi, float)
    ok = (T >= T_min) & (T <= T_max) & np.isfinite(psi) & (psi > 0)
    if ok.sum() < 3:
        return {"slope": np.nan, "H": np.nan, "n": int(ok.sum()), "r2": np.nan}
    x, y = np.log(T[ok]), np.log(psi[ok])
    A = np.column_stack([np.ones(ok.sum()), x])
    beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2) if ok.sum() > 2 else np.nan
    return {"slope": float(beta[1]), "H": float(beta[1] + 0.5), "n": int(ok.sum()), "r2": float(r2)}


def rbergomi_cov(t: np.ndarray, H: float, rho: float) -> np.ndarray:
    """Covariance of (V_t1..V_tn, Z_t1..Z_tn) with V_t = sqrt(2H) int_0^t (t-s)^(H-1/2) dW_s and Z a Brownian motion
    with d<W, Z> = rho dt (rho = 1: Z = W). For u = min(s, t) > 0 and x = u / max(s, t) (Bayer, Friz & Gatheral 2016):
        Cov(V_s, V_t) = 2H u^(2H) x^(1/2-H) / (H+1/2) 2F1(1/2-H, 1; H+3/2; x),   Var(V_t) = t^(2H),
        Cov(V_ti, Z_tj) = rho sqrt(2H)/(H+1/2) (t_i^(H+1/2) - (t_i - min(t_i, t_j))^(H+1/2)),   Cov(Z_s, Z_t) = u."""
    t = np.asarray(t, float)
    g = H - 0.5
    n = t.size
    ti, tj = np.meshgrid(t, t, indexing="ij")
    u, v = np.minimum(ti, tj), np.maximum(ti, tj)
    x = np.where(v > 0, u / np.where(v > 0, v, 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cvv = np.where(u > 0, 2 * H * u ** (2 * H) * x ** (-g) / (1 + g) * hyp2f1(-g, 1.0, 2 + g, x), 0.0)
    np.fill_diagonal(cvv, t ** (2 * H))
    cvz = rho * np.sqrt(2 * H) / (H + 0.5) * (ti ** (H + 0.5) - (ti - u) ** (H + 0.5))
    C = np.empty((2 * n, 2 * n))
    C[:n, :n] = cvv
    C[n:, n:] = u
    C[:n, n:] = cvz
    C[n:, :n] = cvz.T
    return C


def rbergomi_prices(T_list, k_list, H, eta, rho, xi0, n_paths=20000, steps_per_year=252, seed=0):
    """European OTM prices (forward 1, undiscounted) and Black implied vols at log-moneyness k for each maturity."""
    T_max = max(T_list)
    n = max(int(np.ceil(T_max * steps_per_year)), 8)
    t = np.linspace(T_max / n, T_max, n)
    C = rbergomi_cov(t, H, rho)
    L = np.linalg.cholesky(C + 1e-12 * np.eye(2 * n))
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_paths, 2 * n)) @ L.T
    V, Z = G[:, :n], G[:, n:]
    v = xi0 * np.exp(eta * V - 0.5 * eta ** 2 * t ** (2 * H))
    dZ = np.diff(np.concatenate([np.zeros((n_paths, 1)), Z], axis=1), axis=1)
    dt = np.diff(np.concatenate([[0.0], t]))
    v_prev = np.concatenate([np.full((n_paths, 1), xi0), v[:, :-1]], axis=1)
    logS = np.cumsum(np.sqrt(v_prev) * dZ - 0.5 * v_prev * dt, axis=1)
    out = {}
    for T in T_list:
        j = int(np.argmin(np.abs(t - T)))
        ST = np.exp(logS[:, j])
        ST = ST / ST.mean()                                  # martingale correction of the discretisation drift
        Ks = np.exp(np.asarray(k_list, float))
        cp = np.where(Ks >= 1.0, 1.0, -1.0)
        px = np.array([np.mean(np.maximum(c * (ST - K), 0.0)) for c, K in zip(cp, Ks)])
        out[float(t[j])] = implied_vol_black(px, cp, 1.0, Ks, float(t[j]), 1.0)
    return out


# ------------------------------------------------------------------------------------------------ forward variance
def variance_swap_total_variance(wfun, *, n: int = 4001, n_sd: float = 14.0) -> float:
    """Variance-swap total variance of one maturity from its total-implied-variance smile w(k), k = ln(K/F):
        E[-2 ln(S_T/F_T)] = 2 int OTM(K)/K^2 dK = 2 int OTM(e^k) e^(-k) dk     (forward 1, undiscounted),
    by Simpson's rule on the put side [-span, 0] and the call side [0, span] separately (the integrand has a kink at
    k = 0), span = n_sd sqrt(w(0)) + 0.5. Needs an arbitrage-free smile with wing slopes below 2 (e.g. eSSVI), so
    that the integrand decays in both wings."""
    from scipy.integrate import simpson
    w0 = float(wfun(np.zeros(1))[0])
    span = n_sd * np.sqrt(max(w0, 1e-10)) + 0.5
    total = 0.0
    for k, cp in ((np.linspace(-span, 0.0, n), -1.0), (np.linspace(0.0, span, n), 1.0)):
        w = np.maximum(np.asarray(wfun(k), float), 1e-14)
        otm = black_price(cp, 1.0, np.exp(k), 1.0, np.sqrt(w), 1.0)
        total += simpson(otm * np.exp(-k), x=k)
    return float(2.0 * total)


def forward_variance_curve(Ts, w_total):
    """Piecewise-constant forward variance xi0(t) from total variances w_total at maturities Ts:
    xi0 = (w_i - w_{i-1}) / (T_i - T_{i-1}) on [T_{i-1}, T_i) with (T_0, w_0) = (0, 0), the last value held after T_n.
    Increments are floored at 0 (they are non-negative on a calendar-arbitrage-free surface). Returns a vectorised
    function of t; left-closed pieces make xi0 at a grid time t_i the value used on [t_i, t_{i+1})."""
    Ts = np.asarray(Ts, float)
    o = np.argsort(Ts)
    Ts, w = Ts[o], np.asarray(w_total, float)[o]
    knots = np.concatenate([[0.0], Ts])
    xi = np.maximum(np.diff(np.concatenate([[0.0], w])), 0.0) / np.diff(knots)

    def xi0(t):
        t = np.asarray(t, float)
        i = np.clip(np.searchsorted(Ts, t, side="right"), 0, Ts.size - 1)
        return xi[i]
    xi0.knots, xi0.values = Ts, xi
    return xi0


# ------------------------------------------------------------------------------------------------ calibration engine
class RBergomiMC:
    """Rough Bergomi smiles with a forward variance curve, the mixing estimator and common random numbers.

    T_nodes: maturities that must be on the grid (the slices). The grid has at most 1/steps_per_year between nodes.
    xi0: forward variance function of t (``forward_variance_curve``). n_paths: number of normal vectors drawn once
    (doubled by antithetic sign flips when `antithetic`).
    refine: simulate on a grid `refine` times finer; ``coarse=True`` in `integrals` / `prices` / `smile` then gives
    the steps_per_year scheme on the same Brownian paths (every refine-th node), so the two resolutions can be
    compared without Monte Carlo noise between them.

    Time step. I and Q are left-point sums, so the correlation between a step's Brownian increment and the variance
    later in the same step is missed. For small H and strong correlation the short-dated smile converges slowly as
    the grid is refined (the paired checks in scripts/rough_study.py measure it), and what matters for a maturity T is
    the step size relative to T. The grid is therefore uniform between consecutive nodes with at most
    1/steps_per_year per step and, with `min_steps`, at most T / min_steps in the interval ending at node T: a
    three-week smile gets as many steps as a four-month one.
    """

    def __init__(self, T_nodes, xi0, *, n_paths: int = 2 ** 15, steps_per_year: int = 1460, min_steps: int = 0,
                 seed: int = 0, antithetic: bool = True, refine: int = 1):
        T_nodes = np.sort(np.unique(np.asarray(T_nodes, float)))
        pts = np.concatenate([[0.0], T_nodes])
        t = []
        self.refine = int(refine)
        for a, b in zip(pts[:-1], pts[1:]):
            m = max(int(np.ceil((b - a) * steps_per_year - 1e-9)), int(np.ceil(min_steps * (b - a) / b - 1e-9)), 1)
            m *= self.refine
            t.extend(np.linspace(a, b, m + 1)[1:])
        self.t = np.array(t)
        self.dt = np.diff(np.concatenate([[0.0], self.t]))
        self.node_index = {float(T): int(np.argmin(np.abs(self.t - T))) for T in T_nodes}
        self.xi_left = np.asarray(xi0(np.concatenate([[0.0], self.t[:-1]])), float)   # xi0 at left points t_0..t_{n-1}
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, 2 * self.t.size))
        self.antithetic = bool(antithetic)
        self.Z = np.concatenate([Z, -Z]) if antithetic else Z
        self._cache: dict = {}

    def paths(self, H: float):
        """(V, dW): the Volterra process at t_1..t_n and the increments of its Brownian driver, for every path."""
        key = round(float(H), 12)
        if key not in self._cache:
            n = self.t.size
            L = np.linalg.cholesky(rbergomi_cov(self.t, H, 1.0) + 1e-13 * np.eye(2 * n))
            G = self.Z @ L.T
            W = G[:, n:]
            dW = np.diff(np.concatenate([np.zeros((W.shape[0], 1)), W], axis=1), axis=1)
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (G[:, :n], dW)
        return self._cache[key]

    def integrals(self, H: float, eta: float, coarse: bool = False):
        """I = int sqrt(v) dW and Q = int v dt at every grid time (left-point sums), arrays (paths x nodes).
        coarse: on every refine-th node only (see the class docstring)."""
        V, dW = self.paths(H)
        n = self.t.size
        v = np.empty_like(V)
        v[:, 0] = self.xi_left[0]
        v[:, 1:] = self.xi_left[1:] * np.exp(eta * V[:, :n - 1] - 0.5 * eta * eta * self.t[:n - 1] ** (2 * H))
        dt = self.dt
        if coarse and self.refine > 1:
            r = self.refine
            v = v[:, ::r]
            dW = dW.reshape(dW.shape[0], n // r, r).sum(axis=2)
            dt = dt.reshape(n // r, r).sum(axis=1)
        I = np.cumsum(np.sqrt(v) * dW, axis=1)
        Q = np.cumsum(v * dt, axis=1)
        return I, Q

    def _column(self, T: float, coarse: bool) -> int:
        j = self.node_index[float(T)]
        return (j + 1) // self.refine - 1 if (coarse and self.refine > 1) else j

    def prices(self, H: float, eta: float, rho: float, k_by_T: dict, batches: int = 1, coarse: bool = False):
        """OTM prices (forward 1, undiscounted) at log-strikes k_by_T[T] for each node T. With batches > 1 also the
        standard errors from batch means (paths split into `batches` groups, antithetic pairs kept together)."""
        I, Q = self.integrals(H, eta, coarse)
        out = {}
        half = self.Z.shape[0] // 2 if self.antithetic else 0
        for T, k in k_by_T.items():
            j = self._column(T, coarse)
            k = np.atleast_1d(np.asarray(k, float))
            F = np.exp(rho * I[:, j] - 0.5 * rho * rho * Q[:, j])
            sd = np.sqrt(np.maximum((1.0 - rho * rho) * Q[:, j], 0.0))
            cp = np.where(k >= 0, 1.0, -1.0)
            px = black_price(cp[None, :], F[:, None], np.exp(k)[None, :], 1.0, sd[:, None], 1.0)
            if batches > 1:
                # pair path i with its antithetic twin i + half, then average pairs within each batch
                pairs = 0.5 * (px[:half] + px[half:]) if self.antithetic else px
                bm = np.array([b.mean(axis=0) for b in np.array_split(pairs, batches)])
                out[T] = (bm.mean(axis=0), bm.std(axis=0, ddof=1) / np.sqrt(batches))
            else:
                out[T] = px.mean(axis=0)
        return out

    def smile(self, H: float, eta: float, rho: float, k_by_T: dict, coarse: bool = False) -> dict:
        """Black implied vols at log-strikes k_by_T[T]."""
        pr = self.prices(H, eta, rho, k_by_T, coarse=coarse)
        return {T: implied_vol_black(pr[T], np.where(np.asarray(k) >= 0, 1.0, -1.0), 1.0, np.exp(k), T, 1.0)
                for T, k in k_by_T.items()}


@dataclass
class RBergomiFit:
    H: float
    eta: float
    rho: float
    cost: float
    nfev: int
    success: bool
    starts: list


def calibrate_rbergomi(slices, xi0, *, n_paths: int = 2 ** 14, steps_per_year: int = 1460, min_steps: int = 0,
                       seed: int = 0, n_grid: int = 41, starts=None,
                       bounds=((0.005, 0.05, -0.999), (0.5, 8.0, 0.5)),
                       n_polish: int = 3, max_nfev: int = 200) -> tuple[RBergomiFit, RBergomiMC]:
    """Least-squares calibration of (H, eta, rho) to slices' mid vols with xi0 given.

    Residuals are those of voldesk.surface.heston.fit_heston: (model - mid vol) / max(bid-ask width, 0.001),
    normalised so every slice weighs the same. Model vols are computed on `n_grid` strikes spanning each slice's
    quoted range and interpolated to the quotes by a cubic spline in k. Common random numbers make the objective
    deterministic; a coarse grid of starts is evaluated first and the best `n_polish` are polished by
    scipy.optimize.least_squares (trust region reflective, bounds on all three parameters).
    """
    sl = sorted(slices, key=lambda s: s.T)
    eng = RBergomiMC([s.T for s in sl], xi0, n_paths=n_paths, steps_per_year=steps_per_year, min_steps=min_steps,
                     seed=seed)
    grids = {s.T: np.linspace(s.k.min(), s.k.max(), n_grid) for s in sl}
    data = []
    for s in sl:
        wt = 1.0 / np.maximum(s.iv_spread, 1e-3)
        data.append((s, wt / np.sqrt(np.mean(wt ** 2)) / np.sqrt(s.K.size)))

    def resid(x):
        H, eta, rho = x
        iv = eng.smile(H, eta, rho, grids)
        out = []
        for s, wt in data:
            g = grids[s.T]
            ok = np.isfinite(iv[s.T])
            if ok.sum() < 4:
                out.append(np.full(s.K.size, 1.0))
                continue
            m = CubicSpline(g[ok], iv[s.T][ok])(s.k)
            out.append(np.where(np.isfinite(m), (m - s.iv_mid) * wt, 1.0))
        return np.concatenate(out)

    if starts is None:
        starts = [(H, e, r) for H in (0.03, 0.08, 0.15, 0.3) for e in (1.0, 2.0, 3.0) for r in (-0.95, -0.8, -0.6)]
    scored = sorted(((float(np.sum(resid(np.array(x0)) ** 2)), tuple(x0)) for x0 in starts), key=lambda c: c[0])
    lo, hi = np.array(bounds[0], float), np.array(bounds[1], float)
    best = None
    for _, x0 in scored[:n_polish]:
        sol = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf",
                            x_scale=np.array([0.05, 0.5, 0.1]), diff_step=1e-3, max_nfev=max_nfev)
        if best is None or sol.cost < best.cost:
            best = sol
    H, eta, rho = best.x
    return RBergomiFit(float(H), float(eta), float(rho), float(best.cost), int(best.nfev), bool(best.success),
                       [s for _, s in scored[:n_polish]]), eng
