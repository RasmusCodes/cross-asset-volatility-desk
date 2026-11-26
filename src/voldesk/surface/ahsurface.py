"""Arbitrage-free volatility interpolation (Andreasen & Huge 2011, "Volatility interpolation", Risk).

The surface is a set of prices on a strike grid, not an implied-vol formula. With x = K / F_T and
c = C / (D_T F_T) (undiscounted, forward-normalised calls), Dupire's forward equation is drift-free:
    dc/dT = 1/2 sigma(x, T)^2 x^2 c_xx,      c(x, 0) = (1 - x)^+
(in y = ln x: dc/dT = 1/2 sigma^2 (c_yy - c_y)). Between consecutive quoted maturities T_{i-1} < T_i the
local vol depends on strike only, sigma_i(y), and the prices at T_i come from those at T_{i-1} by n_i equal
implicit Euler steps
    (I - dt Theta D) c^{n+1} = c^n,      Theta = diag(sigma_i(x)^2 x^2 / 2),
on a fixed grid x_0 < ... < x_{N-1} (uniform in y, or sinh-stretched in y so that it is fine near the
money) with a node at the forward, x = 1. D is the three-point second difference on that grid; the
boundaries are Dirichlet, c(x_0) = 1 - x_0 and c(x_{N-1}) = 0. sigma_i is piecewise linear in log sigma
between knots at quantiles of the slice's quoted y, flat beyond them, and is calibrated to the slice's
quotes (bid-ask-weighted vol errors plus a small smoothness penalty).

Stored prices are out of the money: o = c - (1 - x)^+, the put below the forward and the call above it,
zero at both ends of the grid. The step for o is the same scheme; the payoff's kink becomes a source
dt theta_k / w_k at the forward node k. Away from the money every stored price is small, so deep
in-the-money calls (c close to 1) never carry absolute round-off into the recomputed butterflies.

The grid prices define a distribution. With slopes s_j = (o_{j+1} - o_j) / (x_{j+1} - x_j), the masses
    m_0 = s_0,   m_j = s_j - s_{j-1}  (0 < j < N-1),   m_{N-1} = -s_{N-2},   plus 1 at the forward node
sum to one, have mean one (the forward) and reproduce the grid prices as E[(X - x_i)^+] for calls and
E[(x_i - X)^+] for puts. When they are >= 0 the prices are free of static arbitrage at every strike, once
off-grid prices are defined by the same distribution: linear in K between grid strikes, puts worth 0
below the grid and calls worth 0 above it (``iv`` and the calibration price quotes this way).

Why the masses stay >= 0 and prices rise with T, exactly:
  * interior masses obey (I - dt L diag(theta / w)) m^{n+1} = m^n, with w_j = (x_{j+1} - x_{j-1}) / 2 and L
    the slope-difference operator with zero boundary values. L has positive off-diagonals and column sums
    <= 0 (zero away from the boundary), so the matrix is a Z-matrix with column sums >= 1, a non-singular
    M-matrix with a non-negative inverse: m^n >= 0 gives m^{n+1} >= 0 (no butterfly arbitrage). The
    boundary masses only grow; they hold what leaves the grid. The payoff is a unit mass at the forward;
  * c^{n+1} - c^n = dt Theta m^{n+1} / w >= 0 (no calendar arbitrage);
  * between calibrated maturities the prices are k full steps plus one partial step r in [0, dt_i) of the
    calibrated step size dt_i, so c(T) is continuous and, as dc/dr = (I - r Theta D)^-1 Theta m / w >= 0
    (I - r Theta D is a Z-matrix with unit row sums), non-decreasing in T: the interpolation in time is
    arbitrage-free too. (Re-dividing the interval into a T-dependent number of equal steps would not be:
    the step count jumps with T and the prices with it, not always upwards.)
The central (D2 - D1) discretisation of c_yy - c_y agrees with D to O(h^2), but only D's positivity is the
absence of butterflies at the grid strikes, so D is used. ``check_arbitrage`` recomputes masses and
calendar increments from the prices rather than trusting the argument.

Time steps: the interval ending at T_i is split into ceil(dt / max(dt_min, T_i / steps_per_T)) equal implicit
steps (1.5 hours for short maturities, T/800 for long ones by default). Andreasen-Huge use one step per
interval. The scheme is arbitrage-free for any number of steps; more steps make sigma_i closer to the
continuous-time local vol that reprices the surface, which
``voldesk.surface.localvol.forward_pde_local_var`` measures (the gap falls at first order in the step).

Grid: by default y in [-5, 5] (strikes from 0.7% to 148 times the forward), 2001 nodes, sinh-stretched
with beta = 5 (spacing 3.4e-4 in y at the money, 0.025 at the edges). Probability that the forward
equation carries past the edges stays on the edge nodes as an atom; ``check_arbitrage`` reports it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lapack
from scipy.optimize import least_squares

from ..pricing.implied_vol import implied_vol_black

EPS = float(np.finfo(float).eps)
Y_MAX, N_Y, BETA = 5.0, 2001, 5.0              # default grid
DT_MIN, STEPS_PER_T = 1 / 5840, 800.0          # default time steps: 1.5 hours or T/800, whichever is longer


@dataclass
class AHSlice:
    label: str
    T: float
    knots: np.ndarray            # y positions of the local-vol knots
    log_sigma: np.ndarray        # log local vol at the knots (flat beyond the outermost knots)
    n_steps: int                 # equal implicit steps from the previous maturity
    rmse_vol: float
    within_bid_ask: float
    n: int
    success: bool

    def sigma(self, y):
        return np.exp(np.interp(np.asarray(y, dtype=float), self.knots, self.log_sigma))


@dataclass
class AHSurface:
    y: np.ndarray                # grid in log-moneyness (strikes x = exp(y)), with a node at y = 0
    Ts: np.ndarray               # calibrated maturities
    slices: list                 # AHSlice per maturity
    otm: list                    # OTM prices o(x, T_i) on the grid, one array per maturity
    dt_min: float
    steps_per_T: float
    meta: dict = field(default_factory=dict)

    @property
    def x(self) -> np.ndarray:
        return np.exp(self.y)

    # ------------------------------------------------------------------ evaluation
    def _interval(self, T: float):
        """(start time, start OTM prices, slice whose local vol applies, step size) for the interval holding T."""
        n = len(self.Ts)
        i = int(np.searchsorted(self.Ts, T))                 # first calibrated maturity >= T
        if i >= n:                                           # beyond the last: keep its local vol and step
            i = n - 1
            t0, o0 = float(self.Ts[i]), self.otm[i]
        else:
            t0 = float(self.Ts[i - 1]) if i >= 1 else 0.0
            o0 = self.otm[i - 1] if i >= 1 else np.zeros_like(self.y)
        start = float(self.Ts[i - 1]) if i >= 1 else 0.0
        return t0, o0, self.slices[i], (float(self.Ts[i]) - start) / self.slices[i].n_steps

    def otm_at(self, T: float) -> np.ndarray:
        """OTM prices on the grid for any T > 0: the calibrated prices at calibrated maturities, otherwise full
        steps of the interval's step size plus one partial step (continuous and non-decreasing in T)."""
        T = float(T)
        if T <= 0:
            raise ValueError("T must be > 0")
        j = np.flatnonzero(np.abs(self.Ts - T) <= 1e-14 * max(1.0, T))
        if j.size:
            return self.otm[int(j[0])]
        t0, o, sl, h = self._interval(T)
        sig = sl.sigma(self.y)
        k = int(np.floor((T - t0) / h + 1e-9))
        r = (T - t0) - k * h
        if k:
            o = step(self.y, o, sig, k * h, k)
        if r > 1e-12 * h:
            o = step(self.y, o, sig, r, 1)
        return o

    def prices_at(self, T: float) -> np.ndarray:
        """Normalised call prices C / (D F) on the grid."""
        return self.otm_at(T) + payoff(self.y)

    def call_prices(self, k, T: float) -> np.ndarray:
        """Normalised call prices C / (D F) at log-moneyness k, priced by the grid distribution."""
        return interp_otm(self.y, self.otm_at(T), k) + payoff(k)

    def iv(self, k, T: float) -> np.ndarray:
        """Black implied vols of the surface at log-moneyness k (inverted from the OTM option)."""
        k = np.asarray(k, dtype=float)
        return otm_iv(interp_otm(self.y, self.otm_at(T), k), k, float(T))

    def local_vol(self, y, T: float) -> np.ndarray:
        """Calibrated local vol: piecewise constant in T on (T_{i-1}, T_i], flat after the last maturity."""
        i = min(int(np.searchsorted(self.Ts, float(T))), len(self.Ts) - 1)
        return self.slices[i].sigma(y)

    def density(self, T: float) -> np.ndarray:
        """Risk-neutral probabilities of the grid strikes x = exp(y) at maturity T."""
        return masses(self.otm_at(T), self.y)

    def check_times(self, per_interval: int = 3, beyond: float = 1.25) -> np.ndarray:
        """Calibrated maturities, per_interval interior points of every interval (0, T_1], ..., and beyond * T_last."""
        edges = np.concatenate([[0.0], self.Ts])
        f = np.arange(1, per_interval + 1) / (per_interval + 1)
        inner = (edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * f[None, :]).ravel()
        return np.unique(np.concatenate([self.Ts, inner, [beyond * self.Ts[-1]]]))

    def check_arbitrage(self, Ts=None) -> dict:
        """Recompute grid masses (butterflies), boundary masses, the mean and calendar increments from the prices.

        Default times: ``check_times()``. min_mass and min_calendar_increment are >= 0 in exact arithmetic."""
        Ts = self.check_times() if Ts is None else np.unique(np.asarray(Ts, dtype=float))
        x = self.x
        prev = np.zeros_like(self.y)
        min_m, max_b, mean_err, min_cal = np.inf, 0.0, 0.0, np.inf
        for T in Ts:
            o = self.otm_at(T)
            m = masses(o, self.y)
            min_m = min(min_m, float(m.min()))
            max_b = max(max_b, float(m[0]), float(m[-1]))
            mean_err = max(mean_err, abs(float(m @ x) - 1.0))
            min_cal = min(min_cal, float(np.min(o - prev)))
            prev = o
        return {"min_mass": min_m, "min_calendar_increment": min_cal, "max_boundary_mass": max_b,
                "max_mean_error": mean_err, "n_maturities": int(Ts.size)}

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        """The calibrated local vols and settings: enough to rebuild the surface exactly (``from_dict``)."""
        return {"meta": self.meta, "dt_min": self.dt_min, "steps_per_T": self.steps_per_T,
                "slices": [{"label": s.label, "T": s.T, "knots": s.knots.tolist(), "log_sigma": s.log_sigma.tolist(),
                            "n_steps": s.n_steps, "rmse_vol": s.rmse_vol, "within_bid_ask": s.within_bid_ask, "n": s.n,
                            "success": s.success} for s in self.slices]}

    @classmethod
    def from_dict(cls, d: dict) -> "AHSurface":
        """Rebuild prices by marching the stored local vols (no calibration); bit-identical to the fit."""
        meta = dict(d["meta"])
        y = grid(meta["y_max"], meta["n_y"], meta["beta"])
        slices = [AHSlice(s["label"], float(s["T"]), np.array(s["knots"], dtype=float), np.array(s["log_sigma"], dtype=float),
                          int(s["n_steps"]), s["rmse_vol"], s["within_bid_ask"], int(s["n"]), bool(s["success"]))
                  for s in d["slices"]]
        o, t, otm = np.zeros_like(y), 0.0, []
        for s in slices:
            o = step(y, o, s.sigma(y), s.T - t, s.n_steps)
            otm.append(o)
            t = s.T
        return cls(y=y, Ts=np.array([s.T for s in slices]), slices=slices, otm=otm, dt_min=float(d["dt_min"]),
                   steps_per_T=float(d["steps_per_T"]), meta=meta)


# ------------------------------------------------------------------ grid and numerics
def grid(y_max: float = Y_MAX, n: int = N_Y, beta: float = BETA) -> np.ndarray:
    """Log-moneyness grid on [-y_max, y_max]: uniform for beta = 0, else y = y_max sinh(beta u) / sinh(beta) with
    u uniform, which is finer near the money than the uniform grid by sinh(beta) / beta and coarser at the
    edges by about beta. n must be odd: the middle node is the forward, y = 0."""
    n = int(n)
    if n % 2 == 0:
        raise ValueError("n must be odd so that the forward is a grid node")
    u = np.linspace(-1.0, 1.0, n)
    y = y_max * u if beta <= 0 else y_max * np.sinh(beta * u) / np.sinh(beta)
    y[n // 2] = 0.0
    return y


def forward_node(y) -> int:
    k = int(np.argmin(np.abs(y)))
    if y[k] != 0.0 or not 0 < k < len(y) - 1:
        raise ValueError("the grid needs an interior node at exactly y = 0 (the forward); use grid()")
    return k


def payoff(y):
    """Normalised call payoff (1 - x)^+ at log-moneyness y (call price = OTM price + payoff)."""
    return np.maximum(1.0 - np.exp(np.asarray(y, dtype=float)), 0.0)


def masses(o, y) -> np.ndarray:
    """Probabilities of the grid nodes (boundary nodes included) from OTM prices: slope differences, plus the
    payoff's unit mass at the forward node."""
    s = np.diff(o) / np.diff(np.exp(y))
    m = np.diff(np.concatenate([[0.0], s, [0.0]]))
    m[forward_node(y)] += 1.0
    return m


def grid_masses(c, y) -> np.ndarray:
    """The same probabilities from call prices (slope -1 below the grid, 0 above it)."""
    s = np.diff(c) / np.diff(np.exp(y))
    return np.diff(np.concatenate([[-1.0], s, [0.0]]))


def interp_otm(y, o, k) -> np.ndarray:
    """OTM prices (puts for k < 0, calls for k >= 0) at log-moneyness k under the grid distribution: linear in K
    between grid strikes, 0 beyond the grid."""
    x, xq = np.exp(y), np.exp(np.asarray(k, dtype=float))
    return np.where((xq < x[0]) | (xq > x[-1]), 0.0, np.interp(xq, x, o))


def interp_prices(y, c, k) -> np.ndarray:
    """Call prices at log-moneyness k from call prices on the grid (same distribution, linear in K)."""
    x, xq = np.exp(y), np.exp(np.asarray(k, dtype=float))
    out = np.interp(xq, x, c)
    return np.where(xq < x[0], 1.0 - xq, np.where(xq > x[-1], 0.0, out))


def otm_iv(o, k, T: float) -> np.ndarray:
    """Black implied vol from OTM prices o at log-moneyness k (puts for k < 0, calls for k >= 0)."""
    k = np.asarray(k, dtype=float)
    return implied_vol_black(o, np.where(k < 0, -1.0, 1.0), 1.0, np.exp(k), float(T), 1.0)


def n_steps_for(dt: float, T: float, dt_min: float, steps_per_T: float) -> int:
    return max(1, int(np.ceil(dt / max(dt_min, T / steps_per_T) - 1e-9)))


def _factor(x, sigma, dt):
    """LU factors of the interior rows of I - dt Theta D, and dt theta_j / w_j (the kink source weights)."""
    dxm, dxp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    th = dt * 0.5 * (sigma[1:-1] * x[1:-1]) ** 2 / (0.5 * (dxm + dxp))
    lo, up = th / dxm, th / dxp                     # dt theta_j D_{j,j-1}, dt theta_j D_{j,j+1}, j = 1..n-2
    dl, d, du, du2, ipiv, info = lapack.dgttrf(-lo[1:], 1.0 + lo + up, -up[:-1])
    if info != 0:
        raise np.linalg.LinAlgError(f"dgttrf failed (info={info})")
    return (dl, d, du, du2, ipiv), th


def step(y, o, sigma, dt, n_steps=1):
    """n_steps equal implicit Euler steps, of total length dt, of the forward equation for OTM prices
    o = c - (1 - x)^+ on grid exp(y) (o = 0 at both ends; the forward must be a node).

    The same scheme as (I - dt Theta D) c^{n+1} = c^n for calls: with c = o + (1 - x)^+ it reads
    (I - dt Theta D) o^{n+1} = o^n + dt theta_k / w_k e_k, the payoff's unit mass at the forward node k."""
    o = np.array(o, dtype=float)
    if dt <= 0:
        return o
    y = np.asarray(y, dtype=float)
    x = np.exp(y)
    k = forward_node(y)
    f, th = _factor(x, np.broadcast_to(np.asarray(sigma, dtype=float), x.shape), dt / n_steps)
    src = th[k - 1]
    o[0], o[-1] = 0.0, 0.0
    for _ in range(int(n_steps)):
        rhs = o[1:-1].copy()
        rhs[k - 1] += src
        o[1:-1], info = lapack.dgttrs(*f, rhs)
        if info != 0:
            raise np.linalg.LinAlgError(f"dgttrs failed (info={info})")
    return o


def _knots(k, n_max):
    ks = np.unique(np.sort(k))
    m = int(np.clip(ks.size // 15, 4, n_max))
    return np.unique(np.quantile(ks, np.linspace(0, 1, m)))


def fit_ah_surface(slices, *, y_max: float = Y_MAX, n_y: int = N_Y, beta: float = BETA, dt_min: float = DT_MIN,
                   steps_per_T: float = STEPS_PER_T, max_knots: int = 16, lam_smooth: float = 0.02, max_nfev: int = 200):
    """Calibrate the Andreasen-Huge surface slice by slice in order of maturity.

    Knots: max(4, min(max_knots, n_quotes // 15)) at quantiles of the slice's quoted y. Residuals: (model vol
    - mid vol) / max(bid-ask width, 0.001) (weights normalised to unit RMS) plus lam_smooth * second
    differences of log sigma across the knots. Model prices at the quoted strikes come from the grid
    distribution (``interp_otm``). Quotes whose model price cannot be inverted get a residual of 1 (in
    weight units). Defaults (16 knots, lam_smooth 0.02, steps of 1.5 hours or T/800, grid [-5, 5]) were chosen
    on the 12 Nov 2026 SPX close; reports/04_vol_surfaces.md shows the alternatives. Returns
    (AHSurface, list[AHSlice]).
    """
    y = grid(y_max, n_y, beta)
    o_prev = np.zeros_like(y)
    t_prev = 0.0
    out_slices, otm, prev_sl = [], [], None
    for s in sorted(slices, key=lambda z: z.T):
        k, iv_mid, T = s.k, s.iv_mid, s.T
        wt = 1.0 / np.maximum(s.iv_spread, 1e-3)
        wt = wt / np.sqrt(np.mean(wt ** 2))
        knots = _knots(k, max_knots)
        dt = T - t_prev
        n_steps = n_steps_for(dt, T, dt_min, steps_per_T)
        if prev_sl is not None:
            x0 = np.log(prev_sl.sigma(knots))
        else:
            x0 = np.log(np.interp(knots, np.sort(k), iv_mid[np.argsort(k)]))
        x0 = np.clip(x0, np.log(0.01) + 1e-9, np.log(3.0) - 1e-9)

        def model(p, o_prev=o_prev, knots=knots, dt=dt, n_steps=n_steps, k=k, T=T):
            o = step(y, o_prev, np.exp(np.interp(y, knots, p)), dt, n_steps)
            return o, otm_iv(interp_otm(y, o, k), k, T)

        def resid(p, model=model, iv_mid=iv_mid, wt=wt):
            _, iv = model(p)
            r = np.where(np.isfinite(iv), (iv - iv_mid) * wt, 1.0)
            reg = lam_smooth * np.diff(p, 2) if p.size >= 3 else np.zeros(0)
            return np.concatenate([r, reg])

        sol = least_squares(resid, x0, bounds=(np.log(0.01), np.log(3.0)), method="trf", max_nfev=max_nfev, x_scale=1.0)
        o, iv = model(sol.x)
        ok = np.isfinite(iv)
        err = iv[ok] - iv_mid[ok]
        sl = AHSlice(label=s.label, T=T, knots=knots, log_sigma=sol.x, n_steps=n_steps,
                     rmse_vol=float(np.sqrt(np.mean(err ** 2))) if ok.any() else np.nan,
                     within_bid_ask=float(np.mean((iv >= s.iv_bid) & (iv <= s.iv_ask))), n=int(k.size),
                     success=bool(sol.success) and bool(ok.all()))
        out_slices.append(sl)
        otm.append(o)
        o_prev, t_prev, prev_sl = o, T, sl
    surf = AHSurface(y=y, Ts=np.array([sl.T for sl in out_slices]), slices=out_slices, otm=otm, dt_min=dt_min,
                     steps_per_T=steps_per_T,
                     meta={"y_max": y_max, "n_y": n_y, "beta": beta, "lam_smooth": lam_smooth, "max_knots": max_knots})
    return surf, out_slices
