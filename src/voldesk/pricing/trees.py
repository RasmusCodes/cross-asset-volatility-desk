"""Lattice pricers: Cox-Ross-Rubinstein and Leisen-Reimer binomial trees, and a trinomial tree.

European and American exercise, continuous yield q, discrete dividends (spot or escrowed model,
see ``voldesk.dividends``), and the early-exercise boundary. Arrays of options are priced in one
batch: every option gets its own tree with the same number of steps, and the backward induction
runs over the whole batch at once.

Trees (dt = T / N, b = r - q, discount exp(-r dt) per step)
-----------------------------------------------------------
``crr``  Cox, Ross & Rubinstein (1979): u = exp(sigma sqrt dt), d = 1/u,
         p = (exp(b dt) - d) / (u - d). Converges at O(1/N) for European options, with an
         oscillating coefficient (the strike's position between terminal nodes changes with N).
``lr``   Leisen & Reimer (1996), Peizer-Pratt method-2 inversion h(z) of the normal CDF:
         p = h(d2), p' = h(d1), u = exp(b dt) p'/p, d = (exp(b dt) - p u)/(1 - p). The tree is
         centred on the strike, so European prices converge smoothly at O(1/N^2). N must be odd;
         an even N is raised by one (as in QuantLib).
trinomial  Log-space lattice with spacing dx = lambda sigma sqrt(dt) (lambda = sqrt 3 by default,
         which matches the fourth moment of the normal increment). pu, pm, pd match the first two
         moments of S exactly, E[S'] = S exp(b dt) and E[S'^2] = S^2 exp((2b + sigma^2) dt), so the
         lattice prices forwards exactly and put-call parity holds on it.

All three are martingale lattices (the discounted spot is exact), so European put-call parity
holds on each tree to rounding.

Smoothing and extrapolation (optional, any tree): ``smoothing=True`` prices the last step with the
European formula (Broadie & Detemple's BBS), which removes the payoff kink from the lattice;
``richardson=True`` combines N and ~N/2 steps assuming an O(1/N) error. Together (BBSR) they give
American prices about an order of magnitude more accurate than the plain lattice at equal cost
(measured in reports/03_pricing_engine.md); for the trinomial tree the gain is smaller.

Dividends
---------
* ``dividend_model="spot"``: the spot jumps S -> S (1 - f) - D at the ex time. The ex time is
  snapped to the nearest step (an O(dt) timing error, the same order as the tree's own American
  error) and the jump condition V(S, t-) = V(S (1 - f) - D, t+) is imposed by cubic Lagrange
  interpolation in log-price on that step's nodes, with linear extrapolation in price outside
  the lattice. A jump that lands on the expiry step uses the payoff directly. American options
  may be exercised on both sides of the ex time; exercise just before it is what makes calls
  exercise early.
* ``dividend_model="escrowed"``: the lattice is built on S* = S - E_0 and exercise is tested
  against S* + E_t (E = escrowed amount of the remaining dividends, see voldesk.dividends). The
  escrowed amounts use the exact dividend times; exercise just before an ex date is approximated
  by the last step before it.
* With discrete dividends every lattice converges at O(1/N) without a smooth error expansion (the
  ex-date snapping and the one-step exercise lag vary with N), so Richardson extrapolation does
  not help there. ``voldesk.fd`` places ex dates exactly and is the accurate engine for them.

Errors and limits
-----------------
Invalid inputs (non-finite, S, K, T or sigma <= 0) give NaN with ``LatticeStatus.INVALID_INPUT``;
T = 0 gives the payoff. If a branch probability falls outside [0, 1] (CRR or trinomial with too
few steps for the drift, |b| sqrt(dt) > sigma roughly) the price is NaN with
``LatticeStatus.BAD_PROBABILITY``: raise the number of steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .bsm import bsm_price, cp_sign
from .dividends import DividendSchedule, as_schedule


class LatticeStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1        # non-finite input, S/K/T/sigma <= 0, or S* <= 0 in the escrowed model
    BAD_PROBABILITY = 2      # a branch probability outside [0, 1]: use more steps


@dataclass(frozen=True)
class LatticeResult:
    price: np.ndarray
    status: np.ndarray
    steps: int
    times: np.ndarray | None = None      # (..., steps + 1) time of each step, years from valuation
    boundary: np.ndarray | None = None   # (..., steps + 1) critical spot per step, NaN = no exercise


def _peizer_pratt(z, n):
    """Peizer-Pratt method-2 inversion used by Leisen-Reimer (n odd)."""
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    one_minus_e = -np.expm1(-a * a * (n + 1.0 / 6.0))
    return 0.5 + np.where(z > 0, 1.0, -1.0) * 0.5 * np.sqrt(one_minus_e)


def _prepare(cp, S, K, T, r, sigma, q):
    th = cp_sign(cp)
    arrs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (th, S, K, T, r, sigma, q)))
    shape = arrs[0].shape
    th, S, K, T, r, sigma, q = (a.ravel().copy() for a in arrs)
    valid = (np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(sigma)
             & np.isfinite(q) & (S > 0) & (K > 0) & (T >= 0) & (sigma > 0))
    return shape, th, S, K, T, r, sigma, q, valid


def _lagrange_weights(t, k):
    """Lagrange basis weights at position t for nodes 0..k-1 (k = 2, 3 or 4)."""
    if k == 4:
        return (-(t - 1) * (t - 2) * (t - 3) / 6.0, t * (t - 2) * (t - 3) / 2.0,
                -t * (t - 1) * (t - 3) / 2.0, t * (t - 1) * (t - 2) / 6.0)
    if k == 3:
        return ((t - 1) * (t - 2) / 2.0, -t * (t - 2), t * (t - 1) / 2.0)
    return (1.0 - t, t)


def interp_log_lattice(V, s_lo, h, y):
    """Interpolate lattice values at query prices.

    V    (R, m) values at prices s_lo * exp(j h), j = 0..m-1 (one row per option)
    y    (R, m) query prices (may be <= 0)
    Cubic Lagrange in log-price inside [S_0, S_{m-1}] (quadratic/linear if m < 4); linear in price
    outside the lattice, with prices <= 0 mapped to 0.
    """
    R, m = V.shape
    if m == 1:
        return np.repeat(V, y.shape[1], axis=1) if y.shape[1] != 1 else V.copy()
    s_lo = s_lo[:, None]
    h = h[:, None]
    ypos = np.maximum(y, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(ypos > 0, np.log(np.where(ypos > 0, ypos, 1.0) / s_lo) / h, -np.inf)
    out = np.empty_like(y)
    inside = (g >= 0) & (g <= m - 1)
    k = min(4, m)
    gi = np.where(inside, g, 0.0)
    i0 = np.clip(np.floor(gi).astype(np.int64) - (k // 2 - 1), 0, m - k)
    t = gi - i0
    w = _lagrange_weights(t, k)
    acc = np.zeros_like(y)
    for j in range(k):
        acc += w[j] * np.take_along_axis(V, i0 + j, axis=1)
    out[:] = acc
    # linear extrapolation in price outside the lattice
    s0, s1 = s_lo[:, 0], s_lo[:, 0] * np.exp(h[:, 0])
    lo = g < 0
    if lo.any():
        slope = (V[:, 1] - V[:, 0]) / (s1 - s0)
        ext = V[:, :1] + slope[:, None] * (ypos - s0[:, None])
        out[lo] = ext[lo]
    hi = g > m - 1
    if hi.any():
        sa = s_lo[:, 0] * np.exp((m - 2) * h[:, 0])
        sb = s_lo[:, 0] * np.exp((m - 1) * h[:, 0])
        slope = (V[:, m - 1] - V[:, m - 2]) / (sb - sa)
        ext = V[:, m - 1:m] + slope[:, None] * (y - sb[:, None])
        out[hi] = ext[hi]
    return out


def _dividend_events(sched: DividendSchedule, T, dt, N):
    """Map each (option, dividend) pair with 0 < t < T to a step; returns {step: [(rows, cash, frac), ...]}
    with the events of a step listed latest-first (the order the backward induction meets them)."""
    events: dict[int, list] = {}
    for i in range(len(sched) - 1, -1, -1):
        ti, ci, fi = sched.times[i], sched.cash[i], sched.fraction[i]
        rows = np.flatnonzero((ti > 0) & (ti < T))
        if rows.size == 0:
            continue
        steps = np.clip(np.rint(ti / dt[rows]).astype(np.int64), 1, N)
        for s in np.unique(steps):
            sel = rows[steps == s]
            events.setdefault(int(s), []).append((sel, ci, fi))
    return events


def _escrow_at(sched: DividendSchedule, t, T, b):
    """Escrowed amount at time t (array over options) of cash dividends with t < t_i < T, at carry b."""
    out = np.zeros_like(t)
    for ti, ci in zip(sched.times, sched.cash):
        if ci > 0:
            out += np.where((ti > t) & (ti < T), ci * np.exp(-b * (ti - t)), 0.0)
    return out


def _payoff(th, X, K):
    v = th[:, None] * (X - K[:, None])
    return np.maximum(v, 0.0, out=v)


def _lattice_batch(kind, th, S, K, T, r, sigma, q, N, american, sched, model, want_boundary, stretch, smooth=False):
    """Backward induction for a batch of valid options with T > 0. Returns (price, status, boundary)."""
    B = S.size
    dt = T / N
    b = r - q
    disc = np.exp(-r * dt)
    status = np.zeros(B, dtype=np.int8)
    escrowed = model == "escrowed" and sched.has_cash
    pv0 = sched.pv_cash(T, b) if sched.has_cash else np.zeros(B)
    if escrowed:
        S0 = S - pv0
    else:
        S0 = S.copy()
    bad = ~(S0 > 0)

    # ----- lattice parameters
    if kind == "trinomial":
        dx = stretch * sigma * np.sqrt(dt)
        E = np.exp(dx)
        m1, m2 = np.expm1(b * dt), np.expm1((2.0 * b + sigma * sigma) * dt)
        em1 = np.expm1(dx)
        den = em1 * em1 * (E + 1.0)
        pu = (m2 * E - m1 * (E + 1.0)) / den
        pd = E * E * (m2 - m1 * (E + 1.0)) / den
        pm = 1.0 - pu - pd
        badp = ~((pu >= 0) & (pd >= 0) & (pm >= 0))
        h = dx
        width = 2 * N + 1
        lnS = np.log(np.where(bad, 1.0, S0))[:, None] + (np.arange(width) - N)[None, :] * dx[:, None]
    else:
        if kind == "crr":
            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u
            p = (np.exp(b * dt) - d) / (u - d)
        elif kind == "lr":
            # Centre on the strike using the forward net of dividends (the escrowed spot for cash
            # dividends, S prod(1 - f) for proportional ones).
            Sc = S0.copy()
            if not escrowed and len(sched):
                Sc = S0 - pv0
                for ti, fi in zip(sched.times, sched.fraction):
                    if fi > 0:
                        Sc = Sc * np.where((ti > 0) & (ti < T), 1.0 - fi, 1.0)
                Sc = np.where(Sc > 0, Sc, S0)
            vol = sigma * np.sqrt(T)
            with np.errstate(divide="ignore", invalid="ignore"):
                d1 = (np.log(np.where(bad, 1.0, Sc) / K) + (b + 0.5 * sigma * sigma) * T) / vol
            d2 = d1 - vol
            p = _peizer_pratt(d2, N)
            pp = _peizer_pratt(d1, N)
            g = np.exp(b * dt)
            u = g * pp / p
            d = (g - p * u) / (1.0 - p)
        else:
            raise ValueError(f"unknown tree {kind!r}")
        badp = ~((p > 0) & (p < 1) & (d > 0) & (u > d))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.log(u / d)
            lnS = (np.log(np.where(bad, 1.0, S0)) + N * np.log(d))[:, None] + np.arange(N + 1)[None, :] * h[:, None]
            inv_step = 1.0 / d
    status[badp & ~bad] = LatticeStatus.BAD_PROBABILITY
    status[bad] = LatticeStatus.INVALID_INPUT
    ok = status == LatticeStatus.OK
    if not ok.all():
        # price the healthy part only; keep array shapes consistent
        idx = np.flatnonzero(ok)
        price = np.full(B, np.nan)
        bnd = np.full((B, N + 1), np.nan) if want_boundary else None
        if idx.size:
            sub = _lattice_batch(kind, th[idx], S[idx], K[idx], T[idx], r[idx], sigma[idx], q[idx], N, american,
                                 sched, model, want_boundary, stretch, smooth)
            price[idx] = sub[0]
            if want_boundary:
                bnd[idx] = sub[2]
        return price, status, bnd

    Sn = np.exp(lnS)                                   # terminal node prices (S* if escrowed)
    V = _payoff(th, Sn, K)
    events = {} if escrowed else _dividend_events(sched, T, dt, N)
    bnd = np.full((B, N + 1), np.nan) if (want_boundary and american) else None

    def node_low(n):
        """Lowest node price at step n (for interpolation)."""
        if kind == "trinomial":
            return S0 * np.exp(-n * dx)
        return S0 * np.exp(n * np.log(d))

    def apply_jumps(n, V, Sn):
        # V holds values just after the ex time (t_n+); replace them by V(t_n-, S) = V(t_n+, S(1-f) - D).
        # Several events on one step are applied latest-first. Exercise checks are done by the caller.
        if n == N:
            # on the expiry step the post-dividend value is the payoff itself: compose the price
            # drops in calendar order (the event list is latest-first) and evaluate it exactly
            post = Sn.copy()
            touched = np.zeros(B, dtype=bool)
            for rows, ci, fi in reversed(events[n]):
                post[rows] = post[rows] * (1.0 - fi) - ci
                touched[rows] = True
            V[touched] = _payoff(th[touched], post[touched], K[touched])
            return V
        for rows, ci, fi in events.get(n, ()):
            post = Sn[rows] * (1.0 - fi) - ci
            V[rows] = interp_log_lattice(V[rows], node_low(n)[rows], h[rows], post)
        return V

    def exercise(n, V, Sn):
        X = Sn + _escrow_at(sched, n * dt, T, b)[:, None] if escrowed else Sn
        ex = _payoff(th, X, K)
        if bnd is not None:
            m = ex > V
            put = th < 0
            # put: highest exercised node; call: lowest exercised node
            has = m.any(axis=1)
            j_hi = X.shape[1] - 1 - np.argmax(m[:, ::-1], axis=1)
            j_lo = np.argmax(m, axis=1)
            j = np.where(put, j_hi, j_lo)
            bnd[:, n] = np.where(has, X[np.arange(B), j], np.nan)
        return np.maximum(V, ex, out=V)

    if events.get(N):
        V = apply_jumps(N, V, Sn)
    if american:
        V = exercise(N, V, Sn)
    no_event_at_expiry = np.ones(B, dtype=bool)
    for rows, _, _ in events.get(N, ()):
        no_event_at_expiry[rows] = False
    for n in range(N - 1, -1, -1):
        if kind == "trinomial":
            V = disc[:, None] * (pd[:, None] * V[:, :-2] + pm[:, None] * V[:, 1:-1] + pu[:, None] * V[:, 2:])
            Sn = Sn[:, 1:-1]
        else:
            lo = V[:, :-1]
            V = lo + p[:, None] * (V[:, 1:] - lo)
            V *= disc[:, None]
            Sn = Sn[:, :-1] * inv_step[:, None]
        if smooth and n == N - 1:
            # Broadie-Detemple smoothing: the last step is priced with the exact European formula,
            # which removes the payoff kink from the lattice (rows with a dividend inside the last
            # step keep the lattice value, whose jump was applied exactly at expiry)
            bs = bsm_price(th[:, None], Sn, K[:, None], dt[:, None], r[:, None], sigma[:, None], q[:, None])
            V = np.where(no_event_at_expiry[:, None], bs, V)
        if n in events:
            if american:
                V = exercise(n, V, Sn)          # exercise ex-dividend at the step itself
            V = apply_jumps(n, V, Sn)
        if american:
            V = exercise(n, V, Sn)
    return V[:, 0], status, bnd


def _price_lattice(kind, cp, S, K, T, r, sigma, q, steps, american, dividends, dividend_model,
                   full_output, stretch=np.sqrt(3.0), batch_size=1024, smoothing=False, richardson=False):
    if dividend_model not in ("spot", "escrowed"):
        raise ValueError("dividend_model must be 'spot' or 'escrowed'")
    sched = as_schedule(dividends)
    if dividend_model == "escrowed" and sched.has_proportional:
        raise ValueError("the escrowed model is defined for cash dividends only")
    N = int(steps)
    if kind == "lr" and N % 2 == 0:
        N += 1
    if N < 1:
        raise ValueError("steps must be >= 1")
    if richardson:
        # two-point extrapolation assuming error = c / N: (N2 P2 - N1 P1) / (N2 - N1), N1 ~ N/2
        N1 = N // 2 if kind != "lr" else (N // 2) | 1
        p2 = _price_lattice(kind, cp, S, K, T, r, sigma, q, N, american, dividends, dividend_model, False,
                            stretch, batch_size, smoothing, False)
        p1 = _price_lattice(kind, cp, S, K, T, r, sigma, q, N1, american, dividends, dividend_model, False,
                            stretch, batch_size, smoothing, False)
        if full_output:
            raise ValueError("full_output is not available with richardson=True")
        return (N * p2 - N1 * p1) / (N - N1)
    shape, th, S, K, T, r, sigma, q, valid = _prepare(cp, S, K, T, r, sigma, q)
    n = S.size
    price = np.full(n, np.nan)
    status = np.full(n, LatticeStatus.INVALID_INPUT, dtype=np.int8)
    times = boundary = None
    if full_output:
        times = np.full((n, N + 1), np.nan)
        boundary = np.full((n, N + 1), np.nan)
    expired = valid & (T == 0)
    if expired.any():
        price[expired] = np.maximum(th[expired] * (S[expired] - K[expired]), 0.0)
        status[expired] = LatticeStatus.OK
    live = np.flatnonzero(valid & (T > 0))
    for start in range(0, live.size, batch_size):
        idx = live[start:start + batch_size]
        pr, st, bd = _lattice_batch(kind, th[idx], S[idx], K[idx], T[idx], r[idx], sigma[idx], q[idx], N,
                                    american, sched, dividend_model, full_output, stretch, smoothing)
        price[idx] = pr
        status[idx] = st
        if full_output:
            times[idx] = (T[idx] / N)[:, None] * np.arange(N + 1)[None, :]
            if bd is not None:
                boundary[idx] = bd
    price = price.reshape(shape)[()]
    if not full_output:
        return price
    tail = (N + 1,)
    return LatticeResult(price=price, status=status.reshape(shape)[()], steps=N,
                         times=times.reshape(shape + tail), boundary=boundary.reshape(shape + tail))


def binomial_price(cp, S, K, T, r, sigma, q=0.0, *, steps: int = 501, method: str = "lr", american: bool = False,
                   dividends=None, dividend_model: str = "spot", smoothing: bool = False, richardson: bool = False,
                   full_output: bool = False):
    """Binomial-tree price (``method`` 'crr' or 'lr'); broadcasts over array inputs.

    ``dividends``: a DividendSchedule (or iterable of (time, cash) pairs) shared by all options;
    each option uses the events with 0 < t < T. ``smoothing=True`` prices the last step with the
    European formula (Broadie & Detemple's BBS); ``richardson=True`` extrapolates from N and
    about N/2 steps assuming an O(1/N) error (BBSR when combined with smoothing).
    ``full_output=True`` returns a LatticeResult with status codes, the step times and (American)
    the early-exercise boundary.
    """
    if method not in ("crr", "lr"):
        raise ValueError("method must be 'crr' or 'lr'")
    return _price_lattice(method, cp, S, K, T, r, sigma, q, steps, american, dividends, dividend_model, full_output,
                          smoothing=smoothing, richardson=richardson)


def trinomial_price(cp, S, K, T, r, sigma, q=0.0, *, steps: int = 500, american: bool = False, dividends=None,
                    dividend_model: str = "spot", stretch: float = float(np.sqrt(3.0)), smoothing: bool = False,
                    richardson: bool = False, full_output: bool = False):
    """Trinomial-tree price (moment-matched, node spacing stretch * sigma * sqrt(dt)); see binomial_price."""
    return _price_lattice("trinomial", cp, S, K, T, r, sigma, q, steps, american, dividends, dividend_model,
                          full_output, stretch=stretch, smoothing=smoothing, richardson=richardson)
