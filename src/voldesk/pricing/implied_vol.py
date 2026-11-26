"""Implied volatility: invert Black-76 / Black-Scholes-Merton European prices.

Algorithm (vectorised over arrays)
----------------------------------
1. Normalise: x = ln(F/K), beta = price / (D sqrt(F K)).
2. Remove intrinsic value: tv = beta - max(th 2 sinh(x/2), 0). By put-call parity tv is the
   normalised value of the out-of-the-money option at the same strike, and by symmetry every
   problem becomes "OTM call at x = -|x|": solve c(x, s) = tv for s = sigma sqrt(T). In-the-money
   inputs are therefore inverted through their far better conditioned OTM counterpart.
3. No-arbitrage checks (status codes below): tv < 0 means the price is below intrinsic value;
   tv >= e^{x/2} means a call above the discounted forward / a put above the discounted strike.
4. x = 0 has a closed form: s = 2 sqrt(2) erfinv(tv).
5. Otherwise: safeguarded Halley iterations on g(s) = ln c(x, s) - ln tv. The inflection point
   s_c = sqrt(2|x|) splits the domain. If tv < c(x, s_c) the root lies in [s_lo, s_c] with
   s_lo = |x| / sqrt(-2 ln tv), a guaranteed lower bound because c(x, s) < exp(-x^2/(2 s^2))/2.
   Otherwise it lies in [s_c, s_hi] with s_hi found by doubling. The start is the Corrado-Miller
   approximation when it falls inside the bracket; any step leaving the bracket is replaced by
   bisection, so convergence does not depend on the starting point.
6. sigma = s / sqrt(T).

Only European prices are valid inputs. An American price fed to this solver returns the
European-equivalent volatility, biased upward by the early-exercise premium.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.special import erfinv

from .bsm import SQRT2, SQRT2PI, cp_sign, otm_normalized

EPS = np.finfo(float).eps


class IVStatus(IntEnum):
    OK = 0
    BELOW_INTRINSIC = 1      # price < discounted forward intrinsic value: no volatility fits
    AT_INTRINSIC = 2         # zero time value (within rounding): sigma = 0 returned
    ABOVE_UPPER_BOUND = 3    # call >= D F or put >= D K: no finite volatility fits
    INVALID_INPUT = 4        # non-finite, F/K/D <= 0 or T <= 0
    NOT_CONVERGED = 5        # iteration cap hit (not observed in the validation grids)


@dataclass(frozen=True)
class IVResult:
    sigma: np.ndarray
    status: np.ndarray
    iterations: np.ndarray


def _solve_otm(x: np.ndarray, tv: np.ndarray, max_iter: int):
    """Solve c(x, s) = tv for s, with x < 0 and 0 < tv < exp(x/2). Returns (s, iterations, ok)."""
    m = x.size
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        lnb = np.log(tv)
        sc = np.sqrt(-2.0 * x)
        upper = tv >= otm_normalized(x, sc)
        lo = np.where(upper, sc, -x / np.sqrt(-2.0 * lnb))
        hi = np.where(upper, np.inf, sc)
        ok = np.ones(m, dtype=bool)
        if upper.any():
            u = np.flatnonzero(upper)
            h = np.maximum(2.0 * sc[u], sc[u] + 1.0)
            for _ in range(60):
                need = otm_normalized(x[u], h) <= tv[u]
                if not need.any():
                    break
                h = np.where(need, 2.0 * h, h)
            hi[u] = h
            ok[u] = ~need                      # could not bracket (tv within rounding of the bound)
        # Corrado-Miller start, written in normalised coordinates.
        sh, ch = np.sinh(0.5 * x), np.cosh(0.5 * x)
        a = tv - sh
        s_cm = SQRT2PI / (2.0 * ch) * (a + np.sqrt(np.maximum(a * a - 4.0 * sh * sh / np.pi, 0.0)))
        inside = np.isfinite(s_cm) & (s_cm > lo) & (s_cm < hi)
        s = np.where(inside, s_cm, np.where(upper, 0.5 * (lo + hi), lo))

    iters = np.zeros(m, dtype=np.int16)
    done = ~ok
    for _ in range(max_iter):
        a_idx = np.flatnonzero(~done)
        if a_idx.size == 0:
            break
        xa, sa = x[a_idx], s[a_idx]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            c = otm_normalized(xa, sa)
            g = np.log(c) - lnb[a_idx]
            vega = np.exp(-0.5 * np.square(xa / sa) - 0.125 * sa * sa) / SQRT2PI
            g1 = vega / c
            g2 = g1 * (xa * xa / sa ** 3 - 0.25 * sa) - g1 * g1
            newton = -g / g1
            den = 1.0 + 0.5 * newton * g2 / g1
            step = np.where(np.isfinite(den) & (den > 0.5), newton / den, newton)
            lo_a = np.where(g < 0, np.maximum(lo[a_idx], sa), lo[a_idx])
            hi_a = np.where(g > 0, np.minimum(hi[a_idx], sa), hi[a_idx])
            s_new = sa + step
            out = ~np.isfinite(s_new) | (s_new <= lo_a) | (s_new >= hi_a)
            s_next = np.where(out, 0.5 * (lo_a + hi_a), s_new)
            small_res = np.abs(g) <= 32 * EPS
            s_next = np.where(small_res & np.isfinite(s_new) & (s_new > 0), s_new, s_next)
            # Stop once a (taken) Halley step is below 2^-40 s. The iteration converges cubically,
            # so the accepted point is then at the round-off floor of g, which can sit well above
            # 32 eps in cancellation-prone regions (small s or |x|/s^2 large); a tighter step test
            # would only wander on that floor until the bracket collapsed.
            conv = small_res | (~out & (np.abs(step) <= 2.0 ** -40 * sa)) | (hi_a - lo_a <= 2.0 ** -50 * hi_a)
        s[a_idx], lo[a_idx], hi[a_idx] = s_next, lo_a, hi_a
        iters[a_idx] += 1
        done[a_idx] = conv
    ok &= done
    return s, iters, ok


def implied_vol_black(price, cp, F, K, T, discount=1.0, *, max_iter: int = 50, full_output: bool = False):
    """Black-76 implied volatility of a European option price (discounted with `discount`).

    Returns sigma (NaN where no volatility fits; 0 at exactly intrinsic value), or an IVResult
    with per-element ``IVStatus`` codes and iteration counts when ``full_output=True``.
    """
    th = cp_sign(cp)
    arrs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, th, F, K, T, discount)))
    shape = arrs[0].shape
    price, th, F, K, T, D = (a.ravel() for a in arrs)
    n = price.size
    sigma = np.full(n, np.nan)
    status = np.full(n, IVStatus.INVALID_INPUT, dtype=np.int8)
    iters = np.zeros(n, dtype=np.int16)

    valid = (np.isfinite(price) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(D)
             & (F > 0) & (K > 0) & (T > 0) & (D > 0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = np.log(F / K)
        beta = price / (D * np.sqrt(F * K))
        intrinsic = np.maximum(th * 2.0 * np.sinh(0.5 * x), 0.0)
        tv = beta - intrinsic
        ax = -np.abs(x)
        upper_bound = np.exp(0.5 * ax)
        tol_int = 4.0 * EPS * (intrinsic + np.abs(beta))

    below = valid & (tv < -tol_int)
    at = valid & ~below & (tv <= tol_int)
    above = valid & ~below & ~at & (tv >= upper_bound)
    solve = valid & ~below & ~at & ~above
    status[below] = IVStatus.BELOW_INTRINSIC
    status[at] = IVStatus.AT_INTRINSIC
    sigma[at] = 0.0
    status[above] = IVStatus.ABOVE_UPPER_BOUND

    s = np.full(n, np.nan)
    atm = solve & (ax == 0.0)
    s[atm] = 2.0 * SQRT2 * erfinv(tv[atm])
    status[atm] = IVStatus.OK
    idx = np.flatnonzero(solve & (ax < 0.0))
    if idx.size:
        s_i, it_i, ok_i = _solve_otm(ax[idx], tv[idx], max_iter)
        s[idx] = s_i
        iters[idx] = it_i
        status[idx] = np.where(ok_i, IVStatus.OK, IVStatus.NOT_CONVERGED)
    good = status == IVStatus.OK
    sigma[good] = s[good] / np.sqrt(T[good])

    sigma = sigma.reshape(shape)[()]
    if full_output:
        return IVResult(sigma=sigma, status=status.reshape(shape)[()], iterations=iters.reshape(shape)[()])
    return sigma


def implied_vol_bsm(price, cp, S, K, T, r, q=0.0, **kwargs):
    """Black-Scholes-Merton implied volatility (continuous dividend yield q); see implied_vol_black."""
    S, T, r, q = (np.asarray(v, dtype=float) for v in (S, T, r, q))
    with np.errstate(invalid="ignore", over="ignore"):
        F = S * np.exp((r - q) * T)
        D = np.exp(-r * T)
    return implied_vol_black(price, cp, F, K, T, D, **kwargs)
