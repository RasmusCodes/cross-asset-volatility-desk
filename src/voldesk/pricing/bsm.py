"""European option prices under Black-Scholes-Merton (spot) and Black-76 (forward).

Conventions (used throughout ``voldesk``)
----------------------------------------
cp      +1 call / -1 put. Also accepts 'c'/'call'/'p'/'put' (any case) or booleans (True = call).
S, F    spot price / forward price of the underlying for the option's expiry.
K       strike.
T       time to expiry in years. The caller chooses the clock; the data layer uses ACT/365F.
r       continuously compounded risk-free rate per year.
q       continuously compounded dividend yield (or any carry adjustment) per year. The cost of
        carry is b = r - q: q = r gives Black-76 on a futures price, q = r_foreign gives
        Garman-Kohlhagen for FX.
sigma   annualised lognormal volatility.
discount  discount factor D to the payment date (Black-76 interface).

Argument order follows py_vollib: ``bsm_price(cp, S, K, T, r, sigma, q)``.
All functions broadcast over numpy arrays and return float arrays (0-d for scalar inputs).

Numerics
--------
Every price is computed as D * sqrt(F K) * b(x, s, cp) with x = ln(F/K), s = sigma sqrt(T)
and b the normalised Black function (notation of Jaeckel, "Let's Be Rational", 2015):

* the out-of-the-money part uses, for d1 < -1,
      b = 1/2 exp(-x^2/(2 s^2) - s^2/8) [erfcx(-d1/sqrt2) - erfcx(-d2/sqrt2)],
  an exact rewrite of e^{x/2} N(d1) - e^{-x/2} N(d2) (Jaeckel's general-region formula). It
  puts the Gaussian tail factor, whose argument-rounding error the textbook formula amplifies,
  in a common prefactor and does not underflow. Against 50-digit mpmath (tests/test_pricing.py)
  the relative error is ~1e-13 for d1 < -2 versus ~1e-11 for the
  textbook formula. For d1 >= -1 the textbook formula is used (slightly more accurate there).
  What remains is cancellation: the relative error of the time value is of order
  eps * (1 + |x|/s^2 + 1/s), i.e. it grows for far-OTM low-variance inputs (|x|/s^2 large;
  measured <= 4e-13 for |x|/s^2 < 1000, ~1e-11 at |x|/s^2 ~ 2e4) and for near-the-money inputs
  with tiny s (1/s; ~4e-13 at s = 0.001, e.g. a 0DTE option ~30 minutes before expiry).
  Jaeckel's small-s and large-|x|/s expansions would remove both terms; at quote precision
  they are irrelevant, so they are not implemented;
* an in-the-money price is intrinsic value plus the value of the out-of-the-money option at the
  same strike (put-call parity), so the time value of deep ITM options is not lost to rounding.

Degenerate inputs: T = 0 or sigma = 0 give the discounted forward intrinsic value
D * max(cp (F - K), 0). Non-finite inputs, S/F/K <= 0, T < 0 or sigma < 0 give NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfcx, ndtr

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


def cp_sign(cp) -> np.ndarray:
    """Map call/put flags to +1.0 / -1.0 (array)."""
    a = np.asarray(cp)
    if a.dtype.kind in "US" or a.dtype == object:
        low = np.char.lower(a.astype(str))
        out = np.where(np.isin(low, ["c", "call"]), 1.0, np.where(np.isin(low, ["p", "put"]), -1.0, np.nan))
        if np.isnan(out).any():
            raise ValueError("cp strings must be 'c', 'call', 'p' or 'put'")
        return out
    if a.dtype == bool:
        return np.where(a, 1.0, -1.0)
    out = np.asarray(a, dtype=float)
    if not np.all(np.isin(out, (1.0, -1.0))):
        raise ValueError("numeric cp must be +1 (call) or -1 (put)")
    return out


def norm_pdf(x):
    return np.exp(-0.5 * np.square(x)) / SQRT2PI


def otm_normalized(x, s):
    """Normalised value of the out-of-the-money option, as a function of x <= 0.

    Returns c(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2) for x <= 0 (an OTM call;
    an OTM put at log-moneyness +|x| has the same value). c(x, 0) = 0.
    """
    x, s = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))
    out = np.zeros(x.shape)
    pos = s > 0
    if not np.any(pos):
        return out
    xs, ss = x[pos], s[pos]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        d1 = xs / ss + 0.5 * ss
        d2 = d1 - ss
        res = np.empty_like(xs)
        low = d1 < -1.0
        if np.any(low):
            xl, sl = xs[low], ss[low]
            pref = 0.5 * np.exp(-0.5 * np.square(xl / sl) - 0.125 * np.square(sl))
            res[low] = pref * (erfcx(-d1[low] / SQRT2) - erfcx(-d2[low] / SQRT2))
        up = ~low
        if np.any(up):
            xu = xs[up]
            res[up] = np.exp(0.5 * xu) * ndtr(d1[up]) - np.exp(-0.5 * xu) * ndtr(d2[up])
    out[pos] = np.maximum(res, 0.0)
    return out


def normalized_black(x, s, cp):
    """b(x, s, cp) = undiscounted Black price / sqrt(F K)."""
    th = cp_sign(cp)
    x = np.asarray(x, dtype=float)
    intrinsic = np.maximum(th * 2.0 * np.sinh(0.5 * x), 0.0)
    return intrinsic + otm_normalized(-np.abs(x), s)


def black_price(cp, F, K, T, sigma, discount=1.0):
    """Black-76 price of a European option on forward F, paid with discount factor `discount`."""
    th = cp_sign(cp)
    F, K, T, sigma, D, th = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (F, K, T, sigma, discount, th)))
    bad = ~(np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma) & np.isfinite(D)
            & (F > 0) & (K > 0) & (T >= 0) & (sigma >= 0) & (D > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log(F / K)
        s = sigma * np.sqrt(np.maximum(T, 0.0))
        val = D * np.sqrt(F * K) * normalized_black(np.where(bad, 0.0, x), np.where(bad, 0.0, s), th)
    return np.where(bad, np.nan, val)[()]


def forward_price(S, T, r, q=0.0):
    """F = S exp((r - q) T)."""
    return np.asarray(S, dtype=float) * np.exp((np.asarray(r, dtype=float) - np.asarray(q, dtype=float)) * np.asarray(T, dtype=float))


def bsm_price(cp, S, K, T, r, sigma, q=0.0):
    """Generalised Black-Scholes-Merton price (continuous dividend yield q)."""
    S, T, r, q = (np.asarray(v, dtype=float) for v in (S, T, r, q))
    with np.errstate(invalid="ignore"):
        F = forward_price(S, T, r, q)
        D = np.exp(-r * T)
    return black_price(cp, F, K, T, sigma, D)


def intrinsic_bounds(cp, F, K, discount=1.0):
    """(lower, upper) no-arbitrage bounds of a European price: D max(cp(F-K),0) and D F (call) / D K (put)."""
    th = cp_sign(cp)
    F, K, D = (np.asarray(v, dtype=float) for v in (F, K, discount))
    lower = D * np.maximum(th * (F - K), 0.0)
    upper = D * np.where(th > 0, F, K)
    return lower, upper
