"""Black-Scholes / Black-76 prices, Greeks and implied volatility."""
from __future__ import annotations

import numpy as np
import pytest

from voldesk.pricing import IVStatus, black_price, bsm_greeks, bsm_price, implied_vol_black, implied_vol_bsm
from voldesk.pricing.bsm import normalized_black


def test_hull_textbook_values():
    # Hull, Options Futures and Other Derivatives, Example 15.6: c = 4.76, p = 0.81
    assert bsm_price(1, 42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.7594, abs=1e-4)
    assert bsm_price(-1, 42, 40, 0.5, 0.10, 0.20) == pytest.approx(0.8086, abs=1e-4)


def test_put_call_parity_grid():
    rng = np.random.default_rng(1)
    S, K = rng.uniform(50, 150, 200), rng.uniform(50, 150, 200)
    T, r, q, s = rng.uniform(0.01, 3, 200), rng.uniform(-0.01, 0.08, 200), rng.uniform(0, 0.05, 200), rng.uniform(0.05, 1.5, 200)
    c, p = bsm_price(1, S, K, T, r, s, q), bsm_price(-1, S, K, T, r, s, q)
    assert np.max(np.abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T)))) < 1e-10


def test_normalized_black_against_mpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    for x, s in [(-0.5, 0.1), (-2.0, 0.3), (-0.05, 0.02), (-1.0, 1.0), (0.0, 0.2)]:
        ref = mp.e ** (x / 2) * mp.ncdf(x / s + s / 2) - mp.e ** (-x / 2) * mp.ncdf(x / s - s / 2)
        got = float(normalized_black(x, s, 1))
        assert abs(got - float(ref)) <= 1e-12 * max(abs(float(ref)), 1e-300) + 1e-300


def test_implied_vol_round_trip():
    rng = np.random.default_rng(2)
    F, T, s = 100.0, rng.uniform(1 / 365, 5, 2000), rng.uniform(0.03, 2.0, 2000)
    z = rng.uniform(-3, 3, 2000)
    K = F * np.exp(z * s * np.sqrt(T))
    cp = np.where(rng.uniform(size=2000) > 0.5, 1.0, -1.0)
    px = black_price(cp, F, K, T, s, 0.97)
    iv = implied_vol_black(px, cp, F, K, T, 0.97)
    ok = px > 1e-10
    assert np.nanmax(np.abs(iv[ok] - s[ok]) / s[ok]) < 1e-8


def test_implied_vol_status_codes():
    r = implied_vol_black(np.array([0.5, 200.0, 5.0]), 1, 100.0, np.array([90.0, 100.0, 100.0]), 1.0, 1.0,
                          full_output=True)
    assert r.status[0] == IVStatus.BELOW_INTRINSIC          # below intrinsic value 10
    assert r.status[1] == IVStatus.ABOVE_UPPER_BOUND        # call worth more than the forward
    assert r.status[2] == IVStatus.OK
    assert implied_vol_bsm(bsm_price(1, 100, 105, 0.3, 0.02, 0.25, 0.01), 1, 100, 105, 0.3, 0.02, 0.01) == pytest.approx(0.25, rel=1e-10)


@pytest.mark.parametrize("cp", [1, -1])
def test_greeks_against_finite_differences(cp):
    S, K, T, r, s, q = 100.0, 97.0, 0.4, 0.03, 0.22, 0.01
    g = bsm_greeks(cp, S, K, T, r, s, q)
    f = lambda **kw: float(bsm_price(cp, kw.get("S", S), K, kw.get("T", T), kw.get("r", r), kw.get("s", s), q))  # noqa: E731
    h = 1e-4
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(s=s + h) - f(s=s - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    d = lambda **kw: float(bsm_greeks(cp, kw.get("S", S), K, kw.get("T", T), r, kw.get("s", s), q)["delta"])  # noqa: E731
    assert g["vanna"] == pytest.approx((d(s=s + h) - d(s=s - h)) / (2 * h), rel=1e-5)
    assert g["charm"] == pytest.approx(-(d(T=T + h) - d(T=T - h)) / (2 * h), rel=1e-4)
