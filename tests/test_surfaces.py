"""Surface models and static-arbitrage diagnostics, on SYNTHETIC (Heston-generated) chains."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voldesk.data import chains
from voldesk.data.synthetic import DEFAULT_PARAMS, synthetic_chain, synthetic_curve
from voldesk.surface import compare, heston, sabr, svi
from voldesk.surface.slices import slices_from_chain


@pytest.fixture(scope="module")
def synth_slices():
    raw = synthetic_chain("2024-06-03", spot=5000.0, am_settlement=1, seed=3)
    spot = pd.Series({pd.Timestamp("2024-06-03"): 5000.0})
    base = chains.base_frame(raw, spot, synthetic_curve(0.04))
    fwd = chains.parity_forwards(base)
    ch = chains.add_forwards_and_iv(base, fwd)
    return slices_from_chain(ch)


def test_svi_g_function_and_density():
    p = svi.SVIParams(a=0.01, b=0.1, rho=-0.5, m=0.02, s=0.15)
    k = np.linspace(-2, 1, 2001)
    g = svi.g_function(k, p.w(k), p.dw(k), p.d2w(k))
    assert g.min() > 0
    dens = svi.density(k, p.w(k), p.dw(k), p.d2w(k))
    assert np.trapezoid(dens, k) == pytest.approx(1.0, abs=0.02)


def test_svi_slice_recovers_parameters():
    true = svi.SVIParams(a=0.005, b=0.08, rho=-0.6, m=0.03, s=0.12)
    k = np.linspace(-0.6, 0.3, 40)
    T = 0.5
    iv = np.sqrt(true.w(k) / T)
    f = svi.fit_svi_slice(k, iv, T, iv_bid=iv - 0.002, iv_ask=iv + 0.002)
    assert f.rmse_vol < 1e-4
    assert f.min_g >= -1e-6


def test_heston_cos_matches_quadrature():
    p = heston.HestonParams(0.04, 2.0, 0.05, 0.8, -0.7)
    K = np.linspace(70, 140, 15)
    for T in (0.1, 0.5, 2.0):
        a = heston.heston_price(np.where(K >= 100, 1.0, -1.0), 100.0, K, T, 0.98, p)
        b = heston.heston_price_cos(np.where(K >= 100, 1.0, -1.0), 100.0, K, T, 0.98, p)
        assert np.max(np.abs(a - b)) < 1e-7


def test_sabr_beta1_recovers_parameters():
    F, T = 100.0, 0.25
    K = F * np.exp(np.linspace(-0.3, 0.15, 25))
    iv = sabr.sabr_iv(K, F, T, 0.2, 1.0, -0.6, 1.2)
    f = sabr.fit_sabr_slice(K, F, T, iv, iv_bid=iv - 0.001, iv_ask=iv + 0.001)
    assert (f.alpha, f.rho, f.nu) == pytest.approx((0.2, -0.6, 1.2), rel=1e-3)


def test_parity_forwards_recover_true_forward():
    raw = synthetic_chain("2024-06-03", spot=5000.0, r=0.04, q=0.013, seed=4)
    spot = pd.Series({pd.Timestamp("2024-06-03"): 5000.0})
    base = chains.base_frame(raw, spot, synthetic_curve(0.04))
    fwd = chains.parity_forwards(base)
    T = base.groupby("exdate")["T"].first()
    fw = fwd.set_index("exdate")["F"]
    true = 5000.0 * np.exp((0.04 - 0.013) * T)
    assert np.max(np.abs(fw.reindex(true.index) / true - 1)) < 5e-4      # within 5 bp despite quote noise


def test_all_models_fit_synthetic_surface(synth_slices):
    assert len(synth_slices) >= 5
    fits = compare.fit_all(synth_slices)
    for m in ("svi", "essvi", "ssvi", "sabr", "heston"):
        rows = pd.DataFrame(fits[m].slices)
        assert len(rows) == len(synth_slices), m
        assert rows["rmse"].median() < 1.5, m                            # vol points
    # the Heston fit to Heston-generated quotes is close to the generating parameters
    p = fits["heston"].params
    assert p["rho"] == pytest.approx(DEFAULT_PARAMS.rho, abs=0.15)
    # eSSVI and SSVI are arbitrage-free on the whole grid by construction
    for m in ("essvi", "ssvi"):
        a = fits[m].arbitrage
        assert a["share_g_neg_quoted"] == 0 and a["share_g_neg_extrap"] == 0 and a["share_calendar_viol"] == 0, m


def test_arbitrage_metrics_flag_crossing_slices():
    good = svi.SVIParams(0.01, 0.1, -0.5, 0.0, 0.1)
    bad = svi.SVIParams(0.0, 0.05, -0.5, 0.0, 0.1)          # lower total variance at the longer maturity
    surf = svi.SVISurface([0.1, 0.5], [good, bad])

    class S:  # minimal slice stand-ins for the quoted ranges
        def __init__(self, T):
            self.T, self.k = T, np.linspace(-0.3, 0.2, 10)
    m = compare.arbitrage_metrics(surf, [S(0.1), S(0.5)])
    assert m["share_calendar_viol"] > 0
