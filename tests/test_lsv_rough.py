"""Local-stochastic volatility (particle method) and rough Bergomi calibration, on SYNTHETIC data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from voldesk.data import chains
from voldesk.data.synthetic import DEFAULT_PARAMS, synthetic_chain, synthetic_curve
from voldesk.pricing.bsm import black_price
from voldesk.surface import heston as hs
from voldesk.surface import localvol, lsv, rough, svi
from voldesk.surface.slices import Slice, slices_from_chain


# ------------------------------------------------------------------------------------------------ LSV building blocks
def test_time_grid_contains_nodes_and_respects_step():
    t = lsv.time_grid(1.0, nodes=[0.05, 0.3333, 0.7], dt=lsv.graded_dt(1 / 1460, 1 / 365, 0.1))
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    for node in (0.05, 0.3333, 0.7):
        assert np.min(np.abs(t - node)) < 1e-12
    assert np.diff(t).max() <= 1 / 365 + 1e-12
    assert np.diff(t[t <= 0.1]).max() <= 1 / 1460 + 1e-12


@pytest.mark.parametrize("local_linear", [False, True])
def test_kernel_regression_matches_brute_force(local_linear):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000) * 0.1
    y = 0.04 + 0.2 * x + 0.5 * x ** 2 + rng.standard_normal(4000) * 0.01
    h = 0.02
    g, m, w = lsv.kernel_regression(x, y, h, bins_per_h=20, local_linear=local_linear)
    for g0 in np.linspace(-0.2, 0.2, 9):
        u = (x - g0) / h
        kk = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if local_linear:
            A = np.column_stack([np.ones_like(x), x - g0]) * np.sqrt(kk)[:, None]
            bf = np.linalg.lstsq(A, y * np.sqrt(kk), rcond=None)[0][0]
        else:
            bf = (kk * y).sum() / kk.sum()
        assert np.interp(g0, g, m) == pytest.approx(bf, abs=3e-5)
        assert np.interp(g0, g, w) == pytest.approx(kk.sum(), rel=2e-2)


def test_otm_prices_recover_black_prices():
    rng = np.random.default_rng(2)
    T, sig = 0.5, 0.2
    X = -0.5 * sig ** 2 * T + sig * np.sqrt(T) * rng.standard_normal(200_000)
    k = np.array([-0.3, -0.1, 0.0, 0.1, 0.2])
    p_cv, se_cv = lsv.otm_prices(X, k)
    p_raw, se_raw = lsv.otm_prices(X, k, control=False)
    exact = lsv.black_otm(k, T, sig)
    assert np.all(np.abs(p_cv - exact) < 4 * se_cv)
    assert np.all(np.abs(p_raw - exact) < 4 * se_raw)
    assert np.all(se_cv <= se_raw + 1e-15)
    iv, se_iv = lsv.otm_iv(p_cv, k, T, se_cv)
    assert np.all(np.abs(iv - sig) < 4 * se_iv)


def test_heston_cf_time_derivative_and_local_variance():
    p = DEFAULT_PARAMS
    u = np.linspace(0.05, 30, 25)
    for T in (0.02, 0.5):
        phi, dphi = lsv._heston_cf_and_dT(u, T, p)
        fd = (hs.heston_cf(u, T + 1e-6, p) - hs.heston_cf(u, T - 1e-6, p)) / 2e-6
        assert np.max(np.abs(dphi - fd)) < 1e-6

    # Dupire by finite differences of COS call prices: 2 dc/dT / (c_yy - c_y)
    def call(y, T):
        K = np.exp(y)
        return hs.heston_price_cos(np.ones_like(K), 1.0, K, T, 1.0, p, n=4096)
    T = 0.25
    y = np.linspace(-0.4, 0.2, 7)
    eT, ey = 1e-4, 1e-3
    dT = (call(y, T + eT) - call(y, T - eT)) / (2 * eT)
    cy = (call(y + ey, T) - call(y - ey, T)) / (2 * ey)
    cyy = (call(y + ey, T) - 2 * call(y, T) + call(y - ey, T)) / ey ** 2
    assert lsv.heston_local_variance(p, y, T) == pytest.approx(2 * dT / (cyy - cy), rel=2e-3)


# ------------------------------------------------------------------------------------------------ LSV calibration
@pytest.fixture(scope="module")
def synth_essvi():
    raw = synthetic_chain("2024-06-03", spot=5000.0, am_settlement=1, seed=3)
    spot = pd.Series({pd.Timestamp("2024-06-03"): 5000.0})
    base = chains.base_frame(raw, spot, synthetic_curve(0.04))
    ch = chains.add_forwards_and_iv(base, chains.parity_forwards(base))
    slices = slices_from_chain(ch)
    surf, fits = svi.fit_essvi_surface(slices)
    return surf, slices


def test_lsv_with_deterministic_variance_is_local_vol(synth_essvi):
    """sigma = 0: every particle has the same variance V_t, so the regression is exact and L^2 V_t = sigma_D^2."""
    surf, _ = synth_essvi
    p = hs.HestonParams(v0=0.03, kappa=2.0, theta=0.05, sigma=0.0, rho=-0.7)

    def local_var(x, t):
        return localvol.local_variance(surf, x, t)
    cal = lsv.calibrate_lsv(local_var, p, 0.25, n_particles=5_000, dt=1 / 365, seed=0)
    lev = cal.leverage
    for n in (0, 10, 40, len(lev.times) - 2):
        t0, t1 = lev.times[n], lev.times[n + 1]
        # deterministic variance of the full-truncation Euler scheme at t0
        v = p.v0
        for m in range(n):
            v = v + p.kappa * (p.theta - v) * (lev.times[m + 1] - lev.times[m])
        g = lev.grids[n]
        np.testing.assert_allclose(lev.values[n] ** 2 * v, local_var(g, 0.5 * (t0 + t1)), rtol=1e-10)


def test_lsv_reprices_synthetic_surface(synth_essvi):
    """Particle calibration to an eSSVI surface fitted to synthetic Heston quotes, with a different Heston variance
    process (so the leverage is far from 1): the independent pricing pass reprices the surface at the quoted
    strikes within Monte Carlo error."""
    surf, slices = synth_essvi
    sl = [s for s in slices if 25 / 365 <= s.T <= 0.55]
    p = hs.HestonParams(v0=0.01, kappa=3.0, theta=0.03, sigma=0.6, rho=-0.5)
    cal = lsv.calibrate_lsv(lambda x, t: localvol.local_variance(surf, x, t), p, max(s.T for s in sl),
                            nodes=[s.T for s in sl], n_particles=40_000, dt=lsv.graded_dt(1 / 1460, 1 / 365, 0.1),
                            seed=1, atm_vol=lambda t: float(surf.iv(np.zeros(1), max(t, 1e-6))[0]))
    X = lsv.simulate_lsv(cal, [s.T for s in sl], n_paths=100_000, seed=2)
    z_all, err, err_heston = [], [], []
    for s in sl:
        target = lsv.black_otm(s.k, s.T, surf.iv(s.k, s.T))
        pr, se = lsv.otm_prices(X[s.T], s.k)
        iv, _ = lsv.otm_iv(pr, s.k, s.T, se)
        z_all.append((pr - target) / se)
        err.append((iv - surf.iv(s.k, s.T)) * 100)
        err_heston.append((hs.heston_iv(1.0, np.exp(s.k), s.T, 1.0, p) - surf.iv(s.k, s.T)) * 100)
    z_all, err, err_heston = np.concatenate(z_all), np.concatenate(err), np.concatenate(err_heston)
    assert np.mean(np.abs(z_all) <= 3) >= 0.95
    assert np.sqrt(np.mean(err ** 2)) < 0.25            # vol points
    # the leverage does real work: the variance process alone misses the surface by far more
    assert np.sqrt(np.mean(err_heston ** 2)) > 5 * np.sqrt(np.mean(err ** 2))


def test_lsv_heston_target_gives_unit_leverage():
    """For a Heston 'market' and the same Heston variance process the exact leverage is 1."""
    p = DEFAULT_PARAMS
    cal = lsv.calibrate_lsv(lambda x, t: lsv.heston_local_variance(p, x, t, n=1024), p, 0.5, nodes=[0.25, 0.5],
                            n_particles=40_000, dt=lsv.graded_dt(1 / 730, 1 / 365, 0.1), seed=3)
    d = cal.diagnostics
    late = d["t"] >= 0.1
    assert np.all(np.abs(d["L_p50"][late] - 1) < 0.03)
    assert np.all((d["L_p01"][late] > 0.85) & (d["L_p99"][late] < 1.2))       # tails: thinner regression
    X = lsv.simulate_lsv(cal, [0.25, 0.5], n_paths=100_000, seed=4)
    for T in (0.25, 0.5):
        k = np.linspace(-0.25, 0.12, 6) * np.sqrt(T / 0.25)
        pr, se = lsv.otm_prices(X[T], k)
        iv, se_iv = lsv.otm_iv(pr, k, T, se)
        exact = hs.heston_iv(1.0, np.exp(k), T, 1.0, p)
        assert np.max(np.abs(iv - exact) * 100) < 0.2   # vol points


# ------------------------------------------------------------------------------------------------ rough Bergomi
def test_rbergomi_covariance_matches_integrals():
    H, rho = 0.12, -0.6
    t = np.array([0.02, 0.1, 0.3])
    C = rough.rbergomi_cov(t, H, rho)
    g = H - 0.5
    for i in range(3):
        for j in range(3):
            u = min(t[i], t[j])
            vv = quad(lambda s: 2 * H * (t[i] - s) ** g * (t[j] - s) ** g, 0, u, limit=200)[0]
            vz = rho * np.sqrt(2 * H) * quad(lambda s: (t[i] - s) ** g, 0, u, limit=200)[0]
            assert C[i, j] == pytest.approx(vv, rel=1e-6)
            assert C[i, 3 + j] == pytest.approx(vz, rel=1e-6, abs=1e-12)
            assert C[3 + i, 3 + j] == pytest.approx(u)


def test_variance_swap_and_forward_variance_curve():
    # flat smile: variance-swap variance = implied variance
    w = 0.2 ** 2 * 0.5
    assert rough.variance_swap_total_variance(lambda k: np.full_like(k, w)) == pytest.approx(w, rel=1e-6)
    # a negatively skewed arbitrage-free smile: variance swap above ATM implied variance
    sk = svi.ESSVISlice(theta=0.02, rho=-0.7, psi=0.1)
    assert rough.variance_swap_total_variance(sk.w) > sk.theta * 1.01
    xi0 = rough.forward_variance_curve([0.1, 0.3, 1.0], [0.004, 0.015, 0.05])
    t = np.linspace(0, 1, 100_001)
    total = np.cumsum(xi0(t[:-1]) * np.diff(t))
    assert total[np.searchsorted(t, 0.3) - 1] == pytest.approx(0.015, rel=1e-3)
    assert total[-1] == pytest.approx(0.05, rel=1e-3)


def test_rbergomi_without_vol_of_vol_is_black_scholes():
    """eta = 0: deterministic variance xi0(t), so every smile is flat at sqrt(int xi0 / T) (mixing estimator exact
    in law; only Monte Carlo noise in the conditional forwards remains)."""
    xi0 = rough.forward_variance_curve([0.1, 0.3], [0.1 * 0.04, 0.1 * 0.04 + 0.2 * 0.09])
    eng = rough.RBergomiMC([0.1, 0.3], xi0, n_paths=2 ** 13, steps_per_year=365, seed=0)
    k = np.linspace(-0.2, 0.1, 5)
    iv = eng.smile(0.1, 0.0, -0.8, {0.1: k, 0.3: k})
    assert np.max(np.abs(iv[0.1] - 0.2)) < 2e-3
    assert np.max(np.abs(iv[0.3] - np.sqrt((0.004 + 0.018) / 0.3))) < 2e-3


def test_rbergomi_coarse_scheme_on_refined_paths():
    """refine = 2, coarse = True must be the coarser scheme on every 2nd node of the same paths: with eta = 0 the
    variance is deterministic and piecewise constant on the coarse steps, so both resolutions give identical I and Q
    at the maturities."""
    xi0 = rough.forward_variance_curve([0.05, 0.2], [0.05 * 0.03, 0.05 * 0.03 + 0.15 * 0.05])
    eng = rough.RBergomiMC([0.05, 0.2], xi0, n_paths=256, steps_per_year=365, seed=1, refine=2)
    I_f, Q_f = eng.integrals(0.1, 0.0)
    I_c, Q_c = eng.integrals(0.1, 0.0, coarse=True)
    for T in (0.05, 0.2):
        jf, jc = eng._column(T, False), eng._column(T, True)
        np.testing.assert_allclose(I_c[:, jc], I_f[:, jf], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(Q_c[:, jc], Q_f[:, jf], rtol=1e-12)
    assert Q_c.shape[1] * 2 == Q_f.shape[1]
    # with vol of vol the two resolutions differ, but only by discretisation
    k = {0.05: np.array([-0.1, 0.0, 0.05])}
    a, b = eng.smile(0.1, 1.5, -0.7, k)[0.05], eng.smile(0.1, 1.5, -0.7, k, coarse=True)[0.05]
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.max(np.abs(a - b)) < 0.01


def _rbergomi_slices(params, Ts, xi0, k_span=0.25):
    """SYNTHETIC rough Bergomi smiles as Slice objects (forward 1), quoted with a 0.4 vol-point bid-ask."""
    eng = rough.RBergomiMC(Ts, xi0, n_paths=2 ** 15, steps_per_year=365, seed=99)
    out = []
    for T in Ts:
        k = np.linspace(-k_span * np.sqrt(T / 0.1), 0.4 * k_span * np.sqrt(T / 0.1), 15)
        iv = eng.smile(*params, {T: k})[T]
        K = np.exp(k)
        cp = np.where(k >= 0, 1.0, -1.0)
        px = black_price(cp, 1.0, K, T, iv, 1.0)
        out.append(Slice(root="SYN", expiration=pd.Timestamp("2024-07-01"), settlement="AM", T=T, F=1.0, D=1.0, K=K,
                         cp=cp, bid=px, ask=px, iv_bid=iv - 0.002, iv_mid=iv, iv_ask=iv + 0.002,
                         open_interest=np.ones_like(K), meta={"synthetic": True}))
    return out


def test_rbergomi_calibration_recovers_parameters():
    true = (0.1, 1.9, -0.85)
    Ts = [0.05, 0.12, 0.25]
    xi0 = rough.forward_variance_curve(Ts, [0.05 * 0.03, 0.12 * 0.035, 0.25 * 0.04])
    sl = _rbergomi_slices(true, Ts, xi0)
    fit, _ = rough.calibrate_rbergomi(sl, xi0, n_paths=2 ** 13, steps_per_year=365, n_grid=21, max_nfev=60,
                                      starts=[(0.25, 1.0, -0.6), (0.05, 2.5, -0.9)], n_polish=1)
    assert fit.H == pytest.approx(true[0], abs=0.03)
    assert fit.eta == pytest.approx(true[1], abs=0.3)
    assert fit.rho == pytest.approx(true[2], abs=0.05)
