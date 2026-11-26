"""Forecasting: losses, the walk-forward engine (no look-ahead), regimes, correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voldesk.data.synthetic import synthetic_paths
from voldesk.forecast import correlation as cr
from voldesk.forecast import evaluate as ev
from voldesk.forecast import models as fm
from voldesk.forecast import regimes as rg
from voldesk.forecast.realized import forward_sum


def test_walk_forward_never_trains_on_unobserved_targets():
    dates = pd.bdate_range("2010-01-01", periods=400)
    seen = []

    def fp(train, pred):
        seen.append((train.stop, pred.start))
        return np.zeros(pred.stop - pred.start)
    h = 22
    ev.walk_forward(dates, dates[100], 21, h, fp)
    assert all(stop <= start - h + 1 for stop, start in seen)       # last training row's target ends by t_k


def test_qlike_prefers_the_true_variance():
    rng = np.random.default_rng(0)
    var = rng.uniform(1e-5, 4e-4, 20000)
    x = var * rng.chisquare(1, var.size)                             # noisy unbiased proxy
    assert ev.qlike(var, x).mean() < ev.qlike(var * 1.3, x).mean()
    assert ev.qlike(var, x).mean() < ev.qlike(var * 0.7, x).mean()


def test_garch_and_ewma_run_on_synthetic_returns():
    px = synthetic_paths(1500, seed=2)
    r = np.log(px["close"]).diff().dropna()
    fit = fm.fit_garch(r.to_numpy())
    assert 0.8 < fit.persistence < 1.0
    s = fm.garch_next(fit, r)
    assert (s.dropna() > 0).all()
    e = fm.ewma_next(r ** 2)
    assert np.isfinite(e.iloc[-1])
    y = forward_sum(r ** 2, 5)
    assert np.isnan(y.iloc[-1]) and np.isfinite(y.iloc[0])


def test_hmm_separates_regimes():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 0.6, 1500), rng.normal(0, 2.0, 400), rng.normal(0, 0.6, 1000)])
    m = rg.fit_hmm2(x)
    assert m.sigma[1] / m.sigma[0] > 2.5
    p = rg.filtered_high_prob(x, m)
    assert p[1600:1850].mean() > 0.8 and p[2200:].mean() < 0.2
    assert rg.auc(p, (np.arange(x.size) >= 1500) & (np.arange(x.size) < 1900)) > 0.9


def test_equicorrelation_two_assets():
    s1, s2, rho = 0.3, 0.2, 0.4
    w = np.array([0.5, 0.5])
    var_b = 0.25 * s1 ** 2 + 0.25 * s2 ** 2 + 2 * 0.25 * rho * s1 * s2
    assert cr.equicorr(var_b, w, np.array([s1 ** 2, s2 ** 2])) == pytest.approx(rho)


def test_logistic_and_scores():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3000, 1))
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(2 * x[:, 0] - 1)))).astype(float)
    p = rg.logistic_fit_predict(x[:2000], y[:2000], x[2000:])
    assert rg.auc(p, y[2000:]) > 0.75
    assert rg.brier(p, y[2000:]) < rg.brier(np.full(1000, y[:2000].mean()), y[2000:])
