"""Positions, attribution, costs, margin and hedging theory."""
from __future__ import annotations

import numpy as np
import pytest

from voldesk.strategies import builders as B
from voldesk.strategies import hedging_sim as H
from voldesk.strategies.book import OptionLeg, Position, attribute
from voldesk.strategies.costs import BASE, CostModel
from voldesk.strategies.margin import MarginLeg, naked_requirement, requirement
from voldesk.strategies.market import Market


def test_taylor_attribution_small_move():
    m0 = Market.flat(100.0, 0.20, 0.03, 0.01)
    pos = B.straddle(m0, 30, -1.0)
    m1 = m0.roll(1 / 365).move(0.005).shift_vol(0.002)
    a = attribute(pos, m0, m1)
    gross = sum(abs(a[k]) for k in ("delta", "gamma", "vega", "vanna", "volga", "theta", "rates"))
    assert abs(a["residual"]) < 0.02 * gross                # third and higher orders: ~1-2% of the gross terms
    assert a["theta"] > 0 and a["gamma"] < 0          # a short straddle earns theta, pays gamma


def test_cost_components_reconstruct_costs():
    cm = CostModel(spread_fraction=0.5, commission=1.0, hedge_bps=1.0)
    qty, bid, ask = np.array([-1.0, 2.0]), np.array([10.0, 3.0]), np.array([10.6, 3.4])
    sf, nc = cm.option_components(qty, bid, ask)
    assert cm.option_cost(qty, bid, ask) == pytest.approx(cm.from_components(sf, nc, 0.0))
    assert cm.trade_price(1, 10.0, 10.6) == pytest.approx(10.45)
    assert cm.trade_price(-1, 10.0, 10.6) == pytest.approx(10.15)


def test_margin_rules():
    S = 5000.0
    # naked short put 10% out of the money: premium + max(15% S - OTM, 10% K)
    req = naked_requirement(-1, 4500.0, S, 20.0)
    assert req == pytest.approx(20 + max(0.15 * S - 500, 0.10 * 4500))
    # put credit spread: margin = max loss = width - credit
    legs = [MarginLeg(-1, 4800, 0.1, -1, 100, 30.0), MarginLeg(-1, 4700, 0.1, 1, 100, 18.0)]
    assert requirement(legs, S)["requirement"] == pytest.approx((100 - 12) * 100)
    # short straddle: the larger side's naked requirement + the other side's premium
    legs = [MarginLeg(1, 5000, 0.1, -1, 100, 80.0), MarginLeg(-1, 5000, 0.1, -1, 100, 75.0)]
    call = naked_requirement(1, 5000, S, 80.0)
    put = naked_requirement(-1, 5000, S, 75.0)
    assert requirement(legs, S)["requirement"] == pytest.approx((max(call, put) + (75.0 if call >= put else 80.0)) * 100)
    # calendar (long later expiry at the same strike covers the short): net debit
    legs = [MarginLeg(1, 5000, 0.08, -1, 100, 60.0), MarginLeg(1, 5000, 0.16, 1, 100, 95.0)]
    assert requirement(legs, S)["requirement"] == pytest.approx(35.0 * 100)


def test_hedging_error_shrinks_with_rebalancing():
    t, S = H.gbm_paths(100.0, 0.0, 0.2, 0.25, 64, 4000, seed=1)
    sd = []
    for every in (8, 2):
        res = H.hedge(S, t, 1, 100.0, 0.2, 0.2, every=every)
        assert abs(res["pnl"].mean()) < 0.05
        sd.append(res["pnl"].std())
    assert sd[1] < sd[0] * 0.65                       # 4x more rebalances -> about half the error


def test_builders_on_flat_market():
    m = Market.flat(100.0, 0.2)
    ic = B.iron_condor(m, 30, 0.25, 0.10)
    assert len(ic.legs) == 4 and ic.value(m) < 0     # a credit structure
    g = Position("x", [OptionLeg(1, 100.0, 0.25, 1.0)]).greeks(m)
    assert 0 < g["delta"] < 100
