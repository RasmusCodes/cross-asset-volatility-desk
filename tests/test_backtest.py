"""The daily backtest engine on a SYNTHETIC multi-day chain: accounting identities, rolls, hedging."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voldesk.backtest.data import with_otm_iv
from voldesk.backtest.engine import Backtester, Leg, Strategy
from voldesk.data import calendar, chains
from voldesk.data.synthetic import synthetic_chain, synthetic_curve
from voldesk.strategies.costs import BASE


@pytest.fixture(scope="module")
def synth_frames():
    days = pd.bdate_range("2024-05-01", "2024-08-20")
    expiries = [calendar.third_friday(2024, m) for m in range(5, 13)]
    rng = np.random.default_rng(7)
    S = 5000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    parts, fws = [], []
    for d, s in zip(days, S):
        raw = synthetic_chain(str(d.date()), spot=float(s), expiries=expiries, am_settlement=1, strike_step=25.0,
                              seed=int(d.strftime("%j")))
        base = chains.base_frame(raw, pd.Series({d: float(s)}), synthetic_curve())
        fw = chains.parity_forwards(base)
        out = chains.add_forwards_and_iv(base, fw)
        parts.append(out)
        fws.append(fw)
    ch = pd.concat(parts, ignore_index=True)
    fw = pd.concat(fws, ignore_index=True)
    ch = with_otm_iv(ch)
    return days, ch, fw


def run(strats, synth_frames):
    days, ch, fw = synth_frames
    bt = Backtester(strats, BASE)
    bt.run([(2024, ch, fw[["date", "exdate", "am", "F"]])], days)
    return bt.results()


def test_accounting_identities(synth_frames):
    res = run([Strategy("ss", (Leg(1, -1), Leg(-1, -1))), Strategy("ssdh", (Leg(1, -1), Leg(-1, -1)), hedge="daily")],
              synth_frames)
    for name, r in res.items():
        d, t = r["daily"], r["trades"]
        assert len(t) >= 2, name                                     # at least one roll
        terms = d[["delta", "gamma", "vega", "vanna", "volga", "theta", "rates", "residual"]].sum(axis=1)
        assert np.allclose(terms, d["mtm"], atol=1e-6)               # attribution adds up to the mid change
        assert (d["costs"] >= 0).all()
        assert np.allclose(d["pnl"], d["mtm"] + d["hedge"] - d["costs"])
        closed = d.loc[: t["exit_date"].max(), "pnl"].sum()
        assert closed == pytest.approx(t["pnl"].sum(), rel=1e-6, abs=1e-3)  # trade P&L = sum of daily P&L to the exits
    dh = res["ssdh"]["daily"]
    held = dh["n_legs"] > 0
    assert held.mean() > 0.9
    assert dh.loc[held, "delta_net"].abs().max() < 1e-6             # hedged to zero delta at every close
    exits = pd.to_datetime(res["ss"]["trades"]["exit_date"])
    regular = exits[exits < synth_frames[0][-1]]                     # the last position is closed at the sample end
    assert len(regular) >= 2 and all(e.dayofweek == 3 for e in regular)   # AM-settled: out on the Thursday before expiry


def test_costs_scale_with_spread_fraction(synth_frames):
    from voldesk.backtest.metrics import with_cost_model
    from voldesk.strategies.costs import FRICTIONLESS, STRESS
    d = run([Strategy("ic", (Leg(-1, -1, delta=0.25), Leg(-1, 1, delta=0.10), Leg(1, -1, delta=0.25),
                             Leg(1, 1, delta=0.10)))], synth_frames)["ic"]["daily"]
    c0 = with_cost_model(d, FRICTIONLESS)["costs"].sum()
    c1 = with_cost_model(d, BASE)["costs"].sum()
    c2 = with_cost_model(d, STRESS)["costs"].sum()
    assert c0 == 0 and 0 < c1 < c2
    assert c1 == pytest.approx(d["costs"].sum())
