"""Data layer: calendar, curves, chain cleaning, universe construction and the pull catalog (no WRDS access needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voldesk.data import calendar, catalog, chains, universe
from voldesk.data.rates import ZeroCurve
from voldesk.data.synthetic import synthetic_chain, synthetic_curve
from voldesk.pricing import IVStatus


def test_settlement_instants():
    # pre-2015 OptionMetrics standard expiry on a Saturday settles at Friday's opening print
    am = calendar.settlement_instant(["2010-06-19"], [1]).iloc[0]
    assert am == pd.Timestamp("2010-06-18 09:30")
    pm = calendar.settlement_instant(["2024-06-21"], [0]).iloc[0]
    assert pm == pd.Timestamp("2024-06-21 16:00")
    T = calendar.time_to_expiry(["2024-06-20"], ["2024-06-21"], [1])[0]
    assert T == pytest.approx(17.5 / 24 / 365)
    assert calendar.third_friday(2024, 6) == pd.Timestamp("2024-06-21")


def test_last_trading_day():
    days = pd.bdate_range("2010-06-01", "2010-06-30")
    ltd = calendar.last_trading_day(["2010-06-19", "2010-06-19"], [1, 0], days)
    assert ltd.iloc[0] == pd.Timestamp("2010-06-17")       # AM: the day before settlement
    assert ltd.iloc[1] == pd.Timestamp("2010-06-18")       # PM/American: the settlement day itself


def test_zero_curve_interpolation():
    df = pd.DataFrame({"date": ["2024-01-02"] * 3, "days": [30, 365, 730], "rate": [5.0, 4.0, 3.5]})
    zc = ZeroCurve(df)
    T = np.array([0.01, 30 / 365, 0.5, 365 / 365, 3.0])
    r = zc.rate("2024-01-02", T)
    assert r[0] == pytest.approx(0.05) and r[-1] == pytest.approx(0.035)       # flat outside the nodes
    # linear in r T between nodes
    t1, t2 = 30 / 365, 1.0
    lam = (0.5 - t1) / (t2 - t1)
    assert r[2] * 0.5 == pytest.approx((1 - lam) * 0.05 * t1 + lam * 0.04 * t2)
    assert np.isnan(zc.rate("2024-02-01", 1.0)[0])                               # > 5 days stale


def test_clean_chain_flags_and_iv():
    raw = synthetic_chain("2024-06-03", spot=5000.0, seed=1)
    raw.loc[raw.index[0], ["best_bid", "best_offer"]] = [10.0, 9.0]               # crossed quote
    base = chains.base_frame(raw, pd.Series({pd.Timestamp("2024-06-03"): 5000.0}), synthetic_curve())
    out = chains.add_forwards_and_iv(base, chains.parity_forwards(base))
    assert out["f_crossed"].sum() >= 1
    otm = chains.otm_quotes(out)
    assert len(otm) > 100 and (otm["iv_status"] == IVStatus.OK).all()
    assert ((otm["cp"] == 1) == (otm["K"] >= otm["F"])).all()


def test_universe_is_point_in_time():
    msp = pd.DataFrame({"permno": [1, 2, 3], "start": ["1990-01-01", "1990-01-01", "1999-06-01"],
                        "ending": ["2030-01-01", "1999-03-31", "2030-01-01"]})
    msf = pd.DataFrame({"permno": [1, 2, 3, 1, 2, 3], "date": ["1999-01-29"] * 3 + ["1999-12-31"] * 3,
                        "prc": [10, 50, 30, 10, -50, 30], "shrout": [100] * 6})
    link = pd.DataFrame({"secid": [11, 22, 33], "sdate": ["1990-01-01"] * 3, "edate": ["2030-01-01"] * 3,
                         "permno": [1, 2, 3], "score": [1, 1, 1]})
    u = universe.top_members(msp, msf, link, n=2, start="1999-01-31", end="1999-12-31")
    jan = u[u["month_end"] == "1999-01-31"]["permno"].tolist()
    dec = u[u["month_end"] == "1999-12-31"]["permno"].tolist()
    assert jan == [2, 1]          # permno 2 (later removed from the index) is the largest member in January
    assert dec == [3, 1]          # by December it has left; the new member 3 is in


def test_catalog_specs_are_well_formed():
    specs = (catalog.reference_specs() + catalog.panel_specs([101594, 107525], years=[2024]) +
             catalog.spx_chain_specs(years=[2024]) + catalog.etf_chain_specs(years=[2024]) + catalog.cboe_specs())
    rels = [s.rel for s in specs]
    assert len(rels) == len(set(rels))
    for s in specs:
        assert s.sql.lower().startswith("select") and s.schema
        assert ";" not in s.sql                                    # one statement per pull
