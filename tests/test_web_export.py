"""The demo bundle must contain fitted parameters and statistics only, never vendor quote arrays."""
from __future__ import annotations

import json

import numpy as np

from voldesk.web.export import scenario_table, surface_bundle

FORBIDDEN = {"bid", "ask", "iv_bid", "iv_ask", "iv_mid", "mid", "K", "quotes"}


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_surface_bundle_has_no_quote_arrays():
    rec = {"date": "2024-06-03", "spot": 5000.0, "rows": 10,
           "slices": [{"T": 0.1, "expiry": "2024-07-19", "F": 5010.0, "D": 0.995, "n": 40, "k_min": -0.3,
                       "k_max": 0.1, "atm_iv": 0.13, "spread_med": 0.004, "fwd_source": "parity"}],
           "models": {"svi": {"params": {"slices": [{"T": 0.1, "expiry": "2024-07-19", "a": 0.001, "b": 0.05,
                                                      "rho": -0.6, "m": 0.02, "s": 0.1}]},
                              "slices": [{"T": 0.1, "expiry": "2024-07-19", "n": 40, "rmse": 0.2, "mae": 0.1,
                                          "inside": 0.9, "max_abs": 0.5}],
                              "seconds": 1.0, "success": True, "arbitrage": {"min_g_quoted": 0.1}}}}
    b = surface_bundle(rec, "SPX", "test")
    json.dumps(b)                                       # serialisable
    assert not (set(_keys(b)) & FORBIDDEN)
    assert b["models"]["svi"]["slices"][0]["b"] == 0.05


def test_scenarios_include_hypothetical_labels():
    import pandas as pd
    idx = pd.to_datetime(["2008-09-12", "2008-10-10"])
    tab = scenario_table(pd.Series([1250.0, 900.0], index=idx), pd.DataFrame(index=idx), pd.DataFrame(index=idx))
    kinds = {t["kind"] for t in tab}
    assert kinds == {"historical", "hypothetical"}
    lehman = [t for t in tab if t["id"] == "lehman"][0]
    assert np.isclose(lehman["spot_return"], 900 / 1250 - 1)
