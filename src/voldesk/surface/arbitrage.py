"""Static-arbitrage diagnostics for option quotes and fitted surfaces.

Quotes (one expiry slice). Calls come straight from call quotes; puts are converted by put-call parity,
C = P + D (F - K), using the slice's parity forward and discount factor. For strikes K1 < K2 < K3:
    vertical       C(K1) >= C(K2) and C(K1) - C(K2) <= D (K2 - K1)   (call spread worth 0..D (K2 - K1))
    butterfly      lam C(K1) + (1 - lam) C(K3) - C(K2) >= 0, lam = (K3 - K2)/(K3 - K1)   (convexity)
A violation on mid prices is a data/quote inconsistency. It is "tradable" only if it survives the bid-ask:
buying the long legs at the ask and selling the short legs at the bid still locks in a profit.

Calendar on quotes: at common log-moneyness, total implied variance must not fall with maturity. Slices
are compared on the intersection of their strike ranges after linear interpolation of mid total variance.

Surfaces: minimum of the butterfly function g(k, T) and of dw/dT over a (k, T) grid.
"""
from __future__ import annotations

import numpy as np

from .svi import g_function


def slice_call_prices(s):
    """(K, C_bid, C_mid, C_ask) for a chain.Slice, calls from OTM quotes (puts via parity)."""
    adj = np.where(s.cp > 0, 0.0, s.D * (s.F - s.K))
    return s.K, s.bid + adj, s.mid + adj, s.ask + adj


def slice_quote_arbitrage(s, tol: float = 1e-9) -> dict:
    K, cb, cm, ca = slice_call_prices(s)
    o = np.argsort(K)
    K, cb, cm, ca = K[o], cb[o], cm[o], ca[o]
    dK = np.diff(K)
    # vertical spreads on consecutive strikes
    v_mid = (cm[1:] - cm[:-1] > tol) | (cm[:-1] - cm[1:] - s.D * dK > tol)
    v_trade = (cb[1:] - ca[:-1] > tol) | (cb[:-1] - ca[1:] - s.D * dK > tol)
    # butterflies on consecutive triples
    lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    b_mid_val = lam * cm[:-2] + (1 - lam) * cm[2:] - cm[1:-1]
    b_trade_val = lam * ca[:-2] + (1 - lam) * ca[2:] - cb[1:-1]
    scale = np.maximum(cm[1:-1], 1e-12)
    return {"n_strikes": int(K.size),
            "vertical_mid": int(v_mid.sum()), "vertical_tradable": int(v_trade.sum()),
            "butterfly_mid": int((b_mid_val < -tol).sum()), "butterfly_tradable": int((b_trade_val < -tol).sum()),
            "worst_butterfly_mid_rel": float(np.min(b_mid_val / scale)) if b_mid_val.size else 0.0}


def calendar_quote_violations(slices, n_grid: int = 41, tol: float = 0.0) -> list[dict]:
    """Consecutive-slice comparison of mid total variance on their common k range."""
    sl = sorted(slices, key=lambda x: x.T)
    out = []
    for a, b in zip(sl[:-1], sl[1:]):
        lo, hi = max(a.k.min(), b.k.min()), min(a.k.max(), b.k.max())
        if hi <= lo:
            continue
        kg = np.linspace(lo, hi, n_grid)
        wa = np.interp(kg, np.sort(a.k), a.w_mid[np.argsort(a.k)])
        wb = np.interp(kg, np.sort(b.k), b.w_mid[np.argsort(b.k)])
        viol = wa - wb > tol
        out.append({"short": a.label, "long": b.label, "T_short": a.T, "T_long": b.T,
                    "share_violating": float(viol.mean()), "worst_dw": float(np.max(wa - wb))})
    return out


def surface_static_arbitrage(surface, Ts, k_grid) -> dict:
    """Min butterfly function and min dw/dT of a surface over the grid; both must be >= 0."""
    min_g, min_dT = np.inf, np.inf
    worst_g_at, worst_dT_at = None, None
    for T in Ts:
        w = surface.w(k_grid, T)
        g = g_function(k_grid, w, surface.dw_dk(k_grid, T), surface.d2w_dk2(k_grid, T))
        dT = surface.dw_dT(k_grid, T)
        i, j = int(np.argmin(g)), int(np.argmin(dT))
        if g[i] < min_g:
            min_g, worst_g_at = float(g[i]), (float(k_grid[i]), float(T))
        if dT[j] < min_dT:
            min_dT, worst_dT_at = float(dT[j]), (float(k_grid[j]), float(T))
    return {"min_g": min_g, "min_g_at": worst_g_at, "min_dw_dT": min_dT, "min_dw_dT_at": worst_dT_at}
