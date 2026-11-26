"""Dynamic hedging on simulated paths: delta-neutral trading and gamma scalping with time- or band-based rebalancing
and proportional costs, and dispersion (index options against component options) on correlated paths.

Paths follow exact geometric-Brownian-motion steps. Everything is vectorised across paths.

hedge() / straddle_hedge(): buy (qty > 0) or sell qty options (or straddles) at the Black-Scholes price at
sigma_price, then hold -qty x delta(sigma_hedge) units of the underlying, rebalanced every `every` steps (time rule)
or whenever the target differs from the holding by more than `band` units per option (band rule). Cash earns r, the
underlying pays a dividend yield q, each trade costs `cost` x |units| x price, and the hedge is unwound at expiry
against the payoff. P&L is in time-0 money, with its parts: option (payoff minus premium, both at expiry value),
hedge, and costs (the fixed entry and exit trades, and the rebalancing trades in between).

Theory this reproduces (tests/test_strategies.py):
  * hedging at the true vol, the P&L has mean ~0 and a standard deviation that falls like 1/sqrt(rebalances);
  * when realized vol s_r differs from implied s_i, hedging at s_r locks in V(s_r) - V(s_i) on every path, whatever
    the drift; hedging at s_i gives the same mean when the drift equals r, with a path-dependent spread (the
    gamma-weighted sum of realized minus implied variance);
  * rebalancing costs grow like sqrt(rebalances), so more frequent hedging trades risk for cost.

Dispersion: short an index straddle priced at the index vol implied by a correlation rho_impl, long straddles on the
components in index proportions, each delta-hedged every step at its implied vol. With component vols realized as
implied, the P&L is a bet that realized correlation comes in below rho_impl.
"""
from __future__ import annotations

import numpy as np

from ..pricing.bsm import bsm_price
from ..pricing.greeks import bsm_greeks


def gbm_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int = 0, q: float = 0.0):
    """(t, S) with S of shape (n_paths, n_steps + 1): exact log-normal steps with drift mu and dividend yield q."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    lr = (mu - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(lr, axis=1)], axis=1))
    return np.linspace(0.0, T, n_steps + 1), S


def _delta(cp, S, K, tau, r, sigma, q):
    return bsm_greeks(cp, S, K, tau, r, sigma, q)["delta"]


def _run(S, t, V0, delta_fn, payoff_fn, r, q, qty, every, band, cost) -> dict:
    n_paths, n1 = S.shape
    N = n1 - 1
    T = float(t[-1])
    dt = T / N
    grow, div = np.exp(r * dt), np.expm1(q * dt)
    h = -qty * delta_fn(S[:, 0], T)
    fixed = cost * np.abs(h) * S[:, 0]
    rebal = np.zeros(n_paths)
    cash = -qty * V0 - h * S[:, 0] - fixed
    trades = np.ones(n_paths)
    for i in range(1, N):
        cash = cash * grow + h * S[:, i] * div
        target = -qty * delta_fn(S[:, i], T - t[i])
        do = np.full(n_paths, i % every == 0) if band is None else np.abs(target - h) > band * abs(qty)
        trade = np.where(do, target - h, 0.0)
        c = cost * np.abs(trade) * S[:, i]
        cash -= trade * S[:, i] + c
        fixed, rebal = fixed * grow, rebal * grow + c
        trades += do
        h = h + trade
    cash = cash * grow + h * S[:, N] * div
    pay = qty * payoff_fn(S[:, N])
    c = cost * np.abs(h) * S[:, N]
    cash += pay + h * S[:, N] - c
    fixed, rebal = fixed * grow + c, rebal * grow
    disc = np.exp(-r * T)
    option = (pay - qty * V0 / disc) * disc
    pnl = cash * disc
    costs = (fixed + rebal) * disc
    return {"pnl": pnl, "option": option, "costs": costs, "rebalance_costs": rebal * disc,
            "hedge": pnl - option + costs, "trades": trades, "premium": qty * V0}


def hedge(S: np.ndarray, t: np.ndarray, cp: int, K: float, sigma_price: float, sigma_hedge: float, r: float = 0.0,
          q: float = 0.0, qty: float = 1.0, every: int = 1, band: float | None = None, cost: float = 0.0) -> dict:
    T = float(t[-1])
    V0 = float(bsm_price(cp, S[0, 0], K, T, r, sigma_price, q))
    return _run(S, t, V0, lambda s, tau: _delta(cp, s, K, tau, r, sigma_hedge, q),
                lambda s: np.maximum(cp * (s - K), 0.0), r, q, qty, every, band, cost)


def straddle_hedge(S: np.ndarray, t: np.ndarray, K: float, sigma_price: float, sigma_hedge: float, r: float = 0.0,
                   q: float = 0.0, qty: float = 1.0, every: int = 1, band: float | None = None, cost: float = 0.0) -> dict:
    """A straddle hedged as one position (the call's and the put's deltas net before trading): gamma scalping."""
    T = float(t[-1])
    V0 = float(bsm_price(1, S[0, 0], K, T, r, sigma_price, q) + bsm_price(-1, S[0, 0], K, T, r, sigma_price, q))
    return _run(S, t, V0, lambda s, tau: _delta(1, s, K, tau, r, sigma_hedge, q) + _delta(-1, s, K, tau, r, sigma_hedge, q),
                lambda s: np.abs(s - K), r, q, qty, every, band, cost)


# ---------------------------------------------------------------------- dispersion
def correlated_gbm(S0, sigma, corr, mu: float, T: float, n_steps: int, n_paths: int, seed: int = 0):
    """(t, S) with S of shape (n_paths, n_steps + 1, n_assets)."""
    S0, sigma = np.asarray(S0, dtype=float), np.asarray(sigma, dtype=float)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    z = rng.standard_normal((n_paths, n_steps, S0.size)) @ L.T
    lr = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1, S0.size)), np.cumsum(lr, axis=1)], axis=1))
    return np.linspace(0.0, T, n_steps + 1), S


def equicorrelation(n: int, rho: float) -> np.ndarray:
    return np.full((n, n), rho) + (1 - rho) * np.eye(n)


def index_vol(sigma, weights, rho: float) -> float:
    """Vol of a value-weighted basket with constant pairwise correlation (weights sum to 1)."""
    a = np.asarray(weights, dtype=float) * np.asarray(sigma, dtype=float)
    return float(np.sqrt(np.sum(a ** 2) + rho * (np.sum(a) ** 2 - np.sum(a ** 2))))


def implied_correlation(sigma_index: float, sigma, weights) -> float:
    """The constant pairwise correlation that reproduces the index vol from the component vols: the formula behind
    Cboe's implied correlation indices, rho = (s_I^2 - sum w^2 s^2) / ((sum w s)^2 - sum w^2 s^2)."""
    a = np.asarray(weights, dtype=float) * np.asarray(sigma, dtype=float)
    return float((sigma_index ** 2 - np.sum(a ** 2)) / (np.sum(a) ** 2 - np.sum(a ** 2)))


def dispersion(sigma, weights, rho_implied: float, rho_real: float, T: float = 30 / 365, n_steps: int = 21,
               n_paths: int = 4000, r: float = 0.0, seed: int = 0) -> dict:
    """Short one ATM index straddle (index = 100 at the start) and long ATM straddles on the components in index
    proportions, all delta-hedged every step at their implied vols; component vols realize as implied and pairwise
    correlation realizes at rho_real. P&L per path in index points."""
    sigma, w = np.asarray(sigma, dtype=float), np.asarray(weights, dtype=float)
    n = sigma.size
    S0 = np.full(n, 100.0)
    shares = w * 100.0 / S0
    t, S = correlated_gbm(S0, sigma, equicorrelation(n, rho_real), r, T, n_steps, n_paths, seed)
    idx = S @ shares
    s_index = index_vol(sigma, w, rho_implied)
    short_index = straddle_hedge(idx, t, 100.0, s_index, s_index, r=r, qty=-1.0)
    comp = sum(straddle_hedge(S[:, :, i], t, 100.0, sigma[i], sigma[i], r=r, qty=float(shares[i]))["pnl"] for i in range(n))
    return {"pnl": short_index["pnl"] + comp, "index_leg": short_index["pnl"], "component_legs": comp,
            "index_vol_implied": s_index, "index_vol_realized": index_vol(sigma, w, rho_real),
            "premium_index": -short_index["premium"]}
