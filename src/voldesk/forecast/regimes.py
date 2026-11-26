"""Volatility regimes: a two-state Gaussian hidden Markov model fitted by EM, filtered (real-time) state probabilities,
and probability-forecast scoring.

Model: daily returns x_t (percent) ~ N(mu_s, sigma_s^2) in state s in {calm, turbulent}, a first-order Markov chain
with transition matrix A. States are labelled by volatility (state 1 = higher sigma). Filtered probabilities
P(s_t | x_1..x_t) use only data up to t, so they can serve as real-time predictors; the EM fit itself is refitted on
expanding windows by the caller (walk-forward), never on future data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class HMM2:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    loglik: float
    n_iter: int

    @property
    def expected_durations(self) -> np.ndarray:
        return 1.0 / (1.0 - np.diag(self.A))


def _emissions(x, mu, sigma):
    return np.column_stack([norm.pdf(x, mu[0], sigma[0]), norm.pdf(x, mu[1], sigma[1])]) + 1e-300


def _forward(B, pi, A):
    T = B.shape[0]
    alpha = np.empty((T, 2))
    c = np.empty(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def fit_hmm2(x: np.ndarray, n_iter: int = 300, tol: float = 1e-7) -> HMM2:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = x.std()
    mu = np.array([x.mean(), x.mean()])
    sigma = np.array([0.7 * s, 1.8 * s])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.8, 0.2])
    prev = -np.inf
    for it in range(1, n_iter + 1):
        B = _emissions(x, mu, sigma)
        alpha, c = _forward(B, pi, A)
        T = len(x)
        beta = np.empty((T, 2))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        A /= A.sum(axis=1, keepdims=True)
        pi = gamma[0]
        w = gamma.sum(axis=0)
        mu = (gamma * x[:, None]).sum(axis=0) / w
        sigma = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w)
        sigma = np.maximum(sigma, 1e-4 * s)
        ll = float(np.log(c).sum())
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    o = np.argsort(sigma)
    return HMM2(pi[o], A[np.ix_(o, o)], mu[o], sigma[o], ll, it)


def filtered_high_prob(x: np.ndarray, m: HMM2) -> np.ndarray:
    """P(turbulent state at t | x_1..x_t) for every t (NaN inputs carry the previous filter forward)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    a = m.pi.copy()
    started = False
    for t, v in enumerate(x):
        if not np.isfinite(v):
            if started:
                a = a @ m.A
                out[t] = a[1]
            continue
        b = np.array([norm.pdf(v, m.mu[0], m.sigma[0]), norm.pdf(v, m.mu[1], m.sigma[1])]) + 1e-300
        a = (a @ m.A if started else a) * b
        a /= a.sum()
        started = True
        out[t] = a[1]
    return out


# ---------------------------------------------------------------- probability scoring
def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) formula; ties get average ranks."""
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_fit_predict(X_train, y_train, X_pred, l2: float = 1e-4):
    """Plain logistic regression (Newton-Raphson with a tiny ridge on the slopes)."""
    X = np.column_stack([np.ones(len(X_train)), np.asarray(X_train, float)])
    y = np.asarray(y_train, float)
    b = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    R[0, 0] = 0.0
    for _ in range(50):
        z = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + R @ b
        H = (X * (p * (1 - p))[:, None]).T @ X + R
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    Xp = np.column_stack([np.ones(len(X_pred)), np.asarray(X_pred, float)])
    return 1 / (1 + np.exp(-np.clip(Xp @ b, -30, 30)))
