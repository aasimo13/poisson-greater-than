"""Core tools for the Poisson GREATER-THAN problem (rebuilt 2026-10-24).

Model: A,B in {0..K-1}; N ~ Poisson(x[A] + y[B] + bg); decide A > B (ties false).
Decoder: any randomized function of N, found by LP (minimize worst case error).
"""
import numpy as np
from scipy.stats import poisson
from scipy.optimize import linprog, brentq

import os
TARGET = float(os.environ.get("EPS", "0.01"))


def labels(K):
    return np.array([1 if a > b else 0 for a in range(K) for b in range(K)])


def pair_means(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return (x[:, None] + y[None, :]).ravel()


def nmax_for(mmax):
    return int(mmax + 12 * np.sqrt(mmax + 1) + 60)


def pmf_matrix(means, nmax=None):
    means = np.asarray(means, float)
    if nmax is None:
        nmax = nmax_for(means.max())
    n = np.arange(nmax)
    P = poisson.pmf(n[None, :], means[:, None])
    tail = np.clip(1.0 - P.sum(axis=1), 0, None)
    return np.hstack([P, tail[:, None]])


def lp_minimax(means, lab, return_decoder=False, nmax=None):
    """Worst case error of the best randomized decoder over hypotheses (means, lab)."""
    P = pmf_matrix(means, nmax)
    H, M = P.shape
    lab = np.asarray(lab)
    # variables: d_0..d_{M-1} (prob of saying TRUE), t
    c = np.zeros(M + 1); c[-1] = 1.0
    A = np.zeros((H, M + 1)); b = np.zeros(H)
    T = lab == 1
    A[T, :M] = -P[T]; A[T, -1] = -1.0; b[T] = -1.0   # 1 - P d <= t
    A[~T, :M] = P[~T]; A[~T, -1] = -1.0; b[~T] = 0.0  # P d <= t
    bounds = [(0, 1)] * M + [(0, 1)]
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    if return_decoder:
        return res.x[-1], res.x[:M]
    return res.x[-1]


def e2(mu, nu):
    """Two point minimax error between Poisson(mu) and Poisson(nu) (randomized test)."""
    mu, nu = min(mu, nu), max(mu, nu)
    if nu <= mu:
        return 0.5
    nmax = nmax_for(nu)
    n = np.arange(nmax)
    pm = poisson.pmf(n, mu); pn = poisson.pmf(n, nu)
    Fm = np.cumsum(pm); Fn = np.cumsum(pn)
    Sm = 1 - Fm                       # P_mu(N > T)
    Fn_prev = np.concatenate([[0.0], Fn[:-1]])   # P_nu(N < T)
    a0 = Sm; b0 = Fn                  # gamma = 0: say nu iff N > T
    a1 = Sm + pm; b1 = Fn_prev        # gamma = 1: say nu iff N >= T
    ok = (a0 <= b0) & (a1 >= b1)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return 0.5
    T = idx[0]
    g = (b0[T] - a0[T]) / (pm[T] + pn[T])
    return a0[T] + g * pm[T]


def h(mu, target=TARGET):
    """Smallest nu > mu with e2(mu, nu) <= target."""
    lo = mu + 1e-9
    hi = mu + 10 * np.sqrt(mu + 1) + 30
    return brentq(lambda v: e2(mu, v) - target, lo, hi, xtol=1e-10, rtol=1e-12)


def scale_to_target(fn, s_lo, s_hi, target=TARGET, rel=1e-6):
    """Smallest s with fn(s) <= target, fn nonincreasing in s (bisection)."""
    while fn(s_hi) > target:
        s_hi *= 1.5
    while s_hi - s_lo > rel * s_hi:
        m = 0.5 * (s_lo + s_hi)
        if fn(m) <= target:
            s_hi = m
        else:
            s_lo = m
    return s_hi


def budget_for_shape(x, y, K=None, bg=0.0):
    """Min max(x)+max(y) (after scaling the shape) reaching TARGET with LP decoder."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    K = len(x)
    lab = labels(K)
    base = pair_means(x, y)
    cost = x.max() + y.max()
    s = scale_to_target(lambda s: lp_minimax(s * base + bg, lab), 1e-6, 400.0 / cost)
    return s * cost, s
