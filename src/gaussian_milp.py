"""Gaussian control: r = x[A] + y[B] + z with z ~ N(0,1) instead of Poisson counting.

Any decoder with worst case error <= 1% must separate every true pair mean m_t from every false pair
mean m_f by at least d = 2 z_{0.99} = 4.6527 (two point test between unit variance Gaussians).
We solve  min max(x) + max(y)  subject to |m_t - m_f| >= d for all true/false pairs, exactly, as a MILP
(one binary per pair picks which of m_t, m_f is the higher one), with min(x) = 0 as normalisation.
Expected optimum: 4d, 6d, 8d for K = 3, 4, 5 (the single threshold design, 2(K-1) d).
Also evaluates the peeled band chain under the same model: levels v_{k+1} = 2 v_k + d, budget 2 v_{K-1}.
For K = 4 that is 14 d, much worse than the threshold design 6 d, unlike the Poisson case.

Usage: python3 src/gaussian_milp.py [Kmax]     (default 5; K=5 takes about a minute)
"""
import sys, time
import numpy as np
from scipy.stats import norm
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

D = 2 * norm.ppf(0.99)


def solve(K, d=D, time_limit=300):
    T = [(a, b) for a in range(K) for b in range(K) if a > b]
    F = [(a, b) for a in range(K) for b in range(K) if a <= b]
    pairs = [(t, f) for t in T for f in F]
    # variables: x[0..K-1], y[0..K-1], X, Y, w[0..K-1] (which x is zero), o[pairs]
    ix, iy, iX, iY, iw = 0, K, 2 * K, 2 * K + 1, 2 * K + 2
    io = iw + K
    nv = io + len(pairs)
    M = 2 * (2 * (K - 1) + 4) * d             # safely above any optimal-ish mean difference
    rows = []; lo = []; hi = []
    def add(coefs, l, u):
        rows.append(coefs); lo.append(l); hi.append(u)
    for a in range(K):
        add({ix + a: 1, iX: -1}, -np.inf, 0)
        add({iy + a: 1, iY: -1}, -np.inf, 0)
        add({ix + a: 1, iw + a: M}, -np.inf, M)       # w_a = 1 forces x_a = 0
    add({iw + a: 1 for a in range(K)}, 1, 1)
    for p, ((ta, tb), (fa, fb)) in enumerate(pairs):
        mt = {ix + ta: 1, iy + tb: 1}
        mf = {ix + fa: 1, iy + fb: 1}
        # o = 1: false higher:  m_f - m_t >= d - M (1 - o)
        c = {}
        for k, v in mf.items(): c[k] = c.get(k, 0) + v
        for k, v in mt.items(): c[k] = c.get(k, 0) - v
        c[io + p] = -M
        add(c, d - M, np.inf)
        # o = 0: true higher:  m_t - m_f >= d - M o
        c = {}
        for k, v in mt.items(): c[k] = c.get(k, 0) + v
        for k, v in mf.items(): c[k] = c.get(k, 0) - v
        c[io + p] = M
        add(c, d, np.inf)
    A = lil_matrix((len(rows), nv))
    for r, c in enumerate(rows):
        for k, v in c.items():
            A[r, k] = v
    cost = np.zeros(nv); cost[iX] = 1; cost[iY] = 1
    integ = np.zeros(nv); integ[iw:] = 1
    ub = np.full(nv, np.inf); ub[iw:] = 1
    t0 = time.time()
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), ub), options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    return res, time.time() - t0, res.x[ix:ix + K], res.x[iy:iy + K]


def pair_gap_ok(x, y, K, d=D, tol=1e-7):
    ts = [x[a] + y[b] for a in range(K) for b in range(K) if a > b]
    fs = [x[a] + y[b] for a in range(K) for b in range(K) if a <= b]
    return min(abs(t - f) for t in ts for f in fs) >= d - tol


def band_chain(K, d=D):
    from banded import build
    v, x, y, lx, ly = build(K, hfun=lambda mu: mu + d)
    return x, y, x.max() + y.max()


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"d = 2 z_0.99 = {D:.4f}")
    print(" K   MILP optimum   in units of d   expected 2(K-1)d   status      time(s)   band chain (units of d)")
    for K in range(3, kmax + 1):
        res, t, x, y = solve(K)
        P = res.fun
        xb, yb, Pb = band_chain(K)
        ok = pair_gap_ok(xb, yb, K)
        print(f"{K:2d}  {P:12.4f}  {P / D:12.4f}   {2 * (K - 1):8d}        {("optimal" if res.status == 0 else "status " + str(res.status)):10s} {t:6.1f}   {Pb / D:8.3f}  (gap condition holds: {ok})", flush=True)
        if K == 4:
            print("      K=4 optimal x =", np.round(x, 3).tolist(), " y =", np.round(y, 3).tolist())
