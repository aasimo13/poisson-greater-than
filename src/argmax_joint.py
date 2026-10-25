"""Joint argmax with three transmitters and one counter (K = 3 levels each).

Model: transmitters X, Y, Z pick levels a, b, c in {0,1,2}; the counter sees N ~ Poisson(x[a] + y[b] + z[c])
and must output argmax(a, b, c) (ties: lowest index, as np.argmax). Worst case error <= 1% over the 27
triples, any randomized decoder (multiclass LP). Cost = max(x) + max(y) + max(z).

Default (verify mode):
  1. take the stored best design (found by the ladder start search below),
  2. solve the multiclass LP, recheck the LP decoder's worst case error with mpmath (30 digits),
  3. compute the tournament baseline: two chained pairwise comparisons from banded.py at eps = 0.005 each
     (union bound to 1%), cost 2 * budget_K3(0.005),
  4. write results/argmax_k3.json (or the path in ARGMAX_OUT).
--search  rerun the ladder start Nelder-Mead (slow, about 10 minutes) and print the design it reaches.
          The ladder puts the 9 sources on the h-chain v_0 = 0, v_{k+1} = h(2 v_k), brightest level of
          the largest value first, then Nelder-Mead on the cost with a penalty above 1% error.

Usage (repo root): python3 src/argmax_joint.py [--search]
"""
import sys, json, os, time, itertools
import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson
from scipy.optimize import linprog, minimize
import mpmath as mp
import core
from core import nmax_for
import banded

K = 3
TARGET = 0.01
BEST = {"x": [4.966888033527838, 115.58239875122808, 1012.671062102421],
        "y": [1.1347223864975444, 49.64817930426333, 507.84128226914265],
        "z": [0.0012632485391160916, 17.259819147484002, 247.60046761785355]}


def labels3(K):
    return np.array([int(np.argmax(t)) for t in itertools.product(range(K), repeat=3)])


def means3(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    return (x[:, None, None] + y[None, :, None] + z[None, None, :]).ravel()


def cost3(x, y, z):
    return max(x) + max(y) + max(z)


def lp_multi(means, lab, M=3, nmax=None, ret=False):
    """min over randomized d[n, m] of the worst case error; mass beyond the grid counts as error."""
    means = np.asarray(means, float)
    if nmax is None:
        nmax = nmax_for(means.max())
    n = np.arange(nmax)
    P = poisson.pmf(n[None, :], means[:, None])
    H = len(means)
    nv = nmax * M + 1
    rows = []; cols = []; vals = []
    for h in range(H):
        rows += [h] * nmax; cols += list(range(lab[h], nmax * M, M)); vals += list(-P[h])
        rows.append(h); cols.append(nv - 1); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(H, nv))
    b = -np.ones(H)
    Aeq = sp.csr_matrix((np.ones(nmax * M), (np.repeat(np.arange(nmax), M), np.arange(nmax * M))), shape=(nmax, nv))
    c = np.zeros(nv); c[-1] = 1
    for method in ("highs", "highs-ipm"):
        r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=np.ones(nmax), bounds=[(0, 1)] * nv, method=method)
        if r.success:
            break
    if not r.success:
        raise RuntimeError(r.message)
    return (r.x[-1], r.x[:-1].reshape(nmax, M)) if ret else r.x[-1]


def min_scale(shape, K, target=TARGET, rel=1e-4):
    x, y, z = shape
    base = means3(x, y, z); lab = labels3(K); c = cost3(x, y, z)
    f = lambda s: lp_multi(s * base, lab)
    lo, hi = 1e-3, 100.0 / c
    while f(hi) > target:
        hi *= 1.6
        if hi * c > 30000:
            return float("inf"), hi
    while hi - lo > rel * hi:
        m = 0.5 * (lo + hi)
        if f(m) <= target: hi = m
        else: lo = m
    return hi * c, hi


def mp_error(means, lab, d):
    """Worst case error of decoder d (float table), pmfs in 30 digit mpmath; tail beyond the grid is error."""
    mp.mp.dps = 30
    nmax = d.shape[0]
    worst = mp.mpf(0)
    for h, m in enumerate(means):
        mu = mp.mpf(float(m)); acc = mp.mpf(0); pm = mp.e ** (-mu)
        for n in range(nmax):
            if n > 0:
                pm = pm * mu / n
            acc += pm * mp.mpf(float(d[n, lab[h]]))
        worst = max(worst, 1 - acc)
    return worst


def tournament_baseline(K=3, eps_each=0.005):
    """Two pairwise banded comparisons (max of 3 = max(max(A,B),C)), each at eps_each."""
    v, x, y, lx, ly = banded.build(K)
    lab = core.labels(K); base = core.pair_means(x, y); cost = x.max() + y.max()
    s = core.scale_to_target(lambda s: core.lp_minimax(s * base, lab), 1e-6, 400.0 / cost, target=eps_each)
    return s * cost, 2 * s * cost


def verify():
    lab = labels3(K)
    x, y, z = (np.array(BEST[k]) for k in "xyz")
    print("stored best design (ladder start)")
    print("x", np.round(x, 3).tolist()); print("y", np.round(y, 3).tolist()); print("z", np.round(z, 3).tolist())
    cost = cost3(x, y, z)
    print("cost", cost)
    means = means3(x, y, z)
    err, d = lp_multi(means, lab, ret=True)
    print("LP err", err)
    mperr = mp_error(means, lab, d)
    print("mpmath worst-case error of LP decoder (30 digits):", mp.nstr(mperr, 20))
    per, tot = tournament_baseline()
    print(f"tournament baseline: one comparison at eps 0.005 = {per:.2f}, two comparisons = {tot:.2f}")
    print(f"joint / tournament = {cost / tot:.2f}  (joint is more expensive)")
    out = {"K": K, "target": TARGET, "design": BEST, "cost": cost, "lp_error": err,
           "mpmath_error": float(mperr), "tournament_baseline": tot, "tournament_per_comparison": per}
    json.dump(out, open(os.environ.get("ARGMAX_OUT", "results/argmax_k3.json"), "w"), indent=1)


def search(seed=0):
    lab = labels3(K); L = 3 * K
    v = [0.0]
    for _ in range(L - 1):
        v.append(core.h(2 * v[-1]))
    lex = [(T, val) for val in range(K - 1, -1, -1) for T in range(3)]
    arr = np.zeros((3, K))
    for lvl, (T, val) in enumerate(lex):
        arr[T, val] = v[L - 1 - lvl]
    p0 = arr.ravel()
    split = lambda p: (np.abs(p[:K]), np.abs(p[K:2 * K]), np.abs(p[2 * K:]))

    def obj(p):
        x, y, z = split(p); e = lp_multi(means3(x, y, z), lab)
        return cost3(x, y, z) + (500 + 2e4 * (e - TARGET) if e > TARGET else 0)

    c, s = min_scale(split(p0), K, rel=1e-2)
    print("scaled ladder start cost", round(c, 1), flush=True)
    p = p0 * s; t = time.time()
    for rep in range(3):
        p = minimize(obj, p, method="Nelder-Mead",
                     options=dict(maxiter=500 * K, xatol=1e-3, fatol=1e-3, adaptive=True)).x
    x, y, z = split(p)
    print("search result cost", round(cost3(x, y, z), 2), "err", lp_multi(means3(x, y, z), lab),
          round(time.time() - t), "s")
    print("x", x.tolist()); print("y", y.tolist()); print("z", z.tolist())


if __name__ == "__main__":
    if "--search" in sys.argv:
        search()
    else:
        verify()
