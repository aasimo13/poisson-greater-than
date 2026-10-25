"""Background light for the K = 3 comparator, run as margin 2 on K = 4 (same problem,
see threshold_family.py). Every window gets b extra counts on top of x[A]+y[B]; cost is
the max signal mean (background not charged). Both the best single threshold design
(any encoder, both directions, randomized cutoff) and the banded design (exact LP decoder)
are re-optimized at each b by Nelder-Mead from four starting shapes. Search results, not proofs.

usage: python3 src/background_k3.py 0,0.1,0.3,0.5,1,2,3   (about 10 minutes per value)
"""
import sys, json
import numpy as np
from scipy.stats import poisson
from scipy.optimize import linprog, minimize
K = 4; EPS = 0.01
PAIRS = [(a, b) for a in range(K) for b in range(K)]
CLS = np.array([a + b >= 2 for a, b in PAIRS])

def pmf_table(lam):
    n = np.arange(int(lam.max() + 10 * np.sqrt(lam.max() + 1) + 25) + 1)
    return poisson.pmf(n[None, :], lam[:, None])

def lp_err(lam):
    P = pmf_table(lam); nv = P.shape[1] + 1; c = np.zeros(nv); c[-1] = 1
    A = np.hstack([np.where(CLS[:, None], -P, P), -np.ones((len(CLS), 1))])
    r = linprog(c, A_ub=A, b_ub=np.where(CLS, -1.0, 0.0), bounds=[(0, 1)] * nv, method="highs")
    return r.fun if r.success else 1.0

def thr_err(lam):
    P = pmf_table(lam); C = np.cumsum(P, 1)
    below = np.hstack([np.zeros((len(lam), 1)), C[:, :-1]]); above = 1 - C
    q = np.linspace(0, 1, 51)[:, None, None]; best = 1.0
    for side in (above, below):
        pt = side[None] + q * P[None]
        best = min(best, np.where(CLS[None, :, None], 1 - pt, pt).max(1).min())
    return best

def sig(s):
    s = np.abs(s); return np.array([s[a] + s[K + b] for a, b in PAIRS])

def need(shape, err, b):
    p = sig(shape)
    if p.max() < 1e-9: return 1e9
    p = p / p.max(); hi = 3000.0; e = err(hi * p + b)
    if e > EPS: return hi + 1e5 * e
    lo = 0.0
    while hi - lo > 0.05:
        m = (lo + hi) / 2; lo, hi = (lo, m) if err(m * p + b) <= EPS else (m, hi)
    return hi

STARTS = {"threshold": [np.array([0, 1, 2, 2.] * 2), np.array([2, 2, 1, 0.] * 2), np.array([0, 1, 2, 3.] * 2), np.array([3, 2, 1, 0.] * 2)],
          "banded": [np.array([4.6, 0, 29.3, 29.3] * 2), np.array([0, 1, 2, 2.] * 2), np.array([.29, 0, .047, 1] * 2), np.array([.047, 0, .29, 1] * 2)]}

if __name__ == "__main__":
    for b in [float(v) for v in sys.argv[1].split(",")]:
        out = {"background": b}
        for name, err in (("threshold", thr_err), ("banded", lp_err)):
            out[name] = round(min(minimize(need, x0, args=(err, b), method="Nelder-Mead",
                                           options={"maxiter": 900, "adaptive": True, "fatol": 0.02}).fun
                                  for x0 in STARTS[name]), 1)
        out["saving_pct"] = round(100 * (1 - out["banded"] / out["threshold"]), 1)
        print(json.dumps(out), flush=True)
