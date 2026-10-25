"""Price curve C(eps): photons (or molecules) needed for the A > B comparison (K = 4) at error target eps.

Part A of the sorting cost study. For eps in EPS it computes the worst case expected cost for
  banded     : one count, the banded design (shape fixed, scaled to hit eps, LP decoder)
  threshold  : one count, the threshold design (same)
  twochan    : two types counted separately, the stored 106.4 design scaled to hit eps (2D LP decoder)
  chain_LB   : certified lower bound for two types, 2 * L3 of the Neyman-Pearson chain (core.h at target eps)
Imports core.py from the same folder (run from the repo root). Writes results/price_curve.json
(or the path in PRICE_OUT). part B is src/tournament.py, which reads that file.

Usage: python3 src/price_curve.py        (a few minutes)
"""
import json, os
import numpy as np
from scipy.optimize import linprog
from core import labels, pair_means, pmf_matrix, lp_minimax, scale_to_target, h

EPS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3]


def shared(x, y, eps):
    x = np.asarray(x, float); y = np.asarray(y, float)
    lab = labels(4); base = pair_means(x, y); cost = x.max() + y.max()
    try:
        s = scale_to_target(lambda s: lp_minimax(s * base, lab), 1e-6, 400.0 / cost, target=eps, rel=1e-5)
    except Exception:
        return None
    return s * cost


def two_chan(eps, x0=[0, 4.701, 22.414, 52.805], y0=[0, 5.005, 23.512, 53.561]):
    x0 = np.array(x0); y0 = np.array(y0)
    lab = labels(4)

    def err(s):
        Px = pmf_matrix(s * x0); Py = pmf_matrix(s * y0)
        P = np.array([np.outer(Px[a], Py[b]).ravel() for a in range(4) for b in range(4)])
        H, M = P.shape
        c = np.zeros(M + 1); c[-1] = 1
        A = np.zeros((H, M + 1)); bb = np.zeros(H); T = lab == 1
        A[T, :M] = -P[T]; A[T, -1] = -1; bb[T] = -1
        A[~T, :M] = P[~T]; A[~T, -1] = -1
        r = linprog(c, A_ub=A, b_ub=bb, bounds=[(0, 1)] * (M + 1), method="highs")
        if not r.success:
            raise RuntimeError(r.message)
        return r.x[-1]

    s = scale_to_target(err, 1e-3, 1.0, target=eps, rel=1e-4)
    return s * (x0.max() + y0.max()), s


def chain(eps):
    L = [0.0]
    for _ in range(3):
        L.append(h(L[-1], target=eps))
    return L


if __name__ == "__main__":
    res = {}
    band = [98.2976, 4.6013, 0, 28.7852]; thr = [1.5, 1, 0.5, 0]
    for e in EPS:
        r = {"eps": e}
        r["banded"] = shared(band, band[::-1], e)
        r["threshold"] = shared(thr, thr[::-1], e)
        try:
            c, s = two_chan(e); r["twochan"] = c; r["twochan_scale"] = s
        except Exception as ex:
            r["twochan"] = None; r["err"] = str(ex)
        L = chain(e); r["chain_levels"] = L; r["chain_LB"] = 2 * L[3]
        res[e] = r
        print(json.dumps(r), flush=True)
    print("check ln((1-e)/e) vs h(0):", [(e, float(np.log((1 - e) / e)), h(0.0, target=e)) for e in EPS])
    json.dump(res, open(os.environ.get("PRICE_OUT", "results/price_curve.json"), "w"), indent=1)
