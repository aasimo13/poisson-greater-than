"""Arrival time comparator: A and B each fire a pulse in one of 4 time slots, a detector counts photons per slot.

Model: sender A (level a) puts mean muA[a] into slot a, sender B (level b) puts mean muB[b] into slot b, plus a
background beta per slot. The four slot counts are independent Poisson. Decide [A > B] (the later slot is A's).
Cost = max(muA) + max(muB) (signal only, background excluded), worst case error <= 1% over the 16 pairs,
any randomized decoder of the 4 slot counts (LP over the count vectors that matter; the truncated mass is
counted as error, so every number is conservative).

Modes (run from the repo root, writes/updates results/timecomp.json):
  python3 src/timecomp.py            verify the stored winner: LP error, cost, mpmath recheck
  python3 src/timecomp.py --beta     fixed shape rescale bisection at beta 0.1, 0.5, 1 (a few minutes to about 20)
  python3 src/timecomp.py --bound    lower bound from three pairwise Neyman-Pearson constraints (about 1 minute)
"""
import sys, os, json, time
import numpy as np
from scipy.stats import poisson
from scipy.optimize import linprog, minimize
import mpmath as mp

OUT = os.environ.get("TIMECOMP_OUT", "results/timecomp.json")
PAIRS = [(a, b) for a in range(4) for b in range(4)]
TRUE = np.array([1.0 if a > b else 0.0 for a, b in PAIRS])
# winner of the Nelder-Mead search at beta = 0 (cost 20.972)
WIN_A = [4.535158813102838, 4.786907722848276, 4.830716518368628, 4.833524524491183]
WIN_B = [16.04681139542948, 16.138659748846283, 16.136413046185083, 16.097174028760445]
# bisection brackets (lo, hi) per beta for the shape rescale
BRACKET = {0.1: (21.0, 40.0), 0.5: (25.0, 70.0), 1.0: (30.0, 90.0)}


def lam_matrix(muA, muB, beta):
    L = np.full((16, 4), float(beta))
    for k, (a, b) in enumerate(PAIRS):
        L[k, a] += muA[a]
        L[k, b] += muB[b]
    return L


def support(L, thr):
    """Count vectors whose max over pairs of the joint probability is >= thr (prefix pruning is a valid superset)."""
    pre = np.zeros((1, 0), dtype=np.int32)
    pp = np.ones((1, 16))
    for s in range(4):
        M = int(poisson.isf(1e-13, max(L[:, s].max(), 1e-9))) + 3
        n = np.arange(M + 1)
        pm = poisson.pmf(n[None, :], L[:, s][:, None])
        newp = pp[:, None, :] * pm.T[None, :, :]
        keep = newp.max(axis=2) >= thr
        i, j = np.nonzero(keep)
        pre = np.hstack([pre[i], j[:, None].astype(np.int32)])
        pp = newp[i, j]
    return pre, pp


def minimax_lp(L, thr=1e-9):
    pre, P = support(L, thr)
    N = P.shape[0]
    tail = np.maximum(0.0, 1.0 - P.sum(axis=0))
    A = np.zeros((16, N + 1)); b = np.zeros(16)
    for k in range(16):
        if TRUE[k]:
            A[k, :N] = -P[:, k]; b[k] = -(1.0 + tail[k])
        else:
            A[k, :N] = P[:, k]; b[k] = -tail[k]
        A[k, N] = -1.0
    c = np.zeros(N + 1); c[N] = 1
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * N + [(0, None)], method="highs")
    e = r.x[N] if r.x is not None else 1.0
    return e, r.x[:N] if r.x is not None else None, pre, N


def save(key, val):
    try:
        d = json.load(open(OUT))
    except Exception:
        d = {}
    d[key] = val
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(d, open(OUT, "w"), indent=1)


def verify():
    t0 = time.time()
    A, B = WIN_A, WIN_B
    cost = max(A) + max(B)
    L = lam_matrix(A, B, 0.0)
    e, d, pre, N = minimax_lp(L)
    print(f"winner muA = {np.round(A, 3).tolist()}  muB = {np.round(B, 3).tolist()}")
    print(f"cost {cost:.2f}   LP minimax error {e:.5f}   (count vectors {N}, {time.time() - t0:.0f} s)")
    mp.mp.dps = 30
    idx = np.nonzero(d > 1e-12)[0]
    worst = mp.mpf(0)
    for k, (a, b) in enumerate(PAIRS):
        lam = [mp.mpf(float(x)) for x in L[k]]
        s = mp.mpf(0)
        for i in idx:
            p = mp.mpf(1)
            for sl in range(4):
                n = int(pre[i, sl]); p *= mp.e ** (-lam[sl]) * lam[sl] ** n / mp.factorial(n)
            s += p * mp.mpf(float(d[i]))
        err = 1 - s if TRUE[k] else s
        worst = max(worst, err)
    print("mpmath 30 digit worst case error of the LP decoder:", mp.nstr(worst, 8))
    save("winner", {"muA": A, "muB": B, "cost": cost, "lp_error": float(e), "mpmath_error": float(worst),
                    "beta": 0.0, "source": "Nelder-Mead on shape, LP decoder, support threshold 1e-9"})


def beta_run(betas):
    R = json.load(open(OUT))["winner"] if os.path.exists(OUT) else None
    A0 = np.array(R["muA"] if R else WIN_A); B0 = np.array(R["muB"] if R else WIN_B)
    c0 = A0.max() + B0.max(); A0 = A0 / c0; B0 = B0 / c0
    for beta in betas:
        lo, hi = BRACKET[beta]
        while hi / lo > 1.01:
            m = (lo * hi) ** 0.5
            e, _, _, N = minimax_lp(lam_matrix(A0 * m, B0 * m, beta), 1e-7)
            ok = e <= 0.01
            print(f"beta {beta} C {m:.2f} err {e:.5f} N {N} ok {ok}", flush=True)
            if ok: hi = m
            else: lo = m
        print(f"RESULT beta {beta} cost <= {hi:.2f}", flush=True)
        try:
            d = json.load(open(OUT)).get("beta_costs", {})
        except Exception:
            d = {}
        d[str(beta)] = hi
        save("beta_costs", d)


def np2(l0, l1):
    """Exact minimax error between two 2 slot Poisson hypotheses (randomized Neyman-Pearson test)."""
    M = int(poisson.isf(1e-13, max(max(l0), max(l1), 1e-9))) + 6
    n = np.arange(M + 1)
    p0 = np.outer(poisson.pmf(n, max(l0[0], 1e-300)), poisson.pmf(n, max(l0[1], 1e-300))).ravel()
    p1 = np.outer(poisson.pmf(n, max(l1[0], 1e-300)), poisson.pmf(n, max(l1[1], 1e-300))).ravel()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lr = np.where(p0 > 0, p1 / p0, np.inf)
    o = np.argsort(-lr, kind="stable")
    c0 = np.concatenate([[0], np.cumsum(p0[o])]); c1 = np.concatenate([[0], np.cumsum(p1[o])])
    al = c0; be = 1 - c1 + max(0, 1 - c1[-1]); k = int(np.argmax(al >= be))
    if k == 0:
        return max(al[0], be[0])
    a0, b0, a1, b1 = al[k - 1], be[k - 1], al[k], be[k]
    t = (b0 - a0) / ((b0 - a0) + (a1 - b1))
    return a0 + t * (a1 - a0)


def bound():
    """Only slots 0,1 matter: variables A0,A1,B0,B1 against the three pairs (0,0), (0,1), (1,1) vs (1,0)."""
    def cons(v):
        A0, A1, B0, B1 = np.abs(v)
        return (np2([A0 + B0, 0.0], [B0, A1]),      # (0,0) FALSE vs (1,0) TRUE
                np2([A0, B1], [B0, A1]),            # (0,1) FALSE vs (1,0) TRUE
                np2([0.0, A1 + B1], [B0, A1]))      # (1,1) FALSE vs (1,0) TRUE

    def obj(v, pen=1e4):
        A0, A1, B0, B1 = np.abs(v); c = max(A1 + B0, A0 + B1, A0 + B0, A1 + B1)
        return c + pen * sum(max(0, e - 0.01) for e in cons(v))

    best = None; rng = np.random.default_rng(2)
    for s in range(12):
        r = minimize(obj, rng.uniform(1, 10, 4), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=3000))
        r = minimize(obj, r.x, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=3000))
        print(s, round(r.fun, 4), np.round(np.abs(r.x), 4).tolist(), flush=True)
        if best is None or r.fun < best[0]:
            best = (r.fun, np.abs(r.x))
    print(f"BOUND (3 pair constraints, 4 variables): {best[0]:.4f}  at A0,A1,B0,B1 = {np.round(best[1], 4).tolist()}")
    save("bound", {"value": float(best[0]), "vars_A0_A1_B0_B1": best[1].tolist(),
                   "source": "3 pairwise NP constraints, 12 random Nelder-Mead starts, seed 2; loose (not certified tight)"})


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--beta" in a:
        beta_run([0.1, 0.5, 1.0])
    elif "--bound" in a:
        bound()
    else:
        verify()
