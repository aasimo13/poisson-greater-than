"""Two molecule types counted separately: certified lower bound and verification of the 106.4 design.

Receiver sees (N1, N2), N1 ~ Pois(x[A]), N2 ~ Pois(y[B]), decides [A > B], worst case error 1%.
Cost = max(x) + max(y).

Part 1. The stored design (cost 106.366) and a locally polished one (cost 106.339) are checked by
        the exact 2D LP over the joint count grid (randomized decoder), then the LP decoder's
        worst case error is recomputed with mpmath (30 digits).
Part 2. Lower bound. Any design must separate, at every pair of adjacent sources, a null Poisson(lo)
        from an alternative Poisson(hi) with both errors <= 1%. The smallest hi for a given lo is the
        randomized Neyman-Pearson level min_hi(lo) (the randomized test is what makes the first level
        exactly ln(99); the non-randomized test would give -ln(0.01) and is not a valid bound).
        Chain: L0 = 0, L_{i+1} = min_hi(L_i). The four x levels and four y levels each need the chain,
        so cost >= 2 * L3.

Usage (repo root): python3 src/twotype_bound.py
"""
import time
import numpy as np
import mpmath as mp
from scipy.stats import poisson
from scipy.optimize import linprog

mp.mp.dps = 30
ALPHA = mp.mpf('0.01')
LAB = np.array([[1.0 if a > b else 0.0 for b in range(4)] for a in range(4)])

DESIGNS = {
    "stored design (cost 106.366)": ([0.001, 4.701, 22.414, 52.805], [0.001, 5.005, 23.512, 53.561]),
    "polished design (cost 106.339)": ([0.001, 4.70018343495914, 22.415664145162424, 52.772600259653785],
                                       [0.001, 5.003789094522123, 23.440505125676324, 53.566100561810316]),
}


# ---------- Part 1: 2D LP and mpmath recheck ----------
def grid_max(lams, tail=1e-9):
    return max(int(poisson.isf(tail / 4, max(l, 1e-12))) + 2 for l in lams)


def solve_lp(x, y, tail=1e-9):
    """Minimax error of the best randomized decoder d(n1, n2); truncated mass counts as error."""
    M1, M2 = grid_max(x, tail), grid_max(y, tail)
    P1 = np.array([poisson.pmf(np.arange(M1 + 1), l) for l in x])
    P2 = np.array([poisson.pmf(np.arange(M2 + 1), l) for l in y])
    nv = (M1 + 1) * (M2 + 1)
    A = np.zeros((16, nv + 1)); b = np.zeros(16); k = 0
    for a in range(4):
        for bb in range(4):
            p = np.outer(P1[a], P2[bb]).ravel()
            tl = max(0.0, 1.0 - p.sum())
            if LAB[a, bb] == 1:
                A[k, :nv] = -p; b[k] = -(1.0 + tl)
            else:
                A[k, :nv] = p; b[k] = -tl
            A[k, nv] = -1.0; k += 1
    c = np.zeros(nv + 1); c[nv] = 1
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * nv + [(0, None)], method="highs")
    return r.x[nv], r.x[:nv].reshape(M1 + 1, M2 + 1), (M1, M2)


def verify_design(name, x, y):
    t0 = time.time()
    t, d, (M1, M2) = solve_lp(x, y)
    print(name)
    print(f"  grid {M1} {M2}  LP worst case error t = {t:.8f}")
    print("  cost max(x_a + y_b) =", max(xa + yb for xa in x for yb in y))
    mx = [[mp.mpf(str(l)) ** n * mp.e ** (-mp.mpf(str(l))) / mp.factorial(n) for n in range(M1 + 1)] for l in x]
    my = [[mp.mpf(str(l)) ** n * mp.e ** (-mp.mpf(str(l))) / mp.factorial(n) for n in range(M2 + 1)] for l in y]
    worst = mp.mpf(0)
    for a in range(4):
        for b in range(4):
            s = mp.mpf(0)
            for n1 in range(M1 + 1):
                for n2 in range(M2 + 1):
                    s += mx[a][n1] * my[b][n2] * mp.mpf(float(d[n1, n2]))
            tl = 1 - sum(mx[a]) * sum(my[b])         # mass outside the grid is counted as error
            err = (1 - s + tl) if a > b else (s + tl)
            worst = max(worst, err)
            print(f"  {a} {b} {'TRUE ' if a > b else 'FALSE'} {mp.nstr(err, 12)}")
    print("  mpmath 30-digit worst error of LP decoder:", mp.nstr(worst, 12))


# ---------- Part 2: Neyman-Pearson chain ----------
def pmf(l, n):
    l = mp.mpf(l)
    return mp.e ** (-l) if (l == 0 and n == 0) else (mp.mpf(0) if l == 0 else mp.e ** (-l + n * mp.log(l) - mp.loggamma(n + 1)))


def sf(l, k):    # P(N > k)
    l = mp.mpf(l)
    if l == 0:
        return mp.mpf(0)
    return mp.gammainc(k + 1, 0, l, regularized=True)


def power(lo, hi, randomized=True):
    """Power at hi of the best level-ALPHA test of Pois(lo) against Pois(hi)."""
    k = 0
    while sf(lo, k) > ALPHA:
        k += 1
    if not randomized:
        return sf(hi, k)
    g = (ALPHA - sf(lo, k)) / pmf(lo, k)
    return sf(hi, k) + g * pmf(hi, k)


def min_hi(lo, randomized=True):
    lo_, hi_ = mp.mpf(lo), mp.mpf(lo) + 1
    while power(lo, hi_, randomized) < 1 - ALPHA:
        hi_ = hi_ * 2 + 1
    a, b = lo_, hi_
    for _ in range(110):
        m = (a + b) / 2
        if power(lo, m, randomized) >= 1 - ALPHA:
            b = m
        else:
            a = m
    return b


def chain_bound():
    print("min_hi(0) randomized NP :", mp.nstr(min_hi(0), 12), " ln(99) =", mp.nstr(mp.log(99), 12))
    print("min_hi(0) nonrandomized :", mp.nstr(min_hi(0, False), 12), " -ln(0.01) =", mp.nstr(-mp.log(ALPHA), 12),
          "(not used: not a valid bound)")
    L = [mp.mpf(0)]
    for _ in range(3):
        L.append(min_hi(L[-1]))
    LB = 2 * L[3]
    print("randomized chain levels", [mp.nstr(v, 10) for v in L], "2*L3 =", mp.nstr(LB, 10))
    vals = [min_hi(mp.mpf(i) / 2) for i in range(0, 120)]
    print("min_hi increasing on grid 0..59.5 step .5:", all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
    return LB


if __name__ == "__main__":
    for name, (x, y) in DESIGNS.items():
        verify_design(name, x, y)
    LB = chain_bound()
    cost = mp.mpf('106.366')
    print("found design cost 106.366; LB", mp.nstr(LB, 8), "gap % over LB:", mp.nstr((cost - LB) / LB * 100, 6),
          " LB shortfall % of cost:", mp.nstr((cost - LB) / cost * 100, 6))
