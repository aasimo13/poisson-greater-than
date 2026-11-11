"""Self test for src/soft_core.py (sampled comparison tools).

Checks the soft LPs against the existing strict code (core.lp_minimax, twotype_bound.solve_lp),
the Neyman-Pearson feasibility test three ways (floats, mpmath, LP), every monotonicity the chain
floor relies on, the floor itself (strict target gives 92.29; K = 3 brute force), the mpmath
rechecks, and the budgets. Prints PASS / FAIL lines and exits nonzero on any failure.

Usage (repo root): python3 src/soft_selftest.py        (under a minute)
The error target is fixed at 0.01 here; an EPS environment variable (read by core.py) is ignored.
"""
import itertools
import math
import os
import sys
import time
import numpy as np
from scipy.stats import poisson, skellam

os.environ.pop("EPS", None)        # core.TARGET must be the 0.01 this file compares against
import core
import twotype_bound as tb
import soft_core as sc
from soft_core import (target, strict_target, lp_shared, lp_two, recheck_shared_mp, recheck_two_mp,
                       budget_shared, budget_two, pair_feasible, pair_margin, pair_margin_mp,
                       pair_lp_slack, min_hi, chain_feasible, chain_floor)

EPS = 0.01
BAND = [98.2976, 4.6013, 0, 28.7852]
STORED = ([0.001, 4.701, 22.414, 52.805], [0.001, 5.005, 23.512, 53.561])
REF_T0 = [0, 3.92, 13.25, 27.99]
REF_T05 = [0, 1.36, 3.96, 7.84]
FAILS = []
T0 = time.time()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}  [{time.time() - T0:.1f}s]", flush=True)
    if not ok:
        FAILS.append(name)


def coin_decoder(M1, M2):
    n1 = np.arange(M1 + 1)[:, None]; n2 = np.arange(M2 + 1)[None, :]
    return np.where(n1 > n2, 1.0, np.where(n1 == n2, 0.5, 0.0))


def test_target():
    s0 = target(4, 0)
    ok = all(s0[a, b] == (1.0 if a > b else 0.5 if a == b else 0.0) for a in range(4) for b in range(4))
    s1 = target(4, 1.0)
    ok &= abs(s1[2, 0] - 1 / (1 + math.exp(-2.0))) < 1e-15 and np.allclose(s1 + s1.T, 1.0, atol=1e-15)
    off = ~np.eye(4, dtype=bool)
    ok &= np.max(np.abs(target(4, 1e-3) - s0)[off]) < 1e-12 and np.all(np.diag(target(4, 1e-3)) == 0.5)
    check("target: T = 0 limit, sigmoid values, symmetry", bool(ok))


def test_strict():
    S = strict_target(4)
    y = BAND[::-1]
    t, d = lp_shared(BAND, y, S)
    ref = core.lp_minimax(core.pair_means(BAND, y), core.labels(4))
    check("(a) lp_shared on the strict target = core.lp_minimax, banded shape",
          abs(t - ref) < 1e-7, f"soft {t!r} core {float(ref)!r}")
    mp_t = recheck_shared_mp(BAND, y, S, d)
    check("(d) recheck_shared_mp agrees with lp_shared, strict banded", abs(mp_t - t) < 1e-7, f"mp {mp_t!r}")

    x, y2 = STORED
    t2, d2, M = lp_two(x, y2, S)
    ref2, dref, Mref = tb.solve_lp(x, y2)
    mine = recheck_two_mp(x, y2, S, d2); theirs = recheck_two_mp(x, y2, S, np.clip(dref, 0, 1))
    check("(a) lp_two on the strict target = twotype_bound.solve_lp, stored 106.366 design",
          abs(t2 - ref2) < 1e-7 and mine <= theirs + 1e-9 and theirs - mine < 2e-7 and M == Mref and t2 <= EPS,
          f"soft {t2!r} twotype objective {float(ref2)!r}; mpmath error of the two decoders {mine!r} {theirs!r}")
    check("(d) recheck_two_mp agrees with lp_two, strict stored design", abs(mine - t2) < 1e-7, f"mp {mine!r}")
    return (x, y2, d2)


def test_recheck_independent():
    x = [0.7, 3.1, 9.0]; y = [0.2, 5.5, 2.4]; bg = 0.3
    means = np.add.outer(x, y) + bg
    M = sc.grid_max(means, 1e-13)
    d = (np.arange(M + 1) >= 5).astype(float)
    Sx = poisson.sf(4, means)
    w = recheck_shared_mp(x, y, Sx, d, bg=bg)
    check("recheck_shared_mp against scipy sf for a threshold decoder", w < 1e-12, f"worst {w:.2e}")
    M1, M2 = sc.grid_max(x, 1e-13), sc.grid_max(y, 1e-13)
    Sk = np.array([[skellam.sf(0, a, b) + 0.5 * skellam.pmf(0, a, b) for b in y] for a in x])
    w2 = recheck_two_mp(x, y, Sk, coin_decoder(M1, M2))
    check("recheck_two_mp against scipy skellam for the N1 > N2 coin decoder", w2 < 1e-10, f"worst {w2:.2e}")
    w3, Q, TL = recheck_shared_mp(x, y, Sx, d[:12], bg=bg, full=True)
    cut = poisson.sf(11, means)
    check("recheck charges the truncated tail", abs(w3 - cut.max()) < 1e-12 and np.allclose(TL, cut, atol=1e-13),
          f"worst {w3:.3e} tail {cut.max():.3e}")
    try:
        recheck_shared_mp(x, y, Sx, np.array([0.0, 1.2]))
        bad = False
    except ValueError:
        bad = True
    check("recheck rejects decoder entries outside [0, 1]", bad)


def test_soft_lp():
    worst = 0.0; rows = []
    for T in [0, 0.25, 0.5, 1, 2, 4]:
        S = target(4, T)
        lv = np.array(REF_T0) * {0: 1.0, 0.25: 0.5, 0.5: 0.2, 1: 0.07, 2: 0.025, 4: 0.01}[T]   # about 70% of what T needs
        t, d2, _ = lp_two(lv, lv, S)
        m = recheck_two_mp(lv, lv, S, d2)
        ts, d = lp_shared(lv, lv[::-1], S, bg=0.2)
        ms = recheck_shared_mp(lv, lv[::-1], S, d, bg=0.2)
        worst = max(worst, abs(t - m), abs(ts - ms)); rows.append((T, round(t, 6), round(ts, 6)))
    check("(d) mpmath rechecks agree with the float LPs, soft targets, zero level, background",
          worst < 1e-7, f"max gap {worst:.1e}; (T, t_two, t_shared) {rows}")

    ok = True; detail = []
    for T, lv in ((0, REF_T0), (0.5, REF_T05)):
        S = target(4, T)
        M = sc.grid_max(lv)
        P = sc.pmf_rows(lv, M)
        qfix = P @ coin_decoder(M, M) @ P.T
        efix = float(np.max(np.abs(qfix - S)))
        t, d2, _ = lp_two(lv, lv, S)
        ok &= t <= efix + 1e-10
        detail.append((T, repr(efix), repr(t)))
    check("LP decoder is no worse than the fixed decoder '1 if N1 > N2, coin on a tie' (T, fixed error, LP error)", ok, str(detail))


def rand_s(rng):
    return [0.0, 1.0, 0.5, rng.uniform(), rng.uniform(0, 0.02), 1 - rng.uniform(0, 0.02)][rng.integers(0, 6)]


def test_pair():
    rng = np.random.default_rng(20261111)
    n = 400; n_feas = 0; n_lp = 0; bad_mp = 0; bad_lp = 0; worst_mp = 0.0
    kinds = {"mu0": 0, "equal_s": 0, "lower_mean_higher_target": 0}
    for _ in range(n):
        mu = [0.0, rng.uniform(0, 3), rng.uniform(0, 40)][rng.integers(0, 3)]
        nu = mu + [0.0, rng.exponential(1.0), rng.exponential(8.0)][rng.integers(0, 3)]
        eps = [0.01, 0.001, 0.05, 0.2][rng.integers(0, 4)]
        s1, s2 = rand_s(rng), rand_s(rng)
        if rng.uniform() < 0.15:
            s2 = s1
        kinds["mu0"] += mu == 0; kinds["equal_s"] += s1 == s2; kinds["lower_mean_higher_target"] += (s1 > s2 and nu > mu)
        if rng.uniform() < 0.5:
            mu, nu, s1, s2 = nu, mu, s2, s1
        m = pair_margin(mu, nu, s1, s2, eps); mm = pair_margin_mp(mu, nu, s1, s2, eps)
        worst_mp = max(worst_mp, abs(m - mm))
        f = pair_feasible(mu, nu, s1, s2, eps)
        n_feas += f
        if abs(mm) > 1e-9 and f != (mm >= 0):
            bad_mp += 1
        sl = pair_lp_slack(mu, nu, s1, s2, eps)
        if abs(sl) > 1e-7:
            n_lp += 1
            bad_lp += f != (sl < 0)
    check("(c) pair_margin floats = mpmath on 400 random cases", worst_mp < 1e-10 and bad_mp == 0,
          f"max gap {worst_mp:.1e}; feasible {n_feas}, infeasible {n - n_feas}; {kinds}")
    check("(c) pair_feasible = two hypothesis LP on the same cases", bad_lp == 0 and n_lp > 300 and 50 < n_feas < n - 50,
          f"{n_lp} cases with a decisive LP slack, {bad_lp} disagreements")

    nb = 0; bad = 0
    for _ in range(150):
        lo = [0.0, rng.uniform(0, 3), rng.uniform(0, 30)][rng.integers(0, 3)]
        eps = [0.01, 0.003, 0.05][rng.integers(0, 3)]
        s1, s2 = rand_s(rng), rand_s(rng)
        h = min_hi(lo, s1, s2, eps)
        if h == lo:
            bad += not pair_feasible(lo, lo, s1, s2, eps) or abs(s1 - s2) > 2 * eps + 1e-12
            continue
        nb += 1
        for delta in (1e-6, 1e-3, 0.05):
            above = lo + (h - lo) * (1 + delta); below = lo + (h - lo) * (1 - delta)
            bad += not pair_feasible(lo, above, s1, s2, eps)
            bad += pair_feasible(lo, below, s1, s2, eps)
            bad += pair_feasible(above, lo, s2, s1, eps) != pair_feasible(lo, above, s1, s2, eps)
            bad += pair_margin_mp(lo, above, s1, s2, eps) < 0 or pair_margin_mp(lo, below, s1, s2, eps) > 0
            if delta >= 1e-3:
                sa = pair_lp_slack(lo, above, s1, s2, eps); sb = pair_lp_slack(lo, below, s1, s2, eps)
                bad += sa > 1e-8 or sb < -1e-8
            if delta == 0.05:
                bad += not sc.pair_feasible_lp(lo, above, s1, s2, eps) or sc.pair_feasible_lp(lo, below, s1, s2, eps)
    check("(c) min_hi is the boundary: feasible just above, infeasible just below (floats, mpmath, LP)",
          bad == 0 and nb > 60, f"{nb} boundaries, {bad} violations")
    check("(c) first step of the strict chain is ln 99", abs(min_hi(0.0, 0.0, 1.0, EPS) - math.log(99)) < 1e-9
          and abs(min_hi(0.0, 1.0, 0.0, EPS) - math.log(99)) < 1e-9, repr(min_hi(0.0, 0.0, 1.0, EPS)))


PAIRS = [(0.0, 1.0), (1.0, 0.0), (0.5, 1.0), (0.0, 0.5), (0.5, 0.0), (0.3, 0.6), (0.9, 0.2), (0.119, 0.5), (0.5, 0.52)]


def test_monotone():
    bad_hi = 0; bad_edge = 0
    for lo in (0.0, 0.5, 3.0, 12.0):
        for s1, s2 in PAIRS:
            g = lo + np.arange(0, 161) * 0.25
            f = np.array([pair_feasible(lo, h, s1, s2, EPS) for h in g])
            bad_hi += bool(np.any(f[:-1] & ~f[1:]))
            h = min_hi(lo, s1, s2, EPS)
            if h < g[-1]:
                i = int(np.argmax(f))
                bad_edge += not (g[i] >= h - 1e-9 and (i == 0 or g[i - 1] < h))
    check("monotone in hi: feasibility on a grid is False...True and switches at min_hi", bad_hi == 0 and bad_edge == 0,
          f"{bad_hi} non monotone rows, {bad_edge} misplaced edges")

    los = np.arange(0, 81) * 0.5
    bad_lo = 0; bad_shift = 0
    for s1, s2 in PAIRS:
        v = np.array([min_hi(l, s1, s2, EPS) for l in los])
        bad_lo += bool(np.any(np.diff(v) < -1e-9))
        bad_shift += bool(np.any(np.diff(v - los) < -1e-9))
    check("monotone in lo: min_hi(lo) nondecreasing on 0..40 (lowering lo keeps feasibility)", bad_lo == 0, f"{bad_lo} violations")
    check("shift: min_hi(lo) - lo nondecreasing on 0..40 (subtracting a constant keeps feasibility)", bad_shift == 0,
          f"{bad_shift} violations")

    rng = np.random.default_rng(7)
    bad = 0; used = 0
    for _ in range(300):
        lo = rng.uniform(0, 20); hi = lo + rng.exponential(6.0); eps = [0.01, 0.05][rng.integers(0, 2)]
        s1, s2 = rand_s(rng), rand_s(rng)
        if pair_feasible(lo, hi, s1, s2, eps):
            used += 1
            lo2 = rng.uniform(0, lo); c = rng.uniform(0, lo); up = hi + rng.exponential(3.0)
            bad += not pair_feasible(lo2, hi, s1, s2, eps)
            bad += not pair_feasible(lo, up, s1, s2, eps)
            bad += not pair_feasible(lo - c, hi - c, s1, s2, eps)
    check("random feasible pairs stay feasible under lower lo, higher hi, common shift down", bad == 0 and used > 50,
          f"{used} feasible pairs, {bad} violations")


def greedy_all_orders(S, eps):
    """Chain floor for the row coordinate by plain enumeration of every order (no pruning)."""
    K, Kb = S.shape
    best = math.inf
    for order in itertools.permutations(range(K)):
        lev = [0.0]
        for j in range(1, K):
            top = lev[-1]
            for i in range(j):
                top = max(top, max(min_hi(lev[i], S[order[i], b], S[order[j], b], eps) for b in range(Kb)))
            lev.append(top)
        best = min(best, lev[-1])
    return best


def test_floor():
    S = strict_target(4)
    fl = chain_floor(S, EPS)
    L = [tb.mp.mpf(0)]
    for _ in range(3):
        L.append(tb.min_hi(L[-1]))
    ref = [float(v) for v in L]
    gap = max(abs(a - b) for a, b in zip(fl["levels"]["x"], ref))
    check("(b) chain_floor on the strict target = 92.29 and the twotype_bound mpmath chain",
          abs(fl["total"] - 92.29313511) < 1e-7 and gap < 1e-8 and abs(fl["floor_y"] - ref[3]) < 1e-8,
          f"total {fl['total']!r} levels {[round(v, 6) for v in fl['levels']['x']]} max gap to mpmath {gap:.1e}")

    rows = []; ok = True
    for T in [0, 0.25, 0.5, 1, 2, 4]:
        St = target(4, T)
        f = chain_floor(St, EPS)
        ok &= chain_feasible(f["levels"]["x"], St, EPS, tol=1e-9) and chain_feasible(f["levels"]["y"], St.T, EPS, tol=1e-9)
        ok &= abs(f["floor_x"] - greedy_all_orders(St, EPS)) < 1e-12 and abs(f["floor_y"] - greedy_all_orders(St.T, EPS)) < 1e-12
        rows.append((T, sc.floor_down(f["total"])))
    check("chain_floor at T = 0, .25, .5, 1, 2, 4: greedy table is pairwise feasible, search = all 24 orders", ok, str(rows))

    rng = np.random.default_rng(3)
    ok = True; nonid = 0
    for _ in range(6):
        Sr = rng.uniform(0, 1, (4, 4))
        f = chain_floor(Sr, 0.05)
        ok &= abs(f["floor_x"] - greedy_all_orders(Sr, 0.05)) < 1e-12 and abs(f["floor_y"] - greedy_all_orders(Sr.T, 0.05)) < 1e-12
        ok &= chain_feasible(f["levels"]["x"], Sr, 0.05, tol=1e-9) and chain_feasible(f["levels"]["y"], Sr.T, 0.05, tol=1e-9)
        ok &= f["levels"]["x"][f["permutation"]["x"][0]] == 0.0 and f["levels"]["x"][f["permutation"]["x"][-1]] == f["floor_x"]
        nonid += f["permutation"]["x"] != [0, 1, 2, 3]
    check("chain_floor on random 4 x 4 targets: pruned search = plain enumeration", ok and nonid > 0,
          f"{nonid} of 6 with a non identity order")

    Sbig = target(4, 100.0)
    fb = chain_floor(Sbig, EPS)
    b1 = budget_two([0, 1, 2, 3], [0, 1, 2, 3], Sbig, EPS); b2 = budget_shared([3, 2, 1, 0], [0, 1, 2, 3], Sbig, EPS)
    check("(e) T = 100 (|s - 0.5| <= eps everywhere): floor 0 and both budgets 0",
          np.max(np.abs(Sbig - 0.5)) <= EPS and fb["total"] == 0.0 and b1 == (0.0, 0.0) and b2 == (0.0, 0.0),
          f"max |s - 0.5| = {np.max(np.abs(Sbig - 0.5)):.5f}")
    fs = chain_floor(target(4, 30.0), EPS)
    fz = chain_floor(target(4, 60.0), EPS)
    bz = budget_two([0, 1, 2, 3], [0, 1, 2, 3], target(4, 60.0), EPS)[0]
    check("T = 30 (one coordinate moves s by more than 2 eps): floor is positive; T = 60: floor 0 but budget > 0",
          fs["total"] > 0 and fz["total"] == 0.0 and 0 < bz < 1, f"floor(30) {fs['total']:.5f}, budget(60) {bz:.5f}")


def brute_k3(S, eps, n=61):
    """Smallest top level over a grid of level triples meeting all pairwise constraints."""
    fl = chain_floor(S, eps)
    out = []
    for key, SS in (("floor_x", S), ("floor_y", S.T)):
        f = fl[key]
        g = np.linspace(0, 1.2 * f + 0.5, n)
        F = {}
        for a in range(3):
            for a2 in range(a + 1, 3):
                F[a, a2] = np.array([[all(pair_feasible(g[i], g[j], SS[a, b], SS[a2, b], eps) for b in range(3))
                                      for j in range(n)] for i in range(n)])
        ok = F[0, 1][:, :, None] & F[0, 2][:, None, :] & F[1, 2][None, :, :]
        mx = np.maximum(np.maximum(g[:, None, None], g[None, :, None]), g[None, None, :])
        out.append((f, float(mx[ok].min()) if ok.any() else math.inf, float(g[1])))
    return out


def test_brute():
    rng = np.random.default_rng(11)
    cases = [("strict", strict_target(3), EPS), ("T=0", target(3, 0), EPS), ("T=0.5", target(3, 0.5), EPS),
             ("random A", rng.uniform(0, 1, (3, 3)), 0.05), ("random B", rng.uniform(0, 1, (3, 3)), 0.05)]
    ok = True; rows = []
    for name, S, eps in cases:
        for f, best, step in brute_k3(S, eps):
            ok &= best >= f - 1e-9 and best <= f + 6 * step
            rows.append((name, round(f, 4), round(best, 4)))
    check("K = 3 brute force over level triples (61^3 grid): nothing below the floor, something close above",
          ok, f"(case, floor, best grid top) {rows}")


def test_scale_and_budget(stored):
    S = target(4, 0.5)
    lv = np.array(REF_T05)
    cs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.6, 2.0, 3.0]
    e2 = [lp_two(c * lv, c * lv, S)[0] for c in cs]
    e1 = [lp_shared(c * lv, c * lv[::-1], S)[0] for c in cs]
    e1b = [lp_shared(c * lv, c * lv[::-1], S, bg=0.5)[0] for c in cs]
    sb = strict_target(4)
    e0 = [lp_shared(c * np.array(BAND), c * np.array(BAND[::-1]), sb)[0] for c in (0.2, 0.5, 0.8, 1.0, 1.1)]
    mono = all(np.all(np.diff(e) <= 5e-7) for e in (e2, e1, e1b, e0))
    check("LP error is nonincreasing in the scale (two channels, one detector, with background, strict)", mono,
          f"two {[round(v, 5) for v in e2]}")

    c0 = budget_shared(BAND, BAND[::-1], sb, EPS)[0]
    r0 = core.budget_for_shape(BAND, BAND[::-1])[0]
    check("budget_shared on the strict target = core.budget_for_shape, banded shape", abs(c0 - r0) < 1e-3 * r0 / 100,
          f"soft {c0!r} core {float(r0)!r}")

    rows = []; ok = True
    for T, shape, bound in ((0, REF_T0, 55.98), (0.5, REF_T05, 15.68)):
        St = target(4, T)
        c, s = budget_two(shape, shape, St, EPS)
        xs = s * np.array(shape)
        t, d2, _ = lp_two(xs, xs, St)
        m = recheck_two_mp(xs, xs, St, d2)
        fl = chain_floor(St, EPS)
        ok &= c <= bound + 0.01 and fl["total"] <= c and m <= EPS + 1e-9
        ok &= chain_feasible(xs, St, EPS + 1e-9) and chain_feasible(xs, St.T, EPS + 1e-9)
        ok &= lp_two(0.98 * xs, 0.98 * xs, St)[0] > EPS
        rows.append((T, round(float(c), 4), sc.floor_down(fl["total"]), f"{m:.9f}"))
    check("budget_two on two reference shapes with x = y: under the fixed decoder bound, over the floor, mpmath recheck <= eps, "
          "necessary pairwise conditions hold", ok, f"(T, cost, floor, mpmath error) {rows}")

    x, y, d2 = stored
    ok = chain_feasible(x, sb, EPS) and chain_feasible(y, sb.T, EPS)
    ok &= chain_feasible(BAND, sb, recheck_shared_mp(BAND, BAND[::-1], sb, lp_shared(BAND, BAND[::-1], sb)[1]))
    ok &= not chain_feasible([0.0, 4.0, 19.94, 46.15], sb, EPS)
    check("designs that meet eps satisfy the pairwise conditions (stored two channel, banded); a short level fails", ok)

    binf = budget_two([1, 1, 1, 1], [0, 1, 2, 3], target(4, 0), EPS, cost_cap=40.0)
    bone = budget_shared([0, 0, 0, 0], [0, 0, 0, 0], target(4, 0), EPS)
    check("budget returns inf for a shape that cannot meet eps", binf == (math.inf, math.inf) and bone == (math.inf, math.inf))


if __name__ == "__main__":
    test_target()
    stored = test_strict()
    test_recheck_independent()
    test_soft_lp()
    test_pair()
    test_monotone()
    test_floor()
    test_brute()
    test_scale_and_budget(stored)
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - T0:.1f}s)")
    sys.exit(1 if FAILS else 0)
