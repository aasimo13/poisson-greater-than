"""One detector, sampled comparison: lower bound from the exact pairwise relaxation (MILP).

Model: N ~ Poisson(x[A] + y[B]), A, B in 0..3, target s[A, B] = sigmoid((A - B) / T), eps = 0.01
(fixed; an EPS environment variable is ignored), cost P = max(x) + max(y). Write m_j for the mean of input pair j and s_j for its target.

Necessary condition for ANY encoder and ANY decoder: restricted to two input pairs j, k the
decoder is a test between Poisson(m_j) and Poisson(m_k) that must land within eps of s_j and s_k.
With H(lo; s_lo, s_hi) = soft_core.min_hi (the smallest higher mean for which such a test exists;
feasibility is monotone in the higher mean) this reads
    m_k >= H(m_j; s_j, s_k)   or   m_j >= H(m_k; s_k, s_j)        for every j, k.
H is nondecreasing in lo (lowering the lower mean keeps feasibility, see soft_core.chain_floor),
so on a grid cell [g_i, g_(i+1)] it is at least H(g_i). As in src/lb_pairwise2.py each mean is
split over segments (multiple choice formulation), on each segment H is replaced by facets of the
lower convex hull of that staircase (local grid of 16 to 2000 cells per segment), minus a 1e-6
margin, and a binary per pair of inputs picks which of the two is the higher one. After each solve
the segments that hold the solution's means are split in three and the MILP is solved again; every
stage is a valid bound and the largest is kept. Added without loss of generality:
  min(x) = 0 and min(y) = 0   a common offset is undone by adding Poisson noise at the receiver
  max(x) >= max(y)            else swap the senders and flip the output: x' = y, y' = x, d' = 1 - d
                              (the target obeys s[b, a] = 1 - s[a, b])
  x[0] <= x[K-1]              else reverse both alphabets and flip the output (s[K-1-a, K-1-b] =
                              1 - s[a, b]); this keeps max(x) and max(y)
Each map sends a design that meets eps to one of the same or lower cost that meets eps, so some
cheapest design satisfies all four. The cost is capped at PMAX, a cost at which a design is known
to exist.

Order binaries. The mean of (a, b) is below the mean of (a', b) exactly when x[a] <= x[a'], for
every b. So one binary per pair of x levels (and per pair of y levels) fixes the higher/lower
choice of all input pairs in a common column (row), and of every two input pairs that are ordered
the same way in both coordinates; only pairs ordered oppositely in x and y keep a free choice.
These equalities hold for every real design (take any strict order of its levels, ties broken by
index), so they cut nothing real; they only make the solver faster.

Check of the encoding: probe() builds the full MILP point of a stored design (after the
symmetry maps above) and reports the largest row or bound violation, which must be 0.

The MILP minimum is a lower bound on P for every design, and so is the solver's dual bound at any
moment, so a run that stops at its time limit still returns a valid (smaller) floor. Status of
the number: "certified" in the sense of the 57.43 and 195.53 of the strict problem, that is a
HiGHS dual bound over tabulated float Neyman-Pearson boundaries, valid for any encoder and
decoder up to solver tolerances. It is not an exact rational certificate and not a proof. It is the
bound of the pairwise relaxation only; whether it is tight is not proved (at T = 0 the ramp design
is within 0.02% of it, at T = 2 the gap is 28%). At T = 4 it comes out below the chain floor,
which is then the floor to quote.

Usage (repo root):
  python3 src/soft_floor_shared.py                       print the STORED bounds from results/soft_floor_shared.json and
                                                         check the encoding against the stored designs (seconds);
                                                         nothing is solved again in this mode
  python3 src/soft_floor_shared.py --run T [SECONDS] [REFINE]   solve for one temperature (default 1200 s, 3 refinements);
                                                         the stored file came from: 0 1440 3, 0.25 900 0, 0.5 900 1,
                                                         1 900 1, 2 900 0, 4 150 2 (1.5 to 15 minutes each)
  python3 src/soft_floor_shared.py --selfcheck           the MILP against brute force for K = 2 (15 seconds)
  python3 src/soft_floor_shared.py --anchor              the same MILP on the strict comparison for K = 3, where the
                                                         README has 57.43 (certified) and 57.4737 (banded pairwise budget)
"""
import json
import math
import os
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
from soft_core import target, strict_target, min_hi, pair_feasible, chain_floor, floor_down, dump_json

EPS = 0.01
MARGIN = 1e-6
OUT = os.environ.get("SOFT_FLOOR_OUT", "results/soft_floor_shared.json")
DESIGNS = "results/soft_shared.json"
TS = [0, 0.25, 0.5, 1, 2, 4]
K = 4
FAILS = []
T0 = time.time()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ---------- underestimators of H ----------
def canon(s_lo, s_hi):
    """H(lo; s_lo, s_hi) = H(lo; 1 - s_lo, 1 - s_hi) (replace d by 1 - d), so one table serves both."""
    a = (round(float(s_lo), 12), round(float(s_hi), 12))
    b = (round(1.0 - float(s_lo), 12), round(1.0 - float(s_hi), 12))
    return min(a, b)


class Tables:
    """Facets below H on a segment, from H at the left ends of a local grid of the segment: cells
    of 5e-5 pmax, at least 16 and at most 2000 per segment. The staircase costs about one cell
    times the slope of H, so the loss is small from the start and shrinks with the segment."""

    def __init__(self, S, pmax, max_cells=2000):
        self.S = np.asarray(S, float).ravel()
        self.pmax = float(pmax)
        self.max_cells = max_cells
        self.cache = {}
        self.keys = set()

    def top(self, s_lo, s_hi):
        key = canon(s_lo, s_hi)
        return min_hi(self.pmax, key[0], key[1], EPS)

    def facets(self, s_lo, s_hi, L, U, nmax=8):
        """Affine functions a mu + b, each <= H(mu; s_lo, s_hi) on [L, U] (valid because H is nondecreasing)."""
        key = canon(s_lo, s_hi)
        ck = (key, float(L), float(U))
        if ck in self.cache:
            return self.cache[ck]
        self.keys.add(key)
        n = int(min(self.max_cells, max(16, math.ceil((U - L) / (5e-5 * self.pmax)))))
        pts = np.linspace(L, U, n + 1)
        px = []; py = []
        for i in range(n):
            h = min_hi(float(pts[i]), key[0], key[1], EPS)   # H >= H(left end) on the whole cell
            px += [float(pts[i]), float(pts[i + 1])]; py += [h, h]
        hull = lower_hull(px, py)
        segs = []
        for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
            if x2 - x1 < 1e-12:
                continue
            a = (y2 - y1) / (x2 - x1)
            segs.append((a, y1 - a * x1 - MARGIN, x2 - x1))
        if not segs:
            out = [(0.0, min(py) - MARGIN)]
        else:
            if len(segs) > nmax:                        # any subset of the facets is still an underestimator
                cum = np.cumsum([s[2] for s in segs]); cum /= cum[-1]
                keep = sorted(set(min(int(np.searchsorted(cum, q)), len(segs) - 1) for q in np.linspace(0, 1, nmax)))
                segs = [segs[k] for k in keep]
            out = [(a, b) for a, b, _ in segs]
        self.cache[ck] = out
        return out

    def under(self, s_lo, s_hi, breaks, m):
        """Value at m of the underestimator the MILP uses for H(.; s_lo, s_hi) with these segments.
        m on a break point may be placed in either segment, so the smaller value counts."""
        vals = []
        for L, U in zip(breaks[:-1], breaks[1:]):
            if L - 1e-12 <= m <= U + 1e-12:
                vals.append(max(a * m + b for a, b in self.facets(s_lo, s_hi, L, U)))
        return min(vals)


def lower_hull(px, py):
    pts = sorted(zip(px, py))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def pair_list(S):
    s = np.asarray(S, float).ravel()
    return [(j, k) for j in range(len(s)) for k in range(j + 1, len(s)) if abs(s[j] - s[k]) > 2 * EPS]


# ---------- MILP ----------
def canonical(x, y):
    """The image of a design under the symmetry maps that has max(x) >= max(y) and x[0] <= x[-1]."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.max() < y.max():
        x, y = y, x
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    return x.copy(), y.copy()


def solve(S, pmax, breaks, tab, time_limit, log=False, probe=None, symmetric=True):
    """Build the MILP and run HiGHS. With probe = (x, y) nothing is solved: the point of that design
    is built and the largest violation of a row or a bound is returned. symmetric = False leaves
    out the two rows that need s[b, a] = 1 - s[a, b] (the strict target does not have it)."""
    K = np.asarray(S).shape[0]
    s = np.asarray(S, float).ravel()
    J = len(s); nseg = len(breaks) - 1
    segs = list(zip(breaks[:-1], breaks[1:]))
    pairs = pair_list(S)
    cls = [sorted(set(round(float(s[k]), 12) for k in range(J) if abs(s[k] - s[j]) > 2 * EPS)) for j in range(J)]
    hmax = max(tab.top(s[j], c) for j in range(J) for c in cls[j]) + 1.0

    nv = 0
    def alloc(n):
        nonlocal nv
        st = nv; nv += n; return st
    ix = alloc(K); iy = alloc(K); iX = alloc(1); iY = alloc(1)
    imu = alloc(J * nseg); iz = alloc(J * nseg)
    iphi = {}
    for j in range(J):
        for c in cls[j]:
            iphi[j, c] = alloc(nseg)
    io = alloc(len(pairs)); iwx = alloc(K); iwy = alloc(K)
    lev = [(a, a2) for a in range(K) for a2 in range(a + 1, K)]
    ir = {q: alloc(1) for q in lev}                      # r[a, a2] = 1: x[a] comes before x[a2]
    ic = {q: alloc(1) for q in lev}                      # c[b, b2] = 1: y[b] comes before y[b2]
    MU = lambda j, k: imu + j * nseg + k
    Z = lambda j, k: iz + j * nseg + k

    R = []; C = []; V = []; lo = []; hi = []
    def add(coefs, l, u):
        r = len(lo)
        for c, v in coefs:
            R.append(r); C.append(c); V.append(v)
        lo.append(l); hi.append(u)

    inf = math.inf
    for a in range(K):
        add([(ix + a, 1), (iX, -1)], -inf, 0)
        add([(iy + a, 1), (iY, -1)], -inf, 0)
    add([(iX, 1), (iY, 1)], -inf, pmax)
    mean = lambda j: [(ix + j // K, 1), (iy + j % K, 1)]
    for j in range(J):
        add(mean(j) + [(MU(j, k), -1) for k in range(nseg)], 0, 0)
        add([(Z(j, k), 1) for k in range(nseg)], 1, 1)
        for k, (L, U) in enumerate(segs):
            add([(MU(j, k), 1), (Z(j, k), -L)], 0, inf)
            add([(MU(j, k), 1), (Z(j, k), -U)], -inf, 0)
            for c in cls[j]:
                for al, be in tab.facets(s[j], c, L, U):
                    add([(iphi[j, c] + k, 1), (MU(j, k), -al), (Z(j, k), -be)], 0, inf)
    for p, (j, k) in enumerate(pairs):
        cj = round(float(s[j]), 12); ck = round(float(s[k]), 12)
        # o = 1: k is the higher mean, m_k >= H(m_j; s_j, s_k)
        add(mean(k) + [(iphi[j, ck] + q, -1) for q in range(nseg)] + [(io + p, -hmax)], -hmax, inf)
        # o = 0: j is the higher mean, m_j >= H(m_k; s_k, s_j)
        add(mean(j) + [(iphi[k, cj] + q, -1) for q in range(nseg)] + [(io + p, hmax)], 0, inf)
    add([(iwx + a, 1) for a in range(K)], 1, 1)
    add([(iwy + a, 1) for a in range(K)], 1, 1)
    for a in range(K):                                   # w = 1 forces the level to 0
        add([(ix + a, 1), (iwx + a, pmax)], -inf, pmax)
        add([(iy + a, 1), (iwy + a, pmax)], -inf, pmax)
    if symmetric:
        add([(iX, 1), (iY, -1)], 0, inf)                 # max(x) >= max(y)
        add([(ix + K - 1, 1), (ix, -1)], 0, inf)         # x[0] <= x[K-1]
    for (a, a2) in lev:                                  # order binaries agree with the levels
        add([(ix + a, 1), (ix + a2, -1), (ir[a, a2], pmax)], -inf, pmax)
        add([(ix + a2, 1), (ix + a, -1), (ir[a, a2], -pmax)], -inf, 0)
        add([(iy + a, 1), (iy + a2, -1), (ic[a, a2], pmax)], -inf, pmax)
        add([(iy + a2, 1), (iy + a, -1), (ic[a, a2], -pmax)], -inf, 0)
    for a in range(K):                                   # and form a total order
        for a2 in range(a + 1, K):
            for a3 in range(a2 + 1, K):
                for iv in (ir, ic):
                    add([(iv[a, a2], 1), (iv[a2, a3], 1), (iv[a, a3], -1)], 0, 1)
    for p, (j, k) in enumerate(pairs):                   # o = 1: input pair k has the higher mean
        a, b, a2, b2 = j // K, j % K, k // K, k % K
        if a == a2:
            add([(io + p, 1), (ic[b, b2], -1)], 0, 0)
        elif b == b2:
            add([(io + p, 1), (ir[a, a2], -1)], 0, 0)
        elif b < b2:                                     # both coordinates say "k higher" iff r = c = 1
            add([(io + p, 1), (ir[a, a2], -1), (ic[b, b2], -1)], -1, 0)
        else:                                            # y says "k higher" iff c[b2, b] = 0
            add([(io + p, 1), (ir[a, a2], -1), (ic[b2, b], 1)], 0, 1)

    A = coo_matrix((V, (R, C)), shape=(len(lo), nv)).tocsc()
    cost = np.zeros(nv); cost[iX] = 1; cost[iY] = 1
    integ = np.zeros(nv, bool)
    integ[iz:iz + J * nseg] = True; integ[io:io + len(pairs)] = True; integ[iwx:iwy + K] = True
    lb = np.zeros(nv); ub = np.full(nv, hmax)
    ub[ix:iY + 1] = pmax; ub[imu:imu + J * nseg] = pmax
    ub[iz:iz + J * nseg] = 1; ub[io:io + len(pairs)] = 1; ub[iwx:iwy + K] = 1
    for q in lev:
        integ[ir[q]] = True; integ[ic[q]] = True; ub[ir[q]] = 1; ub[ic[q]] = 1

    if probe is not None:
        x, y = canonical(*probe)
        v = np.zeros(nv)
        v[ix:ix + K] = x; v[iy:iy + K] = y; v[iX] = x.max(); v[iY] = y.max()
        v[iwx + int(np.argmin(x))] = 1; v[iwy + int(np.argmin(y))] = 1
        m = (x[:, None] + y[None, :]).ravel()
        for j in range(J):
            k = next(q for q, (L, U) in enumerate(segs) if L <= m[j] <= U)
            v[MU(j, k)] = m[j]; v[Z(j, k)] = 1
            for c in cls[j]:
                v[iphi[j, c] + k] = max(0.0, max(al * m[j] + be for al, be in tab.facets(s[j], c, *segs[k])))
        before = lambda t, i, i2: (t[i], i) < (t[i2], i2)
        for (a, a2) in lev:
            v[ir[a, a2]] = before(x, a, a2); v[ic[a, a2]] = before(y, a, a2)
        for p, (j, k) in enumerate(pairs):
            a, b, a2, b2 = j // K, j % K, k // K, k % K
            rx = before(x, a, a2) if a != a2 else None
            cy = before(y, b, b2) if b != b2 else None
            if rx is None:
                v[io + p] = cy
            elif cy is None or rx == cy:
                v[io + p] = rx
            else:
                v[io + p] = m[j] <= m[k]
        act = A @ v
        return float(max(0.0, np.max(np.array(lo) - act), np.max(act - np.array(hi)), np.max(lb - v), np.max(v - ub)))

    import highspy                                       # only needed to solve; the default mode never gets here
    INF = highspy.kHighsInf
    fix = lambda v: np.clip(v, -INF, INF)
    lp = highspy.HighsLp()
    lp.num_col_ = nv; lp.num_row_ = len(lo)
    lp.col_cost_ = cost; lp.col_lower_ = lb; lp.col_upper_ = ub
    lp.row_lower_ = fix(np.array(lo, float)); lp.row_upper_ = fix(np.array(hi, float))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(log))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-6)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    t0 = time.time()
    h.run()
    info = h.getInfo(); st = h.modelStatusToString(h.getModelStatus())
    out = {"pmax": pmax, "segments": nseg, "rows": len(lo), "cols": nv, "pairs": len(pairs), "status": st,
           "time": round(time.time() - t0, 1)}
    db = float(info.mip_dual_bound)
    if st == "Infeasible":
        out["dual_bound"] = pmax                         # nothing at or below pmax passes the relaxation
    else:
        out["dual_bound"] = db if math.isfinite(db) and db > 0 else 0.0
    if info.primal_solution_status == 2:
        v = np.array(h.getSolution().col_value)
        out.update(primal=float(info.objective_function_value), x=v[ix:ix + K].tolist(), y=v[iy:iy + K].tolist())
    return out


def refine(breaks, pts, minw, factor=3):
    breaks = list(breaks); new = set(breaks)
    for p in pts:
        k = int(np.clip(np.searchsorted(breaks, p, side="right") - 1, 0, len(breaks) - 2))
        L, U = breaks[k], breaks[k + 1]
        if U - L < minw:
            continue
        for i in range(1, factor):
            new.add(round(L + (U - L) * i / factor, 6))
    return sorted(new)


def init_breaks(pmax):
    b = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 36, 43, 50, 60,
         70, 80, 90, 100, 115, 130]
    return [float(v) for v in b if v < 0.97 * pmax] + [float(pmax)]


def seeded_breaks(pmax, means):
    """Coarse segments everywhere plus narrow ones around the means of the stored design, where
    the relaxation's optimum is expected. Any set of break points gives a valid bound; this one
    only saves refinement stages (H has ripples about one count wide that a wide hull cuts off)."""
    sc = pmax / 100.0
    b = set(v for v in init_breaks(pmax)[::2]) | {0.0, float(pmax)}
    for m in means:
        for d in (0.06, 0.3):
            for v in (m - d * sc, m + d * sc):
                if 0 < v < pmax:
                    b.add(round(float(v), 6))
    b = sorted(b)
    return [b[0]] + [v for u, v in zip(b[:-1], b[1:]) if v - u > 1e-9]


def design_cost(key):
    return json.load(open(DESIGNS))["T"][key]["families"]["free"]


def admits(S, x, y, tab, breaks):
    """Does the known design pass (a) the exact pairwise conditions and (b) the MILP's relaxed form of
    them on these segments? Both must hold or the encoding would cut off a real design."""
    s = np.asarray(S, float).ravel()
    m = (np.asarray(x, float)[:, None] + np.asarray(y, float)[None, :]).ravel()
    exact = True; relaxed = True; slack = math.inf
    for j, k in pair_list(S):
        exact &= pair_feasible(m[j], m[k], s[j], s[k], EPS + 1e-9)
        lo_, hi_ = (j, k) if m[j] <= m[k] else (k, j)
        need = tab.under(s[lo_], s[hi_], breaks, m[lo_])
        slack = min(slack, m[hi_] - need)
        relaxed &= m[hi_] >= need
    viol = solve(S, max(breaks), breaks, tab, 0.0, probe=(x, y))
    return bool(exact), bool(relaxed and viol <= 1e-9), float(slack)


def brute_k2(S, n=400):
    """K = 2 by brute force: one of x and one of y is 0, the other two run over a grid; smallest
    cost among grid designs that pass every exact pairwise condition, and the grid step."""
    s = np.asarray(S, float).ravel()
    top = 2.0 * chain_floor(S, EPS)["total"] + 2.0
    g = np.linspace(0, top, n)
    best = math.inf
    for zx in range(2):
        for zy in range(2):
            for u in g:
                if u >= best:
                    break
                for v in g:
                    if u + v >= best:
                        break
                    x = [0.0, u] if zx == 0 else [u, 0.0]; y = [0.0, v] if zy == 0 else [v, 0.0]
                    m = [x[0] + y[0], x[0] + y[1], x[1] + y[0], x[1] + y[1]]
                    if all(pair_feasible(m[j], m[k], s[j], s[k], EPS) for j, k in pair_list(S)):
                        best = u + v
    return best, float(g[1])


def selfcheck():
    """The MILP against brute force on K = 2, where the search space is two numbers."""
    for T in (0, 1):
        S = target(2, T)
        brute, step = brute_k2(S)
        pmax = math.ceil(brute + 1.0)
        tab = Tables(S, pmax)
        breaks = init_breaks(pmax); lb = 0.0
        for _ in range(4):
            out = solve(S, pmax, breaks, tab, 60.0)
            lb = max(lb, out["dual_bound"])
            if "x" not in out or out["status"] != "Optimal":
                break
            breaks = refine(breaks, (np.array(out["x"])[:, None] + np.array(out["y"])[None, :]).ravel(), 2e-4 * pmax)
        check(f"K = 2, T = {T}: MILP bound {lb:.4f} <= brute force {brute:.4f} (grid step {step:.4f}) and within 3 steps",
              lb <= brute + 1e-6 and lb >= brute - 3 * step - 0.05, f"last stage {out['status']}, {out['segments']} segments")
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - T0:.1f}s)")
    sys.exit(1 if FAILS else 0)


def anchor(seconds=600.0):
    """Known number: the strict comparison [A > B] for K = 3. The README has a certified floor of
    57.43 (src/lb_pairwise2.py), a design at 57.56 and the exact banded pairwise budget 57.4737,
    so this MILP must land between 57.43 and 57.4737."""
    S = strict_target(3); pmax = 57.57
    tab = Tables(S, pmax)
    breaks = init_breaks(pmax); lb = 0.0; n = 0
    while lb < 57.43 and n < 6:
        left = seconds - (time.time() - T0)
        if left < 20:
            break
        out = solve(S, pmax, breaks, tab, left, symmetric=False)
        lb = max(lb, out["dual_bound"]); n += 1
        print(f"stage {n}: dual bound {out['dual_bound']:.4f} {out['status']}  [{time.time() - T0:.0f}s]", flush=True)
        if "x" not in out or out["status"] != "Optimal":
            break
        breaks = refine(breaks, (np.array(out["x"])[:, None] + np.array(out["y"])[None, :]).ravel(), 2e-4 * pmax)
    check(f"strict comparison, K = 3: MILP bound {lb:.4f} after {n} stages is between 57.43 and 57.4737",
          57.43 <= lb <= 57.4737 + 1e-6)
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - T0:.1f}s)")
    sys.exit(1 if FAILS else 0)


def rounded(bound, pmax):
    """The reported floor: the dual bound minus 5e-5 pmax (more than the slack a 1e-6 integrality
    tolerance leaves on the big-M rows), rounded down to 2 decimals (3 below a cost of 10)."""
    return floor_down(bound, digits=2 if pmax >= 10 else 3, slack=max(5e-5 * pmax, 1e-4))


STATUS = ("certified: HiGHS dual bound of the pairwise relaxation over tabulated float Neyman-Pearson boundaries, "
          "rounded down; valid for any encoder and decoder up to solver tolerances; not an exact certificate; "
          "tightness not proved (compare design_cost)")


def highs_version():
    try:
        from importlib.metadata import version
        return version("highspy")
    except Exception:
        return "unknown"


def run(key, seconds, nref):
    T = float(key); S = target(K, T)
    e = design_cost(key)
    pmax = math.ceil(e["cost"] * 100 + 1) / 100.0          # a design exists at or below this cost
    tab = Tables(S, pmax)
    breaks = seeded_breaks(pmax, (np.array(e["x"])[:, None] + np.array(e["y"])[None, :]).ravel())
    ex, rl, slack = admits(S, e["x"], e["y"], tab, breaks)
    print(f"T={key} pmax {pmax} target pairs with their own H {len(tab.keys)}  design passes exact {ex} relaxed {rl} "
          f"(slack {slack:.4f})  [{time.time() - T0:.0f}s]", flush=True)
    if not (ex and rl):
        raise SystemExit("encoding rejects a known design: bound not established")
    stages = []; best = 0.0
    budget = float(seconds)
    for it in range(nref + 1):
        left = budget - (time.time() - T0)
        if left < 30:
            break
        out = solve(S, pmax, breaks, tab, left, log=True)   # a stage that hits the limit still gives a valid bound
        best = max(best, out["dual_bound"])
        out["best_lb"] = best
        stages.append(out)
        print(json.dumps(out), flush=True)
        if "x" not in out or out["status"] != "Optimal":
            break
        m = (np.array(out["x"])[:, None] + np.array(out["y"])[None, :]).ravel()
        breaks = refine(breaks, m, 2e-4 * pmax)
        ex, rl, slack = admits(S, e["x"], e["y"], tab, breaks)
        if not rl:
            raise SystemExit("refined encoding rejects a known design: bound not established")
        print(f"refined to {len(breaks) - 1} segments  [{time.time() - T0:.0f}s]", flush=True)
    cf = chain_floor(S, EPS)["total"]
    try:
        res = json.load(open(OUT))
    except FileNotFoundError:
        res = {"model": "one detector, pairwise relaxation MILP (see src/soft_floor_shared.py)", "eps": EPS, "T": {}}
    old = res["T"].get(key)
    if old and old["raw_dual_bound"] >= best:               # every run's bound is valid: keep the larger one
        print(f"T={key} keeping the stored bound {old['raw_dual_bound']:.4f} (this run: {best:.4f})", flush=True)
        return
    res["T"][key] = {"T": T, "value": rounded(best, pmax), "raw_dual_bound": best, "pmax": pmax,
                     "design_cost": e["cost"], "chain_floor": floor_down(cf), "stages": stages,
                     "run": {"seconds": seconds, "refinements": nref, "highspy": highs_version()},
                     "status": STATUS}
    res["T"] = dict(sorted(res["T"].items(), key=lambda kv: float(kv[0])))
    dump_json(res, OUT)
    print(f"T={key} floor {res['T'][key]['value']:.3f} (chain floor {floor_down(cf):.4f}, design {e['cost']:.4f})", flush=True)


def show():
    try:
        res = json.load(open(OUT))
    except FileNotFoundError:
        res = {"T": {}}
    for T in TS:
        key = str(T); S = target(K, T)
        cf = floor_down(chain_floor(S, EPS)["total"])
        e = design_cost(key)
        r = res["T"].get(key)
        if r is None:
            check(f"T={key} floor {cf:.4f} design {e['cost']:.4f}", cf <= e["cost"],
                  "one detector MILP bound: not established for this T; the chain floor applies")
            continue
        tab = Tables(S, r["pmax"], max_cells=32)             # coarse cells: this only checks the encoding
        ex, rl, slack = admits(S, e["x"], e["y"], tab, init_breaks(r["pmax"]))
        ok = ex and rl and r["value"] == rounded(r["raw_dual_bound"], r["pmax"]) and r["value"] <= e["cost"] <= r["pmax"]
        top = max(r["stages"], key=lambda st: st["dual_bound"])
        ok &= top["dual_bound"] == r["raw_dual_bound"]
        check(f"T={key} floor {max(cf, r['value']):.4f} design {e['cost']:.4f}", ok,
              f"MILP bound {r['value']:.3f} STORED, not solved again (stage {r['stages'].index(top) + 1} of "
              f"{len(r['stages'])}: {top['status']}, {top['segments']} segments, {top['time']} s) chain floor {cf:.4f}; "
              f"stored design passes the exact pairwise conditions {ex} and the relaxed encoding {rl}")
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - T0:.1f}s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        selfcheck()
    elif "--anchor" in a:
        anchor()
    elif "--run" in a:
        i = a.index("--run")
        run(a[i + 1], float(a[i + 2]) if len(a) > i + 2 else 1200.0, int(a[i + 3]) if len(a) > i + 3 else 3)
    else:
        show()
