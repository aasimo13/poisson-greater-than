"""Certified lower bound from the exact pairwise relaxation (convex hull MILP formulation).

Necessary condition for ANY encoder + decoder with worst case error <= 1%:
every TRUE pair mean m_t and FALSE pair mean m_f must satisfy
    max(m_t, m_f) >= h(min(m_t, m_f)),
where h(mu) is the smallest nu at which Poisson(mu) vs Poisson(nu) reaches two point
minimax error 1% (restricting any decoder to those two hypotheses gives such a test).

h is increasing but neither convex nor concave (it has near steps, e.g. 4.60 -> 5.99
between mu = 0.01 and 0.02). Valid underestimator: on each grid cell [g_i, g_{i+1}],
h >= h(g_i) (monotonicity). On a segment [L, U] we take the lower convex hull of that
staircase and use a subset of its facets: phi(mu) = max_f (a_f mu + b_f) <= h(mu).

Each of the K^2 means m_j is split over segments with the multiple choice (convex hull)
formulation: m_j = sum_k mu_jk, L_k z_jk <= mu_jk <= U_k z_jk, sum_k z_jk = 1,
phi_j = sum_k phi_jk, phi_jk >= a_f mu_jk + b_f z_jk. Orientation binaries o choose which
of each true/false pair is the higher one. The MILP optimum (dual bound) is a valid lower
bound on P = max(x) + max(y) for every scheme, up to floating point in h (brentq 1e-10)
and the MILP tolerances.

Usage: python3 lb_pairwise2.py K PMAX TIME_LIMIT N_REFINE
"""
import sys, json, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

import os
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_hg = os.environ.get("HGRID", "hgrid.npy")
if not os.path.exists(_hg):
    _hg = os.path.join(_DATA, os.path.basename(_hg))
G, HG = np.load(_hg)
OBJ = os.environ.get("OBJ", "max")
MARGIN = 1e-6


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


def facets(L, U, nmax=8):
    """Affine functions a*mu + b, each <= h on [L, U] (rigorous given h increasing)."""
    i0 = max(np.searchsorted(G, L, side="right") - 1, 0)
    i1 = min(np.searchsorted(G, U, side="left"), len(G) - 1)
    px = [L]; py = [HG[i0]]                     # h(L) >= h(g_i0), g_i0 <= L
    for i in range(i0, i1):
        a, b = max(G[i], L), min(G[i + 1], U)
        if b <= a:
            continue
        px += [a, b]; py += [HG[i], HG[i]]      # staircase from the left grid point
    hull = lower_hull(px, py)
    segs = []
    for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
        if x2 - x1 < 1e-12:
            continue
        a = (y2 - y1) / (x2 - x1)
        segs.append((a, y1 - a * x1 - MARGIN, x2 - x1))
    if not segs:
        return [(0.0, HG[i0] - MARGIN)]
    if len(segs) > nmax:
        # keep facets spread over the segment (any subset stays a valid underestimator)
        cum = np.cumsum([s[2] for s in segs]); cum /= cum[-1]
        keep = sorted(set(int(np.searchsorted(cum, q)) for q in np.linspace(0, 1, nmax)))
        keep = [min(k, len(segs) - 1) for k in keep]
        segs = [segs[k] for k in sorted(set(keep))]
    return [(a, b) for a, b, _ in segs]


def solve(K, PMAX, breaks, time_limit=600, verbose=False, cut=None):
    means = [(a, b) for a in range(K) for b in range(K)]
    J = len(means)
    T = [j for j, (a, b) in enumerate(means) if a > b]
    F = [j for j, (a, b) in enumerate(means) if a <= b]
    segs = list(zip(breaks[:-1], breaks[1:]))
    S = len(segs)
    fac = [facets(L, U) for L, U in segs]
    HMAX = float(np.interp(PMAX, G, HG)) + 1.0

    nv = 0
    def alloc(n):
        nonlocal nv
        s = nv; nv += n; return s
    ix = alloc(K); iy = alloc(K); iX = alloc(1); iY = alloc(1)
    imu = alloc(J * S); iz = alloc(J * S); iphi = alloc(J * S)
    pairs = [(t, f) for t in T for f in F]
    io = alloc(len(pairs)); iw = alloc(K)
    MU = lambda j, k: imu + j * S + k
    Z = lambda j, k: iz + j * S + k
    PH = lambda j, k: iphi + j * S + k

    R = []; C = []; V = []; lo = []; hi = []
    def add(coefs, l, u):
        r = len(lo)
        for c, v in coefs:
            R.append(r); C.append(c); V.append(v)
        lo.append(l); hi.append(u)

    for a in range(K):
        add([(ix + a, 1), (iX, -1)], -np.inf, 0)
        add([(iy + a, 1), (iY, -1)], -np.inf, 0)
    add([(iX, 1), (iY, 1)], -np.inf, PMAX)
    for j, (a, b) in enumerate(means):
        add([(ix + a, 1), (iy + b, 1)] + [(MU(j, k), -1) for k in range(S)], 0, 0)
        add([(Z(j, k), 1) for k in range(S)], 1, 1)
        for k, (L, U) in enumerate(segs):
            add([(MU(j, k), 1), (Z(j, k), -L)], 0, np.inf)
            add([(MU(j, k), 1), (Z(j, k), -U)], -np.inf, 0)
            for (al, be) in fac[k]:
                add([(PH(j, k), 1), (MU(j, k), -al), (Z(j, k), -be)], 0, np.inf)
    for p, (t, f) in enumerate(pairs):
        # o=1: false mean is higher:  m_f >= phi_t   (relaxed by HMAX when o=0)
        add([(ix + means[f][0], 1), (iy + means[f][1], 1)] +
            [(PH(t, k), -1) for k in range(S)] + [(io + p, -HMAX)], -HMAX, np.inf)
        # o=0: true mean is higher:   m_t >= phi_f   (relaxed when o=1)
        add([(ix + means[t][0], 1), (iy + means[t][1], 1)] +
            [(PH(f, k), -1) for k in range(S)] + [(io + p, HMAX)], 0, np.inf)
    # shift normalization: some x_a = 0 (shifting x down and y up keeps every mean and P)
    add([(iw + a, 1) for a in range(K)], 1, 1)
    for a in range(K):
        add([(ix + a, 1), (iw + a, PMAX)], -np.inf, PMAX)
    if os.environ.get("MONO") == "1":      # x nonincreasing, y nondecreasing
        for a in range(K - 1):
            add([(ix + a, 1), (ix + a + 1, -1)], 0, np.inf)
            add([(iy + a + 1, 1), (iy + a, -1)], 0, np.inf)
    if cut is not None:
        add([(iX, 1), (iY, 1)], cut, np.inf)

    A = coo_matrix((V, (R, C)), shape=(len(lo), nv)).tocsr()
    cost = np.zeros(nv)
    if OBJ == "avg":       # mean over the K^2 pairs of x_a + y_b = mean(x) + mean(y)
        cost[ix:ix + K] = 1.0 / K; cost[iy:iy + K] = 1.0 / K
    else:
        cost[iX] = 1; cost[iY] = 1
    integ = np.zeros(nv)
    integ[iz:iz + J * S] = 1; integ[io:io + len(pairs)] = 1; integ[iw:iw + K] = 1
    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    ub[ix:iY + 1] = PMAX; ub[imu:imu + J * S] = PMAX
    ub[iz:iz + J * S] = 1; ub[io:io + len(pairs)] = 1; ub[iw:iw + K] = 1
    ub[iphi:iphi + J * S] = HMAX
    t0 = time.time()
    if os.environ.get("USE_HIGHS") == "1":
        return solve_highs(A, lo, hi, cost, lb, ub, integ, time_limit, K, ix, iy, PMAX, S, t0)
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": 1e-6})
    out = {"K": K, "PMAX": PMAX, "S": S, "status": int(res.status),
           "msg": str(res.message)[:80], "time": round(time.time() - t0, 1)}
    if res.x is not None:
        x = res.x[ix:ix + K]; y = res.x[iy:iy + K]
        out.update(primal=float(res.fun), x=x.round(4).tolist(), y=y.round(4).tolist())
    db = getattr(res, "mip_dual_bound", None)
    out["dual_bound"] = None if db is None or not np.isfinite(db) else float(db)
    if res.status == 2:
        out["dual_bound"] = PMAX       # infeasible below PMAX: every scheme needs P > PMAX
    return out, res


def solve_highs(A, lo, hi, cost, lb, ub, integ, time_limit, K, ix, iy, PMAX, S, t0):
    import highspy
    INF = highspy.kHighsInf
    fix = lambda v: np.where(np.isinf(v), np.sign(v) * INF, v)
    Ac = A.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = A.shape[1]; lp.num_row_ = A.shape[0]
    lp.col_cost_ = cost; lp.col_lower_ = fix(lb); lp.col_upper_ = fix(ub)
    lp.row_lower_ = fix(np.array(lo, float)); lp.row_upper_ = fix(np.array(hi, float))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Ac.indptr; lp.a_matrix_.index_ = Ac.indices; lp.a_matrix_.value_ = Ac.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                       for v in integ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-6)
    h.passModel(lp)
    h.run()
    info = h.getInfo(); st = h.modelStatusToString(h.getModelStatus())
    out = {"K": K, "PMAX": PMAX, "S": S, "status": st, "time": round(time.time() - t0, 1),
           "dual_bound": float(info.mip_dual_bound)}
    if info.primal_solution_status == 2:
        x = np.array(h.getSolution().col_value)
        out.update(primal=float(info.objective_function_value),
                   x=x[ix:ix + K].round(4).tolist(), y=x[iy:iy + K].round(4).tolist())
    if st == "Infeasible":
        out["dual_bound"] = "infeasible"
    return out, None


def refine(breaks, pts, factor=3, minw=0.02):
    breaks = list(breaks); new = set(breaks)
    for p in pts:
        k = int(np.clip(np.searchsorted(breaks, p, side="right") - 1, 0, len(breaks) - 2))
        L, U = breaks[k], breaks[k + 1]
        if U - L < minw:
            continue
        for i in range(1, factor):
            new.add(round(L + (U - L) * i / factor, 6))
    return sorted(new)


def init_breaks(PMAX):
    b = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4.6, 6, 8, 10, 13, 16, 20, 25, 30, 36,
         43, 50, 60, 70, 80, 90, 100, 115, 130, 150, 175, 200, 230, 260, 300, 350, 400]
    return [x for x in b if x < PMAX] + [PMAX]


if __name__ == "__main__":
    K = int(sys.argv[1]); PMAX = float(sys.argv[2])
    tl = float(sys.argv[3]); nref = int(sys.argv[4])
    breaks = init_breaks(PMAX)
    best_lb = 0.0
    for it in range(nref + 1):
        out, res = solve(K, PMAX, breaks, tl)
        if isinstance(out["dual_bound"], float) and np.isfinite(out["dual_bound"]):
            best_lb = max(best_lb, out["dual_bound"])
        out["best_lb"] = best_lb
        print(json.dumps(out), flush=True)
        if "x" not in out:
            break
        x = np.array(out["x"]); y = np.array(out["y"])
        breaks = refine(breaks, (x[:, None] + y[None, :]).ravel())
