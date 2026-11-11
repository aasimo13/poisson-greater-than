"""Shared tools for the sampled (soft) comparison.

Model: A, B in {0..K-1}; level tables x[A] >= 0 and y[B] >= 0; cost P = max(x) + max(y).
One detector: N ~ Poisson(x[A] + y[B] + bg). Two channels: N1 ~ Poisson(x[A]), N2 ~ Poisson(y[B]).
The receiver outputs one bit. The target probability of a 1 is s[A, B] = sigmoid((A - B) / T);
T = 0 is the limit 1 / 0.5 / 0 for A > B / A = B / A < B. A randomized decoder d(counts) in [0, 1]
achieves q[A, B] = E d, and the requirement is |q - s| <= eps on every pair, which for a one bit
output is total variation distance <= eps.

Public API
  target(K, T)                              K x K array s[a, b]
  lp_rows(P, tail, s)                       (t, d): the decoder LP for any hypothesis by outcome matrix
  lp_shared(x, y, S, bg=0.0)                (t, d): best randomized decoder on one count
  lp_two(x, y, S)                           (t, d2, (M1, M2)): best randomized decoder on two counts
  recheck_shared_mp(x, y, S, d, bg=0.0)     mpmath worst |q - s| of a stored decoder (float)
  recheck_two_mp(x, y, S, d2)               same for a two count decoder
  budget_shared(x, y, S, eps, bg=0.0)       (cost, scale): smallest scale of the shape meeting eps
  budget_two(x, y, S, eps)                  (cost, scale)
  budget_near(err, c0, eps)                 the same bisection started from a guess (used in searches)
  nm_polish(cost_of, v0, maxfev)            Nelder-Mead on a level table, restarts, returns (cost, v)
  dump_json(obj, path)                      json with one line per numeric array, repr floats
  pair_feasible(mu, nu, s_mu, s_nu, eps)    exact two hypothesis feasibility (Neyman-Pearson)
  pair_margin, pair_margin_mp, pair_lp_slack, pair_feasible_lp   the same question three ways
  min_hi(lo, s_lo, s_hi, eps)               smallest hi >= lo that is feasible against lo
  chain_feasible(levels, S, eps)            all pairwise constraints of one level table
  chain_floor(S, eps)                       floor for two separable channels (and one detector);
                                            it does NOT bound arrival time designs (see soft_time.py)

Truncation. Counts above the grid are never given to the decoder, so the achieved probability lies
in [P d, P d + tail]. Every LP and every recheck charges the tail both ways: P d + tail <= s + t and
P d >= s - t. Grids are cut where the tail is below 1e-9 per pair.

Usage: imported by the soft_*.py scripts (run from the repo root); self test is src/soft_selftest.py.
"""
import functools
import json
import math
import os
import re
import numpy as np
import mpmath as mp
from scipy.optimize import linprog, minimize
from scipy.special import expit, pdtr, pdtrc, pdtrik
from scipy.stats import poisson

TAIL = 1e-9
TOL = 1e-13
MP_DPS = 40
LP_SCALE = 1e6


def target(K, T):
    """s[a, b] = sigmoid((a - b) / T); T = 0 gives 1 if a > b, 0.5 if a = b, 0 if a < b."""
    if T < 0:
        raise ValueError("T must be >= 0")
    diff = np.arange(K)[:, None] - np.arange(K)[None, :]
    if T == 0:
        return np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    return expit(diff / float(T))


def strict_target(K):
    """The original comparison: 1 if a > b else 0 (ties false)."""
    return np.array([[1.0 if a > b else 0.0 for b in range(K)] for a in range(K)])


# ---------- grids and LPs ----------
def grid_max(lams, tail=TAIL):
    """Largest count kept, so that each Poisson(lam) has at most tail / 4 mass above it."""
    lams = np.asarray(lams, float).ravel()
    pos = lams[lams > 0]
    if len(pos) == 0:
        return 0
    return int(np.max(poisson.isf(tail / 4, pos))) + 2


def pmf_rows(means, M):
    """Poisson pmf on 0..M, one row per mean; a mean of 0 is a point mass at 0."""
    means = np.asarray(means, float).ravel()
    if np.any(means < 0):
        raise ValueError("negative mean")
    P = np.zeros((len(means), M + 1))
    pos = means > 0
    if pos.any():
        P[pos] = poisson.pmf(np.arange(M + 1)[None, :], means[pos, None])
    P[~pos, 0] = 1.0
    return P


def lp_rows(P, tail, s):
    """Generic decoder LP: min t with P d + tail <= s + t, P d >= s - t, 0 <= d <= 1.

    P[h, m] is the probability of outcome m under hypothesis h, tail[h] the mass not on the grid,
    s[h] the target. Returns (t, d). Use it for any other count model (time slots and so on).

    HiGHS drops matrix entries below 1e-9, which on a two count grid costs up to about 1e-7 in t.
    The rows are therefore scaled by LP_SCALE (the solver minimizes LP_SCALE t), which moves that
    cutoff to 1e-15 in probability. The returned t is recomputed in floats from the returned
    (clipped) decoder, so it is the error that decoder really has, not the solver's objective.
    """
    P = np.asarray(P, float); tail = np.asarray(tail, float); s = np.asarray(s, float)
    H, M = P.shape
    up = s < 1.0            # with s = 1 the upper row can never bind
    dn = s > 0.0
    nu = int(up.sum())
    A = np.zeros((nu + int(dn.sum()), M + 1))
    A[:, M] = -1.0
    A[:nu, :M] = P[up]
    A[nu:, :M] = -P[dn]
    b = np.concatenate([(s - tail)[up], -s[dn]])
    c = np.zeros(M + 1); c[M] = 1.0
    bounds = [(0, 1)] * M + [(0, None)]
    A[:, :M] *= LP_SCALE
    r = linprog(c, A_ub=A, b_ub=LP_SCALE * b, bounds=bounds, method="highs")
    if r.status != 0:
        A[:, :M] /= LP_SCALE
        r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    d = np.clip(r.x[:M], 0.0, 1.0)
    q = P @ d
    return float(max(np.max(q + tail - s), np.max(s - q), 0.0)), d


def lp_shared(x, y, S, bg=0.0, tail=TAIL):
    """Best randomized decoder on one count N ~ Poisson(x[a] + y[b] + bg).

    Returns (t, d): t is the worst |q - s| with the truncated tail charged both ways, d[n] is the
    probability of output 1 on count n for n = 0..len(d)-1 (counts above that are the tail).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.asarray(S, float).ravel()
    means = (x[:, None] + y[None, :]).ravel() + bg
    P = pmf_rows(means, grid_max(means, tail))
    tl = np.clip(1.0 - P.sum(axis=1), 0.0, None)
    return lp_rows(P, tl, s)


def lp_two(x, y, S, tail=TAIL):
    """Best randomized decoder on two counts N1 ~ Poisson(x[a]), N2 ~ Poisson(y[b]).

    Returns (t, d2, (M1, M2)) with d2[n1, n2] of shape (M1 + 1, M2 + 1).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.asarray(S, float).ravel()
    M1, M2 = grid_max(x, tail), grid_max(y, tail)
    P1, P2 = pmf_rows(x, M1), pmf_rows(y, M2)
    P = np.einsum("ai,bj->abij", P1, P2).reshape(len(x) * len(y), -1)
    tl = np.clip(1.0 - P.sum(axis=1), 0.0, None)
    t, d = lp_rows(P, tl, s)
    return t, d.reshape(M1 + 1, M2 + 1), (M1, M2)


# ---------- mpmath rechecks ----------
def _mp_pmf(m, M):
    p = [mp.exp(-m)]
    for n in range(1, M + 1):
        p.append(p[-1] * m / n)
    return p


def _mp_decoder(d):
    d = np.asarray(d, float)
    if np.any(d < 0) or np.any(d > 1) or not np.all(np.isfinite(d)):
        raise ValueError("decoder entries must lie in [0, 1]")
    return d


def recheck_shared_mp(x, y, S, d, bg=0.0, full=False):
    """Worst |q - s| of the stored decoder d on one count, mpmath, tail counted against.

    Inputs are taken at their exact float values. Returns a float; with full=True also the K x K
    arrays of P d (output 1 probability on the grid) and of the tail mass.
    """
    d = _mp_decoder(d).ravel()
    S = np.asarray(S, float)
    with mp.workdps(MP_DPS):
        dm = [mp.mpf(float(v)) for v in d]
        worst = mp.mpf(0)
        Q = np.zeros(S.shape); TL = np.zeros(S.shape)
        for a in range(S.shape[0]):
            for b in range(S.shape[1]):
                p = _mp_pmf(mp.mpf(float(x[a])) + mp.mpf(float(y[b])) + mp.mpf(float(bg)), len(dm) - 1)
                q = mp.fdot(p, dm); tl = 1 - mp.fsum(p); s = mp.mpf(float(S[a, b]))
                worst = max(worst, q + tl - s, s - q)
                Q[a, b] = float(q); TL[a, b] = float(tl)
        worst = float(worst)
    return (worst, Q, TL) if full else worst


def recheck_two_mp(x, y, S, d2, full=False):
    """Worst |q - s| of the stored decoder d2[n1, n2] on two counts, mpmath, tail counted against."""
    d2 = _mp_decoder(d2)
    S = np.asarray(S, float)
    M1, M2 = d2.shape[0] - 1, d2.shape[1] - 1
    with mp.workdps(MP_DPS):
        px = [_mp_pmf(mp.mpf(float(v)), M1) for v in x]
        py = [_mp_pmf(mp.mpf(float(v)), M2) for v in y]
        sx = [mp.fsum(p) for p in px]; sy = [mp.fsum(p) for p in py]
        rows = [[mp.mpf(float(v)) for v in row] for row in d2]
        worst = mp.mpf(0)
        Q = np.zeros(S.shape); TL = np.zeros(S.shape)
        for b in range(S.shape[1]):
            v = [mp.fdot(row, py[b]) for row in rows]
            for a in range(S.shape[0]):
                q = mp.fdot(px[a], v); tl = 1 - sx[a] * sy[b]; s = mp.mpf(float(S[a, b]))
                worst = max(worst, q + tl - s, s - q)
                Q[a, b] = float(q); TL[a, b] = float(tl)
        worst = float(worst)
    return (worst, Q, TL) if full else worst


# ---------- budgets ----------
def _budget(err, cost1, eps, cost_cap, rel):
    if err(0.0) <= eps:
        return 0.0, 0.0
    if cost1 <= 0:
        return math.inf, math.inf
    cap = cost_cap / cost1
    lo, hi = 0.0, min(1.0 / cost1, cap)
    while err(hi) > eps:
        if hi >= cap:
            return math.inf, math.inf
        lo, hi = hi, min(2.0 * hi, cap)
    while hi - lo > rel * hi:
        m = 0.5 * (lo + hi)
        if err(m) <= eps:
            hi = m
        else:
            lo = m
    return float(hi * cost1), float(hi)


def budget_shared(x, y, S, eps, bg=0.0, cost_cap=4000.0, rel=1e-6):
    """Smallest scale c with lp_shared(c x, c y, S, bg)[0] <= eps. Returns (cost, scale).

    Bisection is valid because the LP optimal error is nonincreasing in the scale (proved, and
    checked numerically in soft_selftest.py): from a count at scale c' > c the decoder can thin
    each photon with probability c / c' (and add Poisson(bg (1 - c / c')) to restore the
    background), which reproduces the scale c count exactly, so every decoder available at c is
    available at c'. This is a property of the optimal decoder; a fixed decoder has no such
    monotonicity for soft targets. The truncated grid breaks it only at the 1e-9 level.
    Returns (0, 0) if no light is needed and (inf, inf) if the shape fails up to cost_cap.
    eps is compared with the float LP value; pass eps minus a small margin (1e-6) if the stored
    decoder must pass the mpmath recheck strictly.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    return _budget(lambda c: lp_shared(c * x, c * y, S, bg)[0], x.max() + y.max(), eps, cost_cap, rel)


def budget_two(x, y, S, eps, cost_cap=400.0, rel=1e-6):
    """Smallest scale c with lp_two(c x, c y, S)[0] <= eps. Returns (cost, scale).

    Same thinning argument as budget_shared, applied to each count separately. The cap is lower
    because the two count LP has (M1 + 1)(M2 + 1) variables.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    return _budget(lambda c: lp_two(c * x, c * y, S)[0], x.max() + y.max(), eps, cost_cap, rel)


def budget_near(err, c0, eps, rel=1e-6, step=0.02, cap=1e9):
    """Smallest c with err(c) <= eps for a nonincreasing err, bracketed outward from the guess c0.

    Same answer as the bisection in budget_shared / budget_two (to rel), with fewer LP solves when
    c0 is close: the bracket grows geometrically from c0 and is then cut by alternating a secant
    step on log(err) with a plain bisection step. Returns the feasible end of the bracket, or inf
    if no scale up to cap works (err is never evaluated above cap).
    """
    lo = hi = min(float(c0), cap)
    e = err(hi)
    if e <= eps:
        ehi = e
        while True:
            lo = hi * (1.0 - step)
            elo = err(lo)
            if elo > eps or lo <= 0:
                break
            hi, ehi = lo, elo
            step = min(0.5, 2.0 * step)
    else:
        elo = e
        while True:
            if lo >= cap:
                return math.inf
            hi = min(cap, lo * (1.0 + step))
            ehi = err(hi)
            if ehi <= eps:
                break
            lo, elo = hi, ehi
            step = min(1.0, 2.0 * step)
    secant = True
    while hi - lo > rel * hi:
        m = 0.5 * (lo + hi)
        if secant and elo > eps > 0 and ehi > 0 and elo < 1e9:
            w = (math.log(elo) - math.log(eps)) / max(math.log(elo) - math.log(ehi), 1e-300)
            m = lo + min(0.95, max(0.05, w + 0.02)) * (hi - lo)
        secant = not secant
        em = err(m)
        if em <= eps:
            hi, ehi = m, em
        else:
            lo, elo = m, em
    return float(hi)


def nm_polish(cost_of, v0, maxfev, rounds=2, spread=0.03):
    """Nelder-Mead on a nonnegative level table (abs of the variables), restarted `rounds` times
    from its own optimum. cost_of(v) is the budget of the shape v. Returns (cost, v); the result
    is a searched upper bound, never an optimum certificate. maxfev = 0 returns the start."""
    v = np.abs(np.asarray(v0, float))
    best = (cost_of(v), v.copy())
    if maxfev <= 0:                       # no search: just price the starting table
        return best
    for _ in range(rounds):
        n = len(v)
        sim = np.vstack([best[1]] + [best[1] + np.eye(n)[i] * (spread * max(best[1][i], 0.1 * best[1].max()))
                                     for i in range(n)])
        r = minimize(lambda z: cost_of(np.abs(z)), best[1], method="Nelder-Mead",
                     options=dict(initial_simplex=sim, maxfev=max(1, maxfev // rounds), xatol=1e-7, fatol=1e-8))
        if r.fun < best[0]:
            best = (float(r.fun), np.abs(r.x))
        spread *= 0.3
    return best


def dump_json(obj, path):
    """json.dump with indent 1, but every array of numbers on one line (decoders stay readable in
    a diff and floats keep their repr)."""
    txt = json.dumps(obj, indent=1)
    txt = re.sub(r"\[\s*([^\[\]{}\"]*?)\s*\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", txt)
    with open(path + ".tmp", "w") as f:        # write then rename, so a killed run never leaves half a file
        f.write(txt + "\n")
    os.replace(path + ".tmp", path)


# ---------- two hypotheses: exact feasibility ----------
def _pmf(k, m):
    if m <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-m + k * math.log(m) - math.lgamma(k + 1))


def _np_upper(mu, alpha):
    """Upper tail test of size alpha under Poisson(mu): 1 if N > k, with probability g if N = k."""
    if alpha >= 1.0:
        return 0, 1.0
    if mu <= 0:
        return 0, max(alpha, 0.0)
    if alpha <= 0.0:
        return math.inf, 0.0
    g = pdtrik(1.0 - alpha, mu)
    k = int(max(0.0, math.floor(g))) if math.isfinite(g) else int(mu)
    while k > 0 and pdtrc(k - 1, mu) <= alpha:
        k -= 1
    while pdtrc(k, mu) > alpha:
        k += 1
    return k, min(1.0, max(0.0, (alpha - float(pdtrc(k, mu))) / _pmf(k, mu)))


def _upper_power(k, g, nu):
    if k == math.inf:
        return 0.0
    return float(pdtrc(k, nu)) + g * _pmf(k, nu)


def _np_lower(mu, alpha):
    """Lower tail test of size alpha under Poisson(mu): 1 if N < k, with probability g if N = k."""
    if alpha <= 0.0:
        return 0, 0.0
    if mu <= 0:
        return 0, min(alpha, 1.0)
    if alpha >= 1.0:
        return math.inf, 0.0
    g = pdtrik(alpha, mu)
    k = int(max(0.0, math.ceil(g))) if math.isfinite(g) else int(mu)
    while k > 0 and pdtr(k - 1, mu) >= alpha:
        k -= 1
    while pdtr(k, mu) < alpha:
        k += 1
    below = float(pdtr(k - 1, mu)) if k > 0 else 0.0
    return k, min(1.0, max(0.0, (alpha - below) / _pmf(k, mu)))


def _lower_power(k, g, nu):
    if k == math.inf:
        return 1.0
    return (float(pdtr(k - 1, nu)) if k > 0 else 0.0) + g * _pmf(k, nu)


def _box(s, eps):
    return max(0.0, s - eps), min(1.0, s + eps)


def pair_margin(mu, nu, s_mu, s_nu, eps):
    """Signed slack of the two hypothesis problem; feasible exactly when it is >= 0.

    Take mu <= nu (otherwise swap the hypotheses). Over all tests d the points (E_mu d, E_nu d)
    fill the convex region between up(alpha), the power under nu of the randomized upper tail test
    of size alpha under mu, and dn(alpha) = 1 - up(1 - alpha), the lower tail test. Both curves are
    nondecreasing and the diagonal lies between them, so the box [a1, a2] x [b1, b2] (the targets
    plus or minus eps, clipped to [0, 1]) meets the region exactly when up(a2) >= b1 and
    dn(a1) <= b2. The margin is min(up(a2) - b1, b2 - dn(a1)).
    """
    if eps < 0:
        return -math.inf
    if mu > nu:
        mu, nu, s_mu, s_nu = nu, mu, s_nu, s_mu
    a1, a2 = _box(s_mu, eps); b1, b2 = _box(s_nu, eps)
    up = _upper_power(*_np_upper(mu, a2), nu)
    dn = _lower_power(*_np_lower(mu, a1), nu)
    return min(up - b1, b2 - dn)


def pair_feasible(mu, nu, s_mu, s_nu, eps, tol=TOL):
    """Is there a test d(n) in [0, 1] with |E_mu d - s_mu| <= eps and |E_nu d - s_nu| <= eps,
    where E_m is over N ~ Poisson(m)? Exact up to float rounding; tol leans toward "feasible",
    which is the safe side for a lower bound."""
    return pair_margin(float(mu), float(nu), float(s_mu), float(s_nu), float(eps)) >= -tol


def _mp_upper(mu, nu, alpha):
    if alpha >= 1:
        return mp.mpf(1)
    if mu == 0:
        return 1 - (1 - alpha) * mp.exp(-nu)
    if alpha <= 0:
        return mp.mpf(0)
    k = 0
    pm = mp.exp(-mu); cm = pm; pn = mp.exp(-nu); cn = pn
    while 1 - cm > alpha:
        k += 1
        pm = pm * mu / k; cm += pm
        pn = pn * nu / k; cn += pn
        if k > 10 ** 6:
            raise RuntimeError("size too small for the working precision")
    return (1 - cn) + (alpha - (1 - cm)) / pm * pn


def pair_margin_mp(mu, nu, s_mu, s_nu, eps):
    """pair_margin in mpmath (40 digits), by direct summation; inputs at their exact float values."""
    with mp.workdps(MP_DPS):
        if mu > nu:
            mu, nu, s_mu, s_nu = nu, mu, s_nu, s_mu
        mu, nu, s_mu, s_nu, eps = [mp.mpf(float(v)) for v in (mu, nu, s_mu, s_nu, eps)]
        a1, a2 = max(0, s_mu - eps), min(1, s_mu + eps)
        b1, b2 = max(0, s_nu - eps), min(1, s_nu + eps)
        up = _mp_upper(mu, nu, a2)
        dn = 1 - _mp_upper(mu, nu, 1 - a1)
        return float(min(up - b1, b2 - dn))


def pair_lp_slack(mu, nu, s_mu, s_nu, eps):
    """Independent check by LP: the smallest t such that some test meets both targets within
    eps + t. Feasible exactly when the result is <= 0 (up to the LP tolerance, about 1e-9)."""
    M = grid_max([mu, nu], 1e-15) + 5
    P = pmf_rows([mu, nu], M)
    s = np.array([s_mu, s_nu], float)
    A = np.zeros((4, M + 2)); A[:, M + 1] = -1.0
    A[:2, :M + 1] = P; A[2:, :M + 1] = -P
    b = np.concatenate([s + eps, -(s - eps)])
    c = np.zeros(M + 2); c[M + 1] = 1.0
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * (M + 1) + [(-2, None)], method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    return float(r.x[M + 1])


def pair_feasible_lp(mu, nu, s_mu, s_nu, eps, tol=1e-9):
    return pair_lp_slack(mu, nu, s_mu, s_nu, eps) <= tol


@functools.lru_cache(maxsize=1 << 18)
def _min_hi(lo, s_lo, s_hi, eps, tol):
    a1, a2 = _box(s_lo, eps); b1, b2 = _box(s_hi, eps)
    ku, gu = _np_upper(lo, a2)
    kl, gl = _np_lower(lo, a1)

    def ok(hi):
        return _upper_power(ku, gu, hi) >= b1 - tol and _lower_power(kl, gl, hi) <= b2 + tol

    if ok(lo):
        return lo
    a, step = lo, 1.0
    while not ok(lo + step):
        a = lo + step
        step *= 2.0
        if step > 1e7:
            return math.inf
    b = lo + step
    while b - a > 1e-13 * max(1.0, b):
        m = 0.5 * (a + b)
        if ok(m):
            b = m
        else:
            a = m
    return b


def min_hi(lo, s_lo, s_hi, eps, tol=TOL):
    """Smallest hi >= lo with pair_feasible(lo, hi, s_lo, s_hi, eps); lo itself if the two
    targets are within 2 eps of each other, inf if no hi works (only possible with eps = 0).

    Monotonicity in hi. For fixed lo the two tests in pair_margin (upper tail of size a2 and lower
    tail of size a1 under Poisson(lo)) do not depend on hi. The first is a nondecreasing function
    of the count and the second a nonincreasing one, and Poisson(hi) is stochastically increasing
    in hi, so up(a2) is nondecreasing and dn(a1) is nonincreasing in hi. Feasibility therefore
    holds on an interval [min_hi, inf) and bisection finds its left end (to about 1e-13 relative).
    """
    if eps < 0:
        return math.inf
    return _min_hi(float(lo), float(s_lo), float(s_hi), float(eps), float(tol))


# ---------- chain floor ----------
def chain_feasible(levels, S, eps, tol=TOL):
    """Do the levels x[a] meet every pairwise constraint of chain_floor for the row coordinate?
    For the column coordinate pass S.T."""
    S = np.asarray(S, float)
    K, Kb = S.shape
    return all(pair_feasible(levels[a], levels[a2], S[a, b], S[a2, b], eps, tol)
               for a in range(K) for a2 in range(a + 1, K) for b in range(Kb))


def _chain(S, eps):
    K, Kb = S.shape
    best = [math.inf, None, None]

    def need(lo, i, j):
        return max(min_hi(lo, S[i, b], S[j, b], eps) for b in range(Kb))

    def dfs(order, lev):
        if lev[-1] >= best[0]:
            return
        if len(order) == K:
            best[:] = [lev[-1], list(order), list(lev)]
            return
        for j in range(K):
            if j in order:
                continue
            top = lev[-1]
            for i, li in zip(order, lev):
                top = max(top, need(li, i, j))
            if top < best[0]:
                dfs(order + [j], lev + [top])

    for j in range(K):
        dfs([j], [0.0])
    if best[1] is None:
        return math.inf, None, None
    levels = [0.0] * K
    for sym, l in zip(best[1], best[2]):
        levels[sym] = l
    return best[0], best[1], levels


def chain_floor(S, eps):
    """Lower bound on max(x) + max(y) for any two channel design and any decoder.

    It also bounds one detector designs (with or without background), because N = N1 + N2 plus
    independent noise is a randomized function of the two counts. It does NOT bound arrival time
    designs: there a pair is told apart by which slot is lit, not by one scalar Poisson count
    (a flat arrival time design at cost 19.87 meets the T = 0 target, the chain floor is 39.6).

    Necessary condition. Fix b. Under the pairs (a, b) and (a2, b) the count N2 has the same law
    and is independent of N1, so averaging any decoder over N2 (and its own randomness) gives a
    test d'(N1) with the same two output probabilities. Hence for all a != a2 and all b,
    pair_feasible(x[a], x[a2], S[a, b], S[a2, b], eps) must hold. Same for y with pairs (a, b),
    (a, b2).

    Chain. Fix the order of the levels, x[p0] <= x[p1] <= ... and let G0 = 0,
    Gj = max(G(j-1), max over i < j and b of min_hi(Gi, S[pi, b], S[pj, b], eps)). Then any level
    table with that order that meets the necessary condition has x[pj] >= Gj. Induction on j,
    using three facts: (1) levels are >= 0; (2) feasibility is monotone in hi (see min_hi);
    (3) feasibility survives lowering lo: from a count under (lo', hi) with lo' < lo, keep each
    photon with probability p = (hi - lo) / (hi - lo') and add an independent Poisson((1 - p) hi);
    the result is Poisson(lo) or Poisson(hi), so every test of (lo, hi) is a test of (lo', hi)
    (the mirrored recipe, thin by (hi - lo) / (hi' - lo) and add Poisson((1 - p) lo), proves (2)
    again). From
    Gi <= x[pi] <= x[pj] and (3) the pair (Gi, x[pj]) is feasible, so x[pj] >= min_hi(Gi, ...) by
    (2). The greedy table itself meets every pairwise condition by (2), so Gj is the exact optimum
    of the pairwise relaxation for that order. floor_x is the minimum of the top level over all
    K! orders (depth first search over prefixes, pruned by the best top level so far).
    The shift fact (subtracting a constant from all levels keeps feasibility, because adding an
    independent Poisson(c) is a channel) is not needed for the bound beyond (1); it is checked in
    the self test anyway.

    Accuracy: float Poisson tails (scipy.special), bisection to 1e-13, rounding leans toward
    "feasible". Good to about 1e-9, not an exact rational certificate. The bisection returns the
    feasible end, which is the high side, so quote the floor rounded DOWN to at most 4 decimals
    (floor_down below); it is tight for the pairwise relaxation only, not for real designs.

    Returns a dict: floor_x, floor_y, total, levels {"x": [level of each a], "y": [level of each
    b]}, permutation {"x": [symbols from lowest to highest level], "y": [...]}, eps.
    """
    S = np.asarray(S, float)
    fx, px, lx = _chain(S, float(eps))
    fy, py, ly = _chain(S.T, float(eps))
    return {"floor_x": fx, "floor_y": fy, "total": fx + fy,
            "levels": {"x": lx, "y": ly}, "permutation": {"x": px, "y": py}, "eps": float(eps)}


def floor_down(v, digits=4, slack=1e-8):
    """Round a floor down (the safe side for a lower bound) to `digits` decimals, after taking
    off `slack` for the float bisection."""
    if not math.isfinite(v) or v <= 0:
        return v
    q = 10.0 ** digits
    return max(0.0, math.floor((v - slack) * q) / q)
