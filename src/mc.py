"""Molecular communication variants of the A > B problem.

Receivers:
  shared : one molecule type, counts add:            N  = Pois(x[A] + y[B])        (the photon result)
  cancel : two types that annihilate pairwise:      D  = N1 - N2, N1 ~ Pois(x[A]), N2 ~ Pois(y[B])  (Skellam)
  two    : two types counted separately:            (N1, N2)
Cost: worst case expected molecules arriving, max over pairs of x[A] + y[B] = max(x) + max(y).
Requirement: worst case error <= 1% over the 16 pairs, any randomized decoder (LP).
"""
import sys, json, time, os
import numpy as np
from scipy.stats import poisson
from scipy.optimize import linprog, minimize
from core import labels, TARGET

K = 4
LAB = labels(K)
PAIRS = [(a, b) for a in range(K) for b in range(K)]
# Inter-symbol interference: a fraction ISI of the previous message's molecules arrives in this window.
# The previous pair is uniform and unknown to the receiver, so each row is a mixture over it.
ISI = float(os.environ.get("ISI", "0"))
# Decision feedback: the receiver conditions on what it knows about the previous message.
#   none   : nothing (mixture over all 16 previous pairs)
#   answer : its previous answer [A' > B'] (assumed correct) -> two conditioning sets
#   genie  : the exact previous pair (best case for any feedback scheme) -> 16 sets
FEEDBACK = os.environ.get("FEEDBACK", "none")
if FEEDBACK == "answer":
    PREV_SETS = [[p for p in PAIRS if p[0] > p[1]], [p for p in PAIRS if p[0] <= p[1]]]
elif FEEDBACK == "genie":
    PREV_SETS = [[p] for p in PAIRS]
else:
    PREV_SETS = [PAIRS]


def lp_err(P):
    """P: hypotheses x observations (rows sum ~1). Minimax error of the best randomized decoder."""
    H, M = P.shape
    c = np.zeros(M + 1); c[-1] = 1
    A = np.zeros((H, M + 1)); b = np.zeros(H)
    T = LAB == 1
    A[T, :M] = -P[T]; A[T, -1] = -1; b[T] = -1
    A[~T, :M] = P[~T]; A[~T, -1] = -1
    for method in ("highs", "highs-ipm", "highs-ds"):
        r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * M + [(0, 1)], method=method)
        if r.status == 0 and r.x is not None:
            return r.x[-1]
    return 1.0      # solver failed: treat this scale as failing (conservative, pushes the budget up)


def pm(mu, n):
    p = poisson.pmf(np.arange(n), mu)
    p[-1] += max(0.0, 1 - p.sum())
    return p


def mix1(level, prev, n):
    """Count distribution for one source level plus ISI from a uniformly random previous level."""
    if ISI == 0:
        return pm(level, n)
    return np.mean([pm(level + ISI * p, n) for p in prev], axis=0)


def obs_matrix(x, y, kind, prev=None):
    """Observation distributions per current pair, mixed over previous pairs in `prev` (uniform)."""
    prev = PAIRS if prev is None else prev
    mx = (x.max() + y.max()) * (1 + ISI)
    n = int(mx + 10 * np.sqrt(mx + 1) + 30)
    if kind == "shared":
        pm_ = [x[a] + y[b] for a, b in prev]
        return np.vstack([mix1(x[a] + y[b], pm_, n) for a, b in PAIRS])
    n1 = int(x.max() * (1 + ISI) + 10 * np.sqrt(x.max() * (1 + ISI) + 1) + 30)
    n2 = int(y.max() * (1 + ISI) + 10 * np.sqrt(y.max() * (1 + ISI) + 1) + 30)
    if len(prev) == len(PAIRS) or ISI == 0:
        # previous A and B independent and uniform: the mixture factorizes
        comps = {(a, b): [(mix1(x[a], x, n1), mix1(y[b], y, n2), 1.0)] for a, b in PAIRS}
    else:
        comps = {(a, b): [(pm(x[a] + ISI * x[ap], n1), pm(y[b] + ISI * y[bp], n2), 1.0 / len(prev))
                          for ap, bp in prev] for a, b in PAIRS}
    if kind == "two":
        rows = []
        for a, b in PAIRS:
            rows.append(sum(w * np.outer(p1, p2) for p1, p2, w in comps[(a, b)]))
        s1 = np.nonzero(np.max([r.sum(1) for r in rows], axis=0) > 1e-12)[0]
        s2 = np.nonzero(np.max([r.sum(0) for r in rows], axis=0) > 1e-12)[0]
        P = np.vstack([r[np.ix_(s1, s2)].ravel() for r in rows])
        keep = P.max(0) > 1e-12
        return P[:, keep]
    if kind == "cancel":
        return np.vstack([sum(w * np.convolve(p1, p2[::-1]) for p1, p2, w in comps[(a, b)]) for a, b in PAIRS])
    raise ValueError(kind)


def budget(x, y, kind, rel=1e-5):
    x = np.asarray(x, float); y = np.asarray(y, float)
    c = x.max() + y.max(); x, y = x / c, y / c
    # The receiver observes (feedback, count) jointly; the previous pair is uniform, so each
    # conditioning set enters with weight |set| / 16 and the error is averaged over previous pairs,
    # exactly as in the no-feedback case. Feedback can only help under this rule.
    f = lambda s: lp_err(np.hstack([len(prev) / len(PAIRS) * obs_matrix(s * x, s * y, kind, prev)
                                    for prev in PREV_SETS]))
    lo, hi = 1.0, 200.0
    while f(hi) > TARGET:
        lo, hi = hi, hi * 2
        if hi > (1800 if kind == "two" else 20000):
            return np.inf
    while hi - lo > rel * hi:
        m = 0.5 * (lo + hi)
        if f(m) <= TARGET: hi = m
        else: lo = m
    return hi


def search(kind, starts, maxfev):
    best = (np.inf, None)
    for x0, y0 in starts:
        p0 = np.sqrt(np.concatenate([x0, y0]) + 1e-3)
        def obj(p):
            v = p ** 2
            x, y = v[:K], v[K:]
            if x.max() + y.max() <= 0: return 1e9
            return budget(x, y, kind, rel=2e-4)
        r = minimize(obj, p0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-3, "adaptive": True})
        v = r.x ** 2
        if r.fun < best[0]:
            best = (r.fun, (v[:K], v[K:]))
        print(json.dumps({"kind": kind, "start": [list(np.round(x0, 3)), list(np.round(y0, 3))],
                          "P": round(float(r.fun), 3), "x": list(np.round(v[:K] / (v[:K].max() + v[K:].max()) * r.fun, 3)),
                          "y": list(np.round(v[K:] / (v[:K].max() + v[K:].max()) * r.fun, 3))}), flush=True)
    return best


if __name__ == "__main__":
    kind = sys.argv[1]; maxfev = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    inc = np.array([0, 1, 2, 3.0]); dec = inc[::-1].copy()
    band = np.array([98.3, 4.6, 0, 28.8]); bandr = band[::-1].copy()
    starts = {
        "shared": [(dec, inc), (band, bandr)],
        "cancel": [(inc, inc), (np.array([0, 1, 3, 6.0]), np.array([0, 1, 3, 6.0])), (band, bandr)],
        "two": [(np.array([0, 4, 16, 36.0]), np.array([0, 4, 16, 36.0])), (np.array([0, 5.7, 25, 58.1]), np.array([0, 6.8, 26.7, 57.6])), (np.array([0, 1, 3, 6.0]), np.array([0, 1, 3, 6.0]))],
    }[kind]
    if os.environ.get("EXTRA"):
        starts = [(np.array(e[0], float), np.array(e[1], float)) for e in json.loads(os.environ["EXTRA"])] + starts
    if os.environ.get("NSTART"):
        starts = starts[:int(os.environ["NSTART"])]
    t = time.time(); b = search(kind, starts, maxfev)
    print("BEST", kind, "ISI", ISI, "FEEDBACK", FEEDBACK, round(float(b[0]), 3), round(time.time() - t, 1), "s", flush=True)
