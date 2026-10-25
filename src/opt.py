"""Shape re-optimization under gain drift or an average photon objective.

Families:
  free     : any x, y >= 0
  monotone : x nonincreasing, y nondecreasing (the threshold family; decoder still unrestricted)
Models:
  nominal  : N ~ Poisson(x[A] + y[B])
  common d : both transmitters share an unknown gain g in [1-d, 1+d] (detector or loss drift)
  indep  d : each transmitter has its own unknown gain gx, gy in [1-d, 1+d]
The decoder is the robust LP decoder: worst case error <= 1% over all pairs AND all gains
in the grid. Final designs are re-verified on a fine gain grid and rescaled if needed.
Costs:
  max : worst case max mean = (max x + max y)(1 + d)
  avg : mean over the 16 pairs of x[A] + y[B] (nominal gain)

Usage: python3 opt.py FAMILY MODEL D COST SEED NSTARTS MAXFEV
"""
import sys, json, time
import numpy as np
from scipy.optimize import minimize, brentq
from core import labels, lp_minimax, TARGET

import os
K = 4
LAB = labels(K)
# background counts per window (dark + ambient); not scaled by the signal gain
BG = float(os.environ.get("BG", "0"))


def gain_grid(model, d, fine=False):
    if model == "nominal" or d == 0:
        return [(1.0, 1.0)]
    if model == "common":
        n = 21 if fine else 5
        return [(g, g) for g in np.linspace(1 - d, 1 + d, n)]
    n = 11 if fine else 3
    gs = np.linspace(1 - d, 1 + d, n)
    return [(a, b) for a in gs for b in gs]


def hyps(x, y, grid):
    means = []; lab = []
    for gx, gy in grid:
        m = (gx * x[:, None] + gy * y[None, :]).ravel() + BG
        means.append(m); lab.append(LAB)
    return np.concatenate(means), np.concatenate(lab)


def err_at_scale(x, y, s, grid):
    m, l = hyps(s * x, s * y, grid)
    return lp_minimax(m, l)


def min_scale(x, y, grid, s0=None, rel=1e-5):
    """Smallest s with robust error <= TARGET (error is nonincreasing in s)."""
    if s0 is None:
        s0 = 100.0 / max(x.max() + y.max(), 1e-9)
    lo, hi = s0 * 0.8, s0 * 1.25
    while err_at_scale(x, y, hi, grid) > TARGET:
        lo, hi = hi, hi * 1.6
        if hi > 1e6:
            return np.inf
    while err_at_scale(x, y, lo, grid) <= TARGET:
        hi, lo = lo, lo / 1.6
        if lo < 1e-9:
            return lo
    f = lambda s: np.log(max(err_at_scale(x, y, s, grid), 1e-300)) - np.log(TARGET)
    try:
        s = brentq(f, lo, hi, xtol=rel * hi)
    except ValueError:
        s = hi
    # make sure we land on the feasible side
    while err_at_scale(x, y, s, grid) > TARGET:
        s *= 1 + rel
    return s


def shape_from(p, family):
    p = np.abs(np.asarray(p, float))
    if family == "free":
        return p[:K], p[K:]
    # monotone: x nonincreasing, y nondecreasing, built from nonnegative increments
    x = np.cumsum(p[:K])[::-1]          # x[K-1] smallest
    y = np.cumsum(p[K:])                 # y[0] smallest
    return x, y


def to_params(x, y, family):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if family == "free":
        return np.concatenate([x, y])
    xr = x[::-1]
    return np.concatenate([np.diff(np.concatenate([[0], xr])), np.diff(np.concatenate([[0], y]))])


def cost_of(x, y, s, d, cost):
    if cost == "max":
        return s * (x.max() + y.max()) * (1 + d)
    return s * (x.mean() + y.mean())


class Obj:
    def __init__(self, family, model, d, cost):
        self.family, self.model, self.d, self.cost = family, model, d, cost
        self.grid = gain_grid(model, d)
        self.s_prev = {}
        self.best = (np.inf, None)
        self.nev = 0

    def __call__(self, p):
        x, y = shape_from(p, self.family)
        norm = x.max() + y.max()
        if norm <= 0:
            return 1e9
        x, y = x / norm, y / norm
        s0 = self.s_prev.get("s", 200.0)
        s = min_scale(x, y, self.grid, s0)
        if not np.isfinite(s):
            return 1e9
        self.s_prev["s"] = s
        c = cost_of(x, y, s, self.d, self.cost)
        self.nev += 1
        if s * (x.max() + y.max()) > 2500:      # far from any useful design; skip big LPs
            return 1e9
        if c < self.best[0]:
            self.best = (c, (x * s, y * s))
        return c


def starts(family, rng, n):
    band = (np.array([98.3, 4.6, 0, 28.8]), np.array([28.8, 0, 4.6, 98.3]))
    thr = (np.array([1.5, 1, 0.5, 0]), np.array([0, 0.5, 1, 1.5]))
    out = []
    extra = os.environ.get("EXTRA")
    if extra:
        import json as _j
        for ex in _j.loads(extra):
            out.append((np.array(ex[0], float), np.array(ex[1], float)))
    if family == "free":
        out += [band, thr]
    else:
        out += [thr]
    while len(out) < n:
        if family == "free":
            x = np.exp(rng.uniform(-3, 0, K)) * (rng.random(K) > 0.2)
            y = np.exp(rng.uniform(-3, 0, K)) * (rng.random(K) > 0.2)
        else:
            x = np.sort(rng.random(K))[::-1]; y = np.sort(rng.random(K))
        out.append((x, y))
    return out[:n]


def verify(x, y, model, d):
    """Re-check on the fine grid; rescale until the fine grid passes."""
    fine = gain_grid(model, d, fine=True)
    norm = x.max() + y.max()
    xs, ys = x / norm, y / norm
    s = norm
    e = err_at_scale(xs, ys, s, fine)
    if e > TARGET:
        s = min_scale(xs, ys, fine, s)
        e = err_at_scale(xs, ys, s, fine)
    return xs * s, ys * s, e


if __name__ == "__main__":
    family, model, d, cost = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    seed = int(sys.argv[5]); nst = int(sys.argv[6]); maxfev = int(sys.argv[7])
    rng = np.random.default_rng(seed)
    t0 = time.time()
    results = []
    for i, (x0, y0) in enumerate(starts(family, rng, nst)):
        ob = Obj(family, model, d, cost)
        p0 = to_params(x0, y0, family) + 1e-3
        r = minimize(ob, p0, method="Nelder-Mead",
                     options={"maxfev": maxfev, "xatol": 1e-5, "fatol": 1e-4, "adaptive": True})
        # one restart from the best point (Nelder-Mead stalls)
        bx, by = ob.best[1]
        r = minimize(ob, to_params(bx, by, family) + 1e-4, method="Nelder-Mead",
                     options={"maxfev": maxfev // 2, "xatol": 1e-6, "fatol": 1e-5, "adaptive": True})
        bx, by = ob.best[1]
        vx, vy, ve = verify(bx, by, model, d)
        c = (vx.max() + vy.max()) * (1 + d) if cost == "max" else (vx.mean() + vy.mean())
        rec = {"family": family, "model": model, "d": d, "bg": BG, "cost": cost, "start": i,
               "search_cost": float(ob.best[0]), "verified_cost": float(c),
               "verified_err": float(ve), "x": vx.round(3).tolist(), "y": vy.round(3).tolist(),
               "max_mean": float(vx.max() + vy.max()), "avg_mean": float(vx.mean() + vy.mean()),
               "nev": ob.nev, "time": round(time.time() - t0, 1)}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    best = min(results, key=lambda r: r["verified_cost"])
    print("BEST " + json.dumps(best), flush=True)
