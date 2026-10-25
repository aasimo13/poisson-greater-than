"""Best single randomized threshold decoder over ALL encoders, with a certified lower bound.

Theory. A threshold rule says "true" when N is low (or when N is high), with randomization at the
cut. Its worst case error is e2(L_T, L_F): the two point minimax error between the largest true mean
L_T and the smallest false mean L_F ("low counts mean true"; the roles flip for "high counts mean true").
Write s = L_T and u = L_F / L_T > 1. The cheapest encoder reaching that (s, u) costs
    P = s * g(u),   low  true: g(u) = K u - (K - 1),        feasible for 1 < u <= 1 + 1/(K-2)
                    high true: g(u) = (K-1) u - (K - 2),     feasible for 1 < u <= 1 + 1/(K-1)
(for high true, s is the largest FALSE mean and u s the smallest TRUE mean).
s*(u) = smallest s with e2(s, u s) <= 1% is nonincreasing in u and g is increasing, so on [u1, u2]
    P >= s*(u2) * g(u1).
Adaptive bisection of (1, umax] until every interval's bound reaches (claimed minimum - tol) gives a
certificate (up to floating point in e2 and brentq).

Usage:
    python3 src/threshold_optimum.py            # K=4 both directions, certificate, cross check
    python3 src/threshold_optimum.py table      # K=3..10, low-true direction (also prints high-true)
"""
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from core import e2, budget_for_shape, TARGET

_cache = {}


def s_star(u):
    """Smallest s with e2(s, u*s) <= TARGET (e2 is nonincreasing in s along a ray)."""
    key = round(u, 14)
    if key in _cache:
        return _cache[key]
    f = lambda s: e2(s, u * s) - TARGET
    lo, hi = 0.5, 8.0
    while f(hi) > 0:
        lo, hi = hi, hi * 2
    s = brentq(f, lo, hi, xtol=1e-10, rtol=1e-13)
    _cache[key] = s
    return s


def g_low(u, K):
    return K * u - (K - 1)


def g_high(u, K):
    return (K - 1) * u - (K - 2)


def setup(K, direction):
    if direction == "low":
        return (lambda u: g_low(u, K)), 1 + 1.0 / (K - 2)
    return (lambda u: g_high(u, K)), 1 + 1.0 / (K - 1)


def minimize_P(K, direction, n_scan=240):
    g, umax = setup(K, direction)
    f = lambda u: s_star(u) * g(u)
    us = np.linspace(1.02, umax, n_scan)
    vals = np.array([f(u) for u in us])
    i = int(np.argmin(vals))
    best_u, best_v = us[i], vals[i]
    lo, hi = us[max(i - 1, 0)], us[min(i + 1, n_scan - 1)]
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    if r.fun < best_v:
        best_u, best_v = r.x, r.fun
    fe = f(umax)                      # the corner
    if fe <= best_v + 1e-12:
        best_u, best_v = umax, fe
    return best_u, s_star(best_u), best_v, umax


def certify(K, direction, claimed, tol, u_start=1.0 + 1e-9, max_intervals=200000):
    """Adaptive bisection. Returns (worst interval bound, number of intervals, ok)."""
    g, umax = setup(K, direction)
    stack = [(u_start, umax)]
    n = 0
    worst = np.inf
    while stack:
        u1, u2 = stack.pop()
        b = s_star(u2) * g(u1)
        n += 1
        if b >= claimed - tol:
            worst = min(worst, b)
            continue
        if n > max_intervals or u2 - u1 < 1e-13:
            return b, n, False
        m = 0.5 * (u1 + u2)
        stack.append((u1, m)); stack.append((m, u2))
    return worst, n, True


def run_k4():
    K = 4
    res = {}
    for direction, label in (("low", "low counts mean true"), ("high", "high counts mean true")):
        u, s, P, umax = minimize_P(K, direction)
        corner = "corner" if abs(u - umax) < 1e-9 else "interior"
        print(f"K=4 {label}: min P = {P:.3f} at u = {u:.6f} ({corner}, umax {umax:.6f}), s = {s:.4f}")
        res[direction] = P
    # certificate for the low direction, tolerance as in the write up
    claimed = res["low"]
    tol = 0.0035
    worst, n, ok = certify(4, "low", claimed, tol)
    print(f"Certificate (low true): {n} intervals, every interval bound >= {worst:.4f}; "
          f"claimed {claimed:.4f} minus tol {tol} -> {'OK' if ok else 'FAILED'}")
    print(f"Certified lower bound for any threshold design (low true): P >= {worst:.4f}")
    worst_h, n_h, ok_h = certify(4, "high", res["high"], tol)
    print(f"Certificate (high true): {n_h} intervals, lower bound {worst_h:.4f} -> {'OK' if ok_h else 'FAILED'}")
    # cross check with the LP decoder on the corner encoder x = [1.5,1,0.5,0]*s, y reversed
    u, s, P, umax = minimize_P(K, "low")
    x = np.array([1.5, 1.0, 0.5, 0.0]); y = x[::-1]
    Pb, sc = budget_for_shape(x, y)
    print(f"Cross check: core.budget_for_shape on the corner encoder gives {Pb:.3f} (scale = L_T = {sc:.3f})")
    return res


def run_table():
    print(" K   low-true P     argmin u (umax)        s      | high-true P")
    for K in range(3, 11):
        u, s, P, umax = minimize_P(K, "low")
        uh, sh, Ph, umh = minimize_P(K, "high")
        print(f"{K:2d}  {P:10.2f}   {u:.6f} ({umax:.6f})  {s:8.3f}   | {Ph:10.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        run_table()
    else:
        run_k4()
