"""Closed forms for the sampled comparison, each checked against the numerics.

Model: as in src/soft_core.py (K = 4 unless stated, target s = sigmoid((A - B) / T)). The error
target is fixed at eps = 0.01; an EPS environment variable is ignored, and the checks below make
no claim for other eps. Every formula is derived in its docstring, evaluated, and compared with a
number computed by other code (LP, chain floor, stored designs, the strict comparison scripts).
Exact forms must agree to rounding; approximations and fits report their relative error and are
labeled as such. Nothing here is a new bound.

  F1  temperature above which no light is needed (exact)
  F2  first step of the Neyman-Pearson chain from a dark level (exact)
  F3  two channel pairwise floor for K = 2 (exact value of that floor; a design attains it only
      near T = 2)
  F4  two channels, one table for both, T = 0: Skellam chain (exact optimum of that family),
      Gaussian recursion (approximation, 3% high)
  F5  one detector ramp, T = 0: one randomized threshold (exact for that family), square root rule
      (approximation, 5% low)
  F6  arrival time floor (the closed form of soft_time.disjoint_floor, restated, not an
      independent check)
  F7  strict comparison, threshold design, K = 3..10: 2 (K - 1) z^2 / (sqrt(K - 1) - sqrt(K - 2))^2
      (approximation, within 0.1% at eps 0.01)
  F8  strict comparison, peeled band construction, K = 3..8: C_K = 2 (sqrt(C_(K-1)) + z)^2
      (approximation with the exact start 2 ln 99; a fit with the start 2 ln 100)

It also assembles results/soft_summary.json: per design class and temperature the best stored
cost, family, mpmath deviation and floor (read from results/soft_shared.json, soft_twochan.json,
soft_time.json, soft_floor_shared.json), the strict comparison numbers from results/ladder.json and
the errors of the forms above. The summary only copies stored numbers: run the four soft_*.py
verifiers for the checks of the designs and floors themselves.

Usage (repo root): python3 src/closed_forms.py        (about 10 seconds; reads the four soft result files and
                                                       ladder.json, writes results/soft_closed_forms.json and
                                                       results/soft_summary.json, or the paths in the environment
                                                       variables SOFT_CLOSED_OUT and SOFT_SUMMARY_OUT)
"""
import json
import math
import os
import sys
import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, pdtr, pdtrc
from scipy.stats import norm, skellam
from soft_core import target, strict_target, lp_shared, lp_two, min_hi, chain_floor, floor_down, dump_json
from soft_time import disjoint_floor

os.environ.pop("EPS", None)        # core.py reads EPS at import; everything here is for eps = 0.01
import banded
import threshold_optimum
from core import budget_for_shape

EPS = 0.01
OUT = os.environ.get("SOFT_CLOSED_OUT", "results/soft_closed_forms.json")
SUMMARY = os.environ.get("SOFT_SUMMARY_OUT", "results/soft_summary.json")
TS = [0, 0.25, 0.5, 1, 2, 4]
FAILS = []
T0 = time.time()
RES = {"eps": EPS, "forms": {}}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def stored(path, key, family):
    return json.load(open(path))["T"][key]["families"][family]["cost"]      # a missing file or row is an error


def rel(a, b):
    return abs(a - b) / abs(b)


# ---------- F1 ----------
def t_star(K, eps):
    """No light gives every count 0, so q is one constant and the best is 1/2. That meets the
    target iff sigmoid((K - 1) / T) <= 1/2 + eps, i.e. T >= (K - 1) / ln((1/2 + eps) / (1/2 - eps))."""
    return (K - 1) / math.log((0.5 + eps) / (0.5 - eps))


def f1():
    ts = t_star(4, EPS)
    z = np.zeros(4)
    above = [lp_shared(z, z, target(4, ts * 1.001))[0], lp_two(z, z, target(4, ts * 1.001))[0]]
    below = [lp_shared(z, z, target(4, ts * 0.999))[0], lp_two(z, z, target(4, ts * 0.999))[0]]
    fl = [chain_floor(target(4, ts * 1.001), EPS)["total"]]
    ok = max(above) <= EPS and min(below) > EPS and fl[0] == 0.0 and abs(expit(3 / ts) - 0.5 - EPS) < 1e-15
    RES["forms"]["F1_no_light_temperature"] = {"T_star": ts, "formula": "(K - 1) / ln((1/2 + eps) / (1/2 - eps))",
                                                "status": "exact", "lp_error_dark_just_above": max(above),
                                                "lp_error_dark_just_below": min(below)}
    check(f"F1 no light is needed iff T >= {ts:.4f}", ok,
          f"dark LP error {max(above):.7f} at 1.001 T*, {min(below):.7f} at 0.999 T*")


# ---------- F2 ----------
def first_step(s0, s1, eps):
    """A dark level (count always 0, target s0) against a level h (target s1). The test is a number
    delta = d(0) plus free values on n > 0, so it reaches every probability in
    [exp(-h) delta, exp(-h) delta + 1 - exp(-h)] under h. With delta within eps of s0:
      s1 > s0: h >= ln((1 - s0 - eps) / (1 - s1 + eps));   s1 < s0: h >= ln((s0 - eps) / (s1 + eps))."""
    if abs(s1 - s0) <= 2 * eps:
        return 0.0
    r = (1 - s0 - eps) / (1 - s1 + eps) if s1 > s0 else (s0 - eps) / (s1 + eps)
    return max(0.0, math.log(r))


def f2():
    worst = 0.0; n = 0
    rng = np.random.default_rng(5)
    cases = [(0.5, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5)]
    cases += [tuple(rng.uniform(0, 1, 2)) for _ in range(200)]
    for s0, s1 in cases:
        worst = max(worst, abs(first_step(s0, s1, EPS) - min_hi(0.0, s0, s1, EPS))); n += 1
    RES["forms"]["F2_first_chain_step"] = {
        "T0_value": first_step(0.5, 1.0, EPS), "T0_formula": "ln((1/2 - eps) / eps) = ln 49",
        "strict_value": first_step(0.0, 1.0, EPS), "status": "exact", "max_abs_error_vs_min_hi": worst, "cases": n}
    check("F2 first chain step from a dark level, T = 0: ln 49 (strict comparison: ln 99)", worst < 1e-9,
          f"{first_step(0.5, 1.0, EPS):.6f}; max |formula - min_hi| over {n} target pairs {worst:.1e}")


# ---------- F3 ----------
def k2_floor(T, eps):
    """K = 2. Each channel has one dark and one lit level; with sg = sigmoid(1 / T) the two column
    constraints are (1/2 -> sg) and (1 - sg -> 1/2), and both orders of the levels give the same
    pair of numbers, so per channel
      h = max(ln((1/2 - eps) / (1 - sg + eps)), ln((sg - eps) / (1/2 + eps))),  floor = 2 h."""
    sg = 1.0 if T == 0 else float(expit(1.0 / T))
    if sg - 0.5 <= 2 * eps:
        return 0.0
    return 2 * max(0.0, math.log((0.5 - eps) / (1 - sg + eps)), math.log((sg - eps) / (0.5 + eps)))


def f3():
    rows = []; worst = 0.0; ok = True
    for T in [0, 0.25, 0.5, 1, 2, 4, 8]:
        S = target(2, T)
        f = k2_floor(T, EPS); ref = chain_floor(S, EPS)["total"]
        worst = max(worst, abs(f - ref))
        h = f / 2
        t = lp_two([0.0, h * (1 + 1e-7)], [h * (1 + 1e-7), 0.0], S)[0]     # B's dark level is its top symbol
        t2 = lp_two([0.0, h * (1 + 1e-7)], [0.0, h * (1 + 1e-7)], S)[0]
        rows.append({"T": T, "floor": f, "chain_floor": ref, "lp_error_at_floor_y_reversed": t,
                     "lp_error_at_floor_y_same": t2, "attained": bool(min(t, t2) <= EPS + 1e-8)})
    RES["forms"]["F3_K2_floor"] = {
        "formula": "2 max(ln((1/2 - eps) / (1 - sg + eps)), ln((sg - eps) / (1/2 + eps))), sg = sigmoid(1 / T)",
        "status": "exact value of the pairwise floor (chain_floor for K = 2), not an optimum: a design attains it "
                  "only where the rows say so (T = 2 here); the formula is 0 from T = 12.5 although light is needed "
                  "until T = 25 (F1 with K = 2)",
        "max_abs_error_vs_chain_floor": worst, "rows": rows}
    check("F3 K = 2 two channel floor in closed form (T = 0: 2 ln 49)",
          worst < 1e-9 and abs(k2_floor(0, EPS) - 2 * math.log(49)) < 1e-12,
          f"max |formula - chain_floor| {worst:.1e}; attained by a design at T = {[r['T'] for r in rows if r['attained']]}, "
          f"not at T = {[r['T'] for r in rows if not r['attained']]}")


# ---------- F4 ----------
def coin_error(u, v):
    """P(N1 < N2) + P(N1 = N2) / 2 for N1 ~ Poisson(u), N2 ~ Poisson(v), u > v >= 0."""
    if v <= 0:
        return 0.5 * math.exp(-u)
    return float(skellam.cdf(-1, u, v) + 0.5 * skellam.pmf(0, u, v))


def skellam_chain(K, eps):
    """T = 0, x = y = L. For a > b the pairs (a, b) and (b, a) are mirror images and the likelihood
    ratio test between them is "N1 > N2, coin on a tie", which minimizes the sum of the two
    errors, so no decoder has a smaller worst error on that pair than the coin decoder's
    e(L_a, L_b) = P(N1 < N2) + P(N1 = N2) / 2; the coin decoder also gives exactly 1/2 on a = b.
    For an increasing table the LP error is therefore the max over a > b of e(L_a, L_b); for any
    table every pair of levels still needs e(larger, smaller) <= eps, which is all the chain uses.
    e is the mean of a nonincreasing function of N1 - N2, so it falls in the larger level and
    rises in the smaller one (stochastic order), and the cheapest table is the chain L_0 = 0, L_1 = ln(1 / (2 eps)),
    L_(k+1) = the u with e(u, L_k) = eps. Cost = 2 L_(K-1). This is the exact optimum of the
    family x = y at T = 0."""
    L = [0.0]
    for _ in range(K - 1):
        v = L[-1]
        L.append(brentq(lambda u: coin_error(u, v) - eps, v + 1e-9, v + 20 * math.sqrt(v + 1) + 30, xtol=1e-13))
    return L


def gauss_chain(K, eps):
    """Gaussian version of the same chain: (u - v) / sqrt(u + v) = z with z the 1 - eps normal
    quantile, so L_(k+1) = L_k + z^2 / 2 + z sqrt(2 L_k + z^2 / 4), started at the exact L_1."""
    z = norm.isf(eps)
    L = [0.0, math.log(1 / (2 * eps))]
    while len(L) < K:
        L.append(L[-1] + z * z / 2 + z * math.sqrt(2 * L[-1] + z * z / 4))
    return L


def f4():
    L = skellam_chain(4, EPS); G = gauss_chain(4, EPS)
    g = np.linspace(0.5, 40, 80)
    mono = all(coin_error(u + 0.3, v) < coin_error(u, v) and coin_error(u + 5, v + 0.3) > coin_error(u + 5, v)
               for u in g for v in g if v < u)
    S = target(4, 0)
    t = lp_two(L, L, S)[0]
    slack = max(coin_error(L[a], L[b]) for a in range(4) for b in range(a - 1))
    ref = stored("results/soft_twochan.json", "0", "equal")
    cost = 2 * L[3]
    ok = mono and abs(t - EPS) < 2e-8 and slack < EPS and abs(L[1] - math.log(50)) < 1e-9
    ok &= rel(cost, ref) < 2e-5 and cost <= ref
    RES["forms"]["F4_equal_tables_T0"] = {
        "skellam_chain_levels": L, "skellam_chain_cost": cost, "status": "exact for the family x = y at T = 0",
        "lp_error_at_chain": t, "stored_equal_family_cost": ref,
        "rel_error_vs_stored": rel(cost, ref),
        "gaussian_levels": G, "gaussian_cost": 2 * G[3], "gaussian_rel_error": rel(2 * G[3], cost),
        "gaussian_formula": "L(k+1) = L(k) + z^2 / 2 + z sqrt(2 L(k) + z^2 / 4), z = z(1 - eps), L(1) = ln(1 / (2 eps))",
        "gaussian_status": "approximation"}
    check(f"F4 two channels, x = y, T = 0: Skellam chain cost {cost:.4f}", ok,
          f"levels {np.round(L, 4).tolist()} LP error there {t:.9f}; stored equal family {ref}; "
          f"Gaussian recursion {2 * G[3]:.3f} ({100 * rel(2 * G[3], cost):.1f}% off)")


# ---------- F5 ----------
def ramp_threshold_ok(g, eps):
    """T = 0, means g k for k = A - B + 3: target 0 at 0, g, 2 g; 1/2 at 3 g; 1 at 4 g, 5 g, 6 g.
    Claim: the table meets eps iff ONE randomized threshold "1 if N > k, with probability gam if
    N = k" does. Proof: take any decoder d that meets eps and the threshold test with the same
    probability under 3 g. Poisson has a monotone likelihood ratio, so that test has the largest
    probability of a 1 under every larger mean and the smallest under every smaller mean among
    tests with that probability under 3 g; it is therefore at least as good as d on all seven
    means. Order thresholds by c = k - gam (larger = stricter). The three conditions
    tail(2 g) <= eps, |tail(3 g) - 1/2| <= eps, tail(4 g) >= 1 - eps are intervals in c, and the
    function returns whether they overlap (2 g and 4 g are the binding means on each side)."""
    def tail(m, k, gam):                       # P(N > k) + gam P(N = k)
        return float(pdtrc(k, m)) + gam * (float(pdtr(k, m)) - (float(pdtr(k - 1, m)) if k > 0 else 0.0))

    def cut(m, alpha):                         # (k, gam) with tail(m, k, gam) = alpha
        k = 0
        while pdtrc(k, m) > alpha:
            k += 1
        pk = float(pdtr(k, m)) - (float(pdtr(k - 1, m)) if k > 0 else 0.0)
        return k, (alpha - float(pdtrc(k, m))) / pk

    hi = cut(3 * g, 0.5 + eps)                 # loosest threshold the tie allows
    lo = cut(3 * g, 0.5 - eps)                 # strictest
    need4 = cut(4 * g, 1 - eps)                # tail(4 g) >= 1 - eps iff the threshold is at most this strict
    need2 = cut(2 * g, eps)                    # tail(2 g) <= eps iff the threshold is at least this strict
    key = lambda kg: kg[0] - kg[1]             # larger key = stricter test (higher cut, less randomization)
    low = max(key(need2), key(hi)); high = min(key(need4), key(lo))
    return low <= high


def f5():
    lo, hi = 1.0, 60.0
    while hi - lo > 1e-10:
        m = 0.5 * (lo + hi)
        if ramp_threshold_ok(m, EPS):
            hi = m
        else:
            lo = m
    g = hi
    grid_ok = all(ramp_threshold_ok(v, EPS) for v in np.linspace(g * 1.0001, 2 * g, 60)) and \
        not any(ramp_threshold_ok(v, EPS) for v in np.linspace(0.3 * g, g * 0.9999, 60))
    S = target(4, 0)
    x = g * np.arange(4.0); y = x[::-1]
    t_at = lp_shared(x * (1 + 1e-7), y * (1 + 1e-7), S)[0]; t_below = lp_shared(x * 0.999, y * 0.999, S)[0]
    ref = stored("results/soft_shared.json", "0", "ramp")
    # 2 sqrt(N) is roughly normal with mean 2 sqrt(m) and unit variance. A threshold c on 2 sqrt(N)
    # keeps the tail at 3 g at most 1/2 + eps iff c >= 2 sqrt(3 g) - z(1/2 + eps) and the tail at 4 g
    # at least 1 - eps iff c <= 2 sqrt(4 g) - z(1 - eps), with z(p) the p quantile of the normal.
    z1, z2 = norm.ppf(1 - EPS), norm.ppf(0.5 + EPS)
    g_sqrt = ((z1 - z2) / (2 * (2 - math.sqrt(3)))) ** 2
    ok = grid_ok and t_at <= EPS + 1e-8 and t_below > EPS
    ok &= rel(6 * g, ref) < 2e-5 and 6 * g <= ref and 0.04 < rel(g_sqrt, g) < 0.06 and g_sqrt < g
    RES["forms"]["F5_ramp_T0"] = {
        "g": g, "cost": 6 * g, "status": "exact for the ramp family at T = 0 (one randomized threshold, Poisson cdf)",
        "lp_error_at_g": t_at, "lp_error_at_0.999g": t_below, "stored_ramp_cost": ref,
        "rel_error_vs_stored": rel(6 * g, ref),
        "sqrt_rule_g": g_sqrt, "sqrt_rule_cost": 6 * g_sqrt, "sqrt_rule_rel_error": rel(g_sqrt, g),
        "sqrt_rule_formula": "g = ((z(1 - eps) - z(1/2 + eps)) / (2 (2 - sqrt 3)))^2, "
                             "from 2 sqrt(4 g) - 2 sqrt(3 g) = z(1 - eps) - z(1/2 + eps)",
        "sqrt_rule_status": "approximation, about 5% low"}
    check(f"F5 one detector ramp, T = 0: one randomized threshold works from g = {g:.5f}, cost {6 * g:.4f}", ok,
          f"LP error {t_at:.9f} at g, {t_below:.7f} at 0.999 g; stored ramp {ref}; "
          f"square root rule {6 * g_sqrt:.2f} ({100 * rel(g_sqrt, g):.1f}% low)")


# ---------- F6 ----------
def f6():
    """Arrival time: max(muA) >= -ln(1 - gap + 2 eps) with gap = sigmoid(2 / T) - sigmoid(-1 / T)
    (levels 3 and 0 of A seen against b = 1), same for B; derivation in src/soft_time.py. This
    restates the formula soft_time.disjoint_floor evaluates, with the binding triple written out."""
    rows = []; ok = True; worst = 0.0
    tm = json.load(open("results/soft_time.json"))["T"]
    for T in TS:
        gap = 1.0 if T == 0 else float(expit(2.0 / T) - expit(-1.0 / T))
        f = 2 * max(0.0, -math.log(min(1.0, 1 - gap + 2 * EPS)))
        r = tm[str(T)]
        best = r["families"][r["best_family"]]["cost"]
        worst = max(worst, abs(f - r["floor"]["raw"]), abs(f - disjoint_floor(target(4, T), EPS)["raw"]))
        ok &= f <= best and r["floor"]["value"] <= f
        rows.append({"T": T, "floor": f, "stored_floor_raw": r["floor"]["raw"], "stored_best_cost": best})
    ok &= worst < 1e-12 and abs(rows[0]["floor"] - 2 * math.log(50)) < 1e-12
    RES["forms"]["F6_arrival_time_floor"] = {
        "formula": "2 (-ln(1 - gap + 2 eps)), gap = sigmoid(2 / T) - sigmoid(-1 / T); T = 0: 2 ln 50",
        "status": "restatement of soft_time.disjoint_floor (proved, loose), not an independent check",
        "max_abs_diff_vs_disjoint_floor": worst, "rows": rows}
    check("F6 arrival time floor in closed form (T = 0: 2 ln 50)", ok,
          str([(r["T"], floor_down(r["floor"], slack=1e-12)) for r in rows]))


# ---------- F7 ----------
def f7():
    """Strict comparison [A > B], one threshold decoder (src/threshold_optimum.py). The cost is
    P = s (K u - (K - 1)), u the ratio of the two means that must be told apart and s = s*(u) the
    smallest lower mean that works, with u <= (K - 1) / (K - 2). For K = 3..10 the minimum sits at
    that corner (found numerically per K, not proved for all K), where P = 2 s (K - 1) / (K - 2).
    Square root rule: 2 sqrt(u s) - 2 sqrt(s) = 2 z with z = z(1 - eps) gives
    s = z^2 / (sqrt(u) - 1)^2 and P = 2 (K - 1) z^2 / (sqrt(K - 1) - sqrt(K - 2))^2, which grows
    like 8 z^2 K^2. An approximation: the error changes sign with K (the counts are integers)."""
    z = norm.ppf(1 - EPS)
    rows = []
    for K in range(3, 11):
        u, s_low, P, umax = threshold_optimum.minimize_P(K, "low", n_scan=60)
        form = 2 * (K - 1) * z * z / (math.sqrt(K - 1) - math.sqrt(K - 2)) ** 2
        rows.append({"K": K, "exact": float(P), "formula": form, "rel_error": float((form - P) / P), "minimum_at_corner": bool(u == umax)})
    worst = max(abs(r["rel_error"]) for r in rows)
    RES["forms"]["F7_threshold_design_strict"] = {
        "formula": "2 (K - 1) z^2 / (sqrt(K - 1) - sqrt(K - 2))^2, z = z(1 - eps)", "status": "approximation",
        "exact_source": "threshold_optimum.minimize_P(K, 'low')", "max_abs_rel_error": worst,
        "max_abs_rel_error_K_ge_4": max(abs(r["rel_error"]) for r in rows if r["K"] >= 4), "rows": rows,
        "note": "eps = 0.01 only; the corner step is numerical per K"}
    check("F7 strict comparison, threshold design, K = 3..10: square root formula within 0.1%",
          worst < 1.0e-3 and all(r["minimum_at_corner"] for r in rows) and abs(rows[1]["exact"] - 321.54) < 0.005,
          "relative errors in percent " + str([round(100 * float(r["rel_error"]), 3) for r in rows]))
    return {r["K"]: r["exact"] for r in rows}


# ---------- F8 ----------
def f8(thr):
    """Strict comparison, peeled band construction (src/banded.py): v_0 = 0, v_(k+1) = h(2 v_k),
    where h(mu) is the smallest mean a two point test tells from mu at error eps; the pairwise
    budget is 2 v_(K-1). h(0) = ln((1 - eps) / eps) exactly (F2) and the square root rule gives
    h(mu) ~ (sqrt(mu) + z)^2, so C_2 = 2 ln((1 - eps) / eps), C_K = 2 (sqrt(C_(K-1)) + z)^2, or
    sqrt(C_K) = (sqrt(C_2) + a) 2^((K - 2) / 2) - a with a = (2 + sqrt 2) z: growth like 2^K.
    The start 2 ln(1 / eps) = 2 ln 100 lands closer to 2 v_(K-1) at eps = 0.01, but it does not
    follow from the model, so that variant is a fit. Neither is a bound: the cost the
    construction really needs (LP decoder, banded.py) is about 0.15% above 2 v_(K-1)."""
    z = norm.ppf(1 - EPS)
    a = (2 + math.sqrt(2)) * z
    C = {"exact_start": {2: 2 * math.log((1 - EPS) / EPS)}, "ln100_start": {2: 2 * math.log(1 / EPS)}}
    rows = []; ok = True
    for K in range(3, 9):
        v, x, y, _, _ = banded.build(K)
        pw = 2 * v[K - 1]
        row = {"K": K, "pairwise_budget": pw}
        for name, c in C.items():
            c[K] = 2 * (math.sqrt(c[K - 1]) + z) ** 2
            closed = ((math.sqrt(c[2]) + a) * 2 ** ((K - 2) / 2) - a) ** 2
            ok &= rel(closed, c[K]) < 1e-12
            row[name] = c[K]; row[name + "_rel_error"] = (c[K] - pw) / pw
        if K <= 6:                                   # the LP at K = 7, 8 takes minutes (python3 src/banded.py 8)
            row["lp_budget"] = float(budget_for_shape(x, y)[0])
            row["lp_budget_over_pairwise"] = row["lp_budget"] / pw
            ok &= 1.0 < row["lp_budget_over_pairwise"] < 1.002
        row["threshold_design"] = thr[K]
        row["band_cheaper_than_threshold"] = bool(row.get("lp_budget", pw) < thr[K])
        rows.append(row)
    e1 = max(abs(r["exact_start_rel_error"]) for r in rows); e2 = max(abs(r["ln100_start_rel_error"]) for r in rows)
    ok &= e1 < 1.2e-3 and e2 < 3e-4 and abs(rows[1]["lp_budget"] - 196.62) < 0.005
    ok &= [r["band_cheaper_than_threshold"] for r in rows] == [True, True, True, False, False, False]
    RES["forms"]["F8_peeled_band_strict"] = {
        "formula": "C_2 = 2 ln((1 - eps) / eps), C_K = 2 (sqrt(C_(K-1)) + z)^2, z = z(1 - eps)",
        "status": "approximation of the pairwise budget 2 v_(K-1) with the exact start; with the start 2 ln(1 / eps) "
                  "it is a fit; neither is an upper bound or a floor",
        "max_abs_rel_error_exact_start": e1, "max_abs_rel_error_ln100_start": e2, "rows": rows,
        "note": "eps = 0.01 only. The peeled band construction is cheaper than the threshold design up to K = 5 and "
                "dearer from K = 6 (at K = 6 already its pairwise budget, which the LP budget exceeds, is above the "
                "threshold cost). This compares two constructions; no floor is known for K >= 5."}
    check("F8 strict comparison, peeled band, K = 3..8: recursion against the pairwise budget 2 v_(K-1)", ok,
          f"exact start 2 ln 99 within {100 * e1:.2f}%, start 2 ln 100 (fit) within {100 * e2:.3f}%; LP budget over pairwise "
          f"budget K = 3..6 {[round(r['lp_budget_over_pairwise'], 5) for r in rows if 'lp_budget' in r]}; "
          f"band cheaper than threshold up to K = 5, not from K = 6")


# ---------- summary ----------
COST_STATUS = "numerical upper bound (searched table, LP decoder, stored decoder rechecked in mpmath)"


def best_row(r, narrow):
    """The row that carries the stored decoder, and the narrower family it coincides with, if any."""
    fams = r["families"]; e = fams[r["best_family"]]
    same = next((f for f in narrow if fams[f]["cost"] <= e["cost"] + 1e-9), None)
    return e, {"cost": e["cost"], "family": r["best_family"], "same_table_as": same, "table_source": e["source"],
               "worst_deviation_mpmath": e["mp_error"], "cost_status": COST_STATUS}


def summary():
    keys = [str(t) for t in TS]
    sh = json.load(open("results/soft_shared.json"))["T"]
    tw = json.load(open("results/soft_twochan.json"))["T"]
    tm = json.load(open("results/soft_time.json"))["T"]
    try:
        fl = json.load(open("results/soft_floor_shared.json"))["T"]
    except FileNotFoundError:
        fl = {}
    lad = json.load(open("results/ladder.json"))
    one = {}; two = {}; arr = {}
    for k in keys:
        e, one[k] = best_row(sh[k], ["ramp"])
        cf = sh[k]["chain_floor"]["value"]; mf = (fl.get(k) or {}).get("value")
        use_milp = mf is not None and mf > cf
        one[k].update({"floor": mf if use_milp else cf,
                       "floor_status": fl[k]["status"] if use_milp else sh[k]["chain_floor"]["status"],
                       "floor_source": "results/soft_floor_shared.json" if use_milp else "results/soft_shared.json (chain_floor)",
                       "chain_floor": cf, "pairwise_milp_floor": mf,
                       "pairwise_milp_note": None if mf is not None else "not established for this T",
                       "source": "results/soft_shared.json"})
        e, two[k] = best_row(tw[k], ["equal"])
        two[k].update({"floor": tw[k]["floor"]["value"], "floor_status": tw[k]["floor"]["status"],
                       "floor_source": "results/soft_twochan.json (chain_floor)", "source": "results/soft_twochan.json"})
        e, arr[k] = best_row(tm[k], ["flat", "strict_shape"])
        arr[k].update({"floor": tm[k]["floor"]["value"], "floor_status": tm[k]["floor"]["status"],
                       "floor_source": "results/soft_time.json (disjoint_floor)", "source": "results/soft_time.json"})
    for rows in (one, two, arr):
        for r in rows.values():
            r["cost_over_floor"] = r["cost"] / r["floor"]
    F = RES["forms"]
    fits = {
        "F1_no_light_temperature": {"value": F["F1_no_light_temperature"]["T_star"], "kind": "exact",
                                    "dark_lp_error_just_above": F["F1_no_light_temperature"]["lp_error_dark_just_above"],
                                    "dark_lp_error_just_below": F["F1_no_light_temperature"]["lp_error_dark_just_below"]},
        "F2_first_chain_step_T0_ln49": {"value": F["F2_first_chain_step"]["T0_value"], "kind": "exact",
                                        "max_abs_error": F["F2_first_chain_step"]["max_abs_error_vs_min_hi"]},
        "F3_K2_two_channel_floor": {"kind": "exact floor value", "max_abs_error": F["F3_K2_floor"]["max_abs_error_vs_chain_floor"],
                                    "attained_at_T": [r["T"] for r in F["F3_K2_floor"]["rows"] if r["attained"]]},
        "F4_skellam_chain_equal_tables_T0": {"value": F["F4_equal_tables_T0"]["skellam_chain_cost"], "kind": "exact for x = y",
                                             "rel_error_vs_stored_equal_family": F["F4_equal_tables_T0"]["rel_error_vs_stored"]},
        "F4_gaussian_recursion": {"value": F["F4_equal_tables_T0"]["gaussian_cost"], "kind": "approximation",
                                  "rel_error": F["F4_equal_tables_T0"]["gaussian_rel_error"]},
        "F5_single_threshold_ramp_T0": {"value": F["F5_ramp_T0"]["cost"], "kind": "exact for the ramp",
                                        "rel_error_vs_stored_ramp": F["F5_ramp_T0"]["rel_error_vs_stored"]},
        "F5_square_root_rule": {"value": F["F5_ramp_T0"]["sqrt_rule_cost"], "kind": "approximation, low",
                                "rel_error": F["F5_ramp_T0"]["sqrt_rule_rel_error"]},
        "F6_arrival_time_floor": {"kind": "restatement of the proved floor",
                                  "max_abs_diff": F["F6_arrival_time_floor"]["max_abs_diff_vs_disjoint_floor"]},
        "F7_threshold_design_strict": {"kind": "approximation",
                                       "max_abs_rel_error_K3_to_10": F["F7_threshold_design_strict"]["max_abs_rel_error"]},
        "F8_peeled_band_strict_exact_start": {"kind": "approximation of the pairwise budget",
                                              "max_abs_rel_error_K3_to_8": F["F8_peeled_band_strict"]["max_abs_rel_error_exact_start"]},
        "F8_peeled_band_strict_ln100_start": {"kind": "fit",
                                              "max_abs_rel_error_K3_to_8": F["F8_peeled_band_strict"]["max_abs_rel_error_ln100_start"]},
    }
    strict_time = disjoint_floor(strict_target(4), EPS)
    out = {
        "question": "min max(x) + max(y) so that the one bit output is 1 with probability within eps of "
                    "sigmoid((A - B) / T) for all 16 pairs (T = 0: 1 / 0.5 / 0)",
        "eps": EPS, "K": 4, "temperatures": TS,
        "reading": "cost = cheapest design found: a numerical upper bound (searched table, LP decoder, the stored "
                   "decoder rechecked in mpmath with the truncated tail charged). floor = lower bound for any encoder "
                   "and decoder of that class, with its own status. No cost is proved optimal. Costs are scaled to "
                   "error eps - 2e-8, so their relative excess over the scale that hits eps exactly is about 1e-6: "
                   "quote them to 2 or 3 decimals. Floors are rounded down. A wider class never costs more: "
                   "two channels <= one detector and arrival time <= one detector (add up the slots).",
        "classes": {"one_detector": one, "two_channels": two, "arrival_time": arr},
        "strict_comparison_reference": {
            "source": "results/ladder.json (output [A > B], ties false, eps 0.01, no background)",
            "one_detector_threshold": {"cost": lad["lines"]["threshold"][0]["cost"],
                                       "floor": lad["certified_floor"]["threshold"]["cost"]},
            "one_detector_banded": {"cost": lad["lines"]["banded"][0]["cost"],
                                    "floor": lad["certified_floor"]["banded"]["cost"]},
            "two_channels": {"cost": lad["lines"]["two_channel"][0]["cost"],
                             "floor": lad["certified_floor"]["two_channel"]["cost"]},
            "arrival_time": {"cost": lad["lines"]["arrival_time"][0]["cost"],
                             "floor": strict_time["value"],
                             "floor_status": "proved (soft_time.disjoint_floor on the strict target, 2 ln 50), loose",
                             "timecomp_bound_value_not_a_floor": json.load(open("results/timecomp.json"))["bound"]["value"],
                             "timecomp_bound_note": "An earlier README and ladder.json listed 9.11 as a loose floor. It is the end point "
                                                 "of a Nelder-Mead search over three pairwise conditions (timecomp.py "
                                                 "--bound); the same three conditions are met at cost 9.1022 "
                                                 "(A0, A1, B0, B1 = 0, 4.59512, 4.50711, 2.91509), so 9.11 is not a "
                                                 "floor and is not used here"}},
        "closed_form_fit_errors": fits,
    }
    dump_json(out, SUMMARY)
    print("T      one detector (floor)      two channels (floor)      arrival time (floor)")
    for k in keys:
        print(f"SUMMARY T={k} one_detector {one[k]['cost']:.4f} floor {one[k]['floor']:.4f} two_channels {two[k]['cost']:.4f} "
              f"floor {two[k]['floor']:.4f} arrival_time {arr[k]['cost']:.4f} floor {arr[k]['floor']:.4f}")
    ok = all(two[k]["cost"] <= one[k]["cost"] and arr[k]["cost"] <= one[k]["cost"] + 1e-9
             and two[k]["floor"] <= two[k]["cost"] and one[k]["floor"] <= one[k]["cost"]
             and arr[k]["floor"] <= arr[k]["cost"] and one[k]["floor"] >= two[k]["floor"] - 1e-12
             and max(one[k]["worst_deviation_mpmath"], two[k]["worst_deviation_mpmath"],
                     arr[k]["worst_deviation_mpmath"]) <= EPS for k in keys)
    check("summary written (copies of stored numbers): two channels <= one detector, arrival time <= one detector, "
          "floors <= costs, every stored deviation <= eps", ok, SUMMARY)


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5(); f6(); f8(f7())
    dump_json(RES, OUT)
    summary()
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - T0:.1f}s)")
    sys.exit(1 if FAILS else 0)
