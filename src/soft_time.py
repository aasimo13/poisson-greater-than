"""Sampled comparison by arrival time: cheapest pulse tables found, per temperature, and a floor.

Model (src/timecomp.py, no background): sender A (level a) puts mean muA[a] photons into time slot
a, sender B (level b) puts muB[b] into slot b, one detector counts per slot, the four counts are
independent Poisson. The receiver outputs 1 with probability d(counts). Target
s[A, B] = sigmoid((A - B) / T) (T = 0: 1 / 0.5 / 0); every pair must have |q - s| <= 0.01.
Cost = max(muA) + max(muB). Rows of q and s are A. The error target is fixed at 0.01; an EPS
environment variable is ignored.

With no background at most two slots are lit, so the LP over the count vectors with at most two
nonzero entries is exact; counts above the grid are charged against the design both ways.

Families (free is the minimum over all three):
  flat          muA = (p, p, p, p), muB = (r, r, r, r): two numbers, the slot alone carries the level
  strict_shape  the winner of the strict comparison (timecomp.py, cost 20.97) rescaled
  free          all eight pulse energies free
The flat and strict_shape rows are reference rows and store no decoder. The free row stores its
decoder, and its cost is certified by the mpmath recheck of that decoder, not by the LP value.
Every cost is a searched UPPER bound (Nelder-Mead), not an optimum. The free search starts from
the cheaper reference row, the SEEDS below, the stored table, the stored tables of the two
neighboring temperatures, and the one detector table of the same temperature
(results/soft_shared.json): a receiver that adds up the four slots sees Poisson(muA[a] + muB[b]),
which is the one detector model, so an arrival time design never has to cost more than a one
detector design. From T = 0.5 up the cheapest tables found are graded: the pulse energy rises with
the level for one sender and falls for the other, so brightness and slot both carry the level.
At T = 0 the cheapest table found (19.31) is nearly flat except that B sends nothing at its
lowest level; the same idea costs 17.58 at T = 0.25, where the flat table (17.38) stays the
cheapest found. These are local searches from a handful of starts, not a global optimum.

Floor. soft_core.chain_floor does NOT apply here. Proved instead (disjoint_floor): fix b and two
levels a, a' of A, both different from b. Slot b looks the same under (a, b) and (a', b), and the
only outcome the two share on slots a, a' is "both dark", with probability exp(-muA[a]) and
exp(-muA[a']). A test that says 1 with probability delta on that outcome reaches at most
1 - exp(-muA[a]) (1 - delta) under the first and at least exp(-muA[a']) delta under the second, so
with s1 = s[a, b] > s2 = s[a', b] it needs (1 - s1 + eps) exp(muA[a]) + (s2 + eps) exp(muA[a']) >= 1,
hence max(muA) >= -ln(1 - (s1 - s2) + 2 eps). The same holds for B. The floor is the sum of the two
best such bounds. It is a closed form, proved, and loose: it ignores ties (a = b) and that one
decoder serves all 16 pairs. No other floor is reported. (Pairwise conditions on two slots, as in
timecomp.py --bound, are also necessary, but their minimum was only ever searched, and a wider
search kept finding lower values, so no number from them is stored here.)

Usage (repo root):
  python3 src/soft_time.py                           verify results/soft_time.json: cost, mpmath recheck of the
                                                     stored decoders, fresh LP, floor (a few seconds)
  python3 src/soft_time.py --search [maxfev] [T ..]  search again from the seeds and the stored tables, keep the
                                                     cheaper ones (long at T = 0 and 0.25; maxfev 0 only reprices)
"""
import json
import math
import os
import sys
import time
import numpy as np
import mpmath as mp
from soft_core import target, lp_rows, grid_max, pmf_rows, budget_near, nm_polish, dump_json, floor_down, MP_DPS, _mp_pmf

EPS = 0.01
MARGIN = 2e-8          # designs are scaled to LP error eps - MARGIN so the mpmath recheck passes strictly
TS = [0, 0.25, 0.5, 1, 2, 4]
OUT = os.environ.get("SOFT_TIME_OUT", "results/soft_time.json")
SHARED = "results/soft_shared.json"
PAIRS = [(a, b) for a in range(4) for b in range(4)]
STRICT = ([4.535158813102838, 4.786907722848276, 4.830716518368628, 4.833524524491183],
          [16.04681139542948, 16.138659748846283, 16.136413046185083, 16.097174028760445])
STRICT_COST = 20.972184273337465
FLAT = {"0": (4.3355, 15.5362), "0.25": (3.4955, 13.8817), "0.5": (3.3093, 13.1145), "1": (2.0446, 7.9264),
        "2": (1.0693, 4.0329), "4": (0.6204, 2.3391)}          # (p, r) starts for the flat row
# starting tables (muA, muB) for the free row, from earlier searches. A rerun also starts from the
# tables already stored in OUT, so repeated searches can only lower the costs.
SEEDS = {
    "0": [([4.46670243, 4.46659494, 3.59731762, 4.33111471], [0.01099599, 14.84398493, 14.84407008, 14.84405700]),
          ([4.71829780, 3.29263980, 3.30795601, 4.71828393], [14.89137350, 14.89133689, 14.89137380, 14.89121492])],
    "0.25": [([3.49550029, 3.49550029, 3.49550029, 3.49550029], [13.88165193, 13.88165193, 13.88165193, 13.88165193]),
             ([4.46680000, 4.46690000, 3.59520000, 4.33500000], [0.01110000, 14.84380000, 14.84400000, 14.84400000])],
    "0.5": [([4.35492111, 4.35502894, 2.28889472, 0.36793232], [0.21349047, 2.27861649, 6.21684057, 6.21679795]),
            ([4.35597216, 4.35804656, 2.28898049, 0.36834494], [0.21379981, 2.27988810, 6.21596424, 6.21803123])],
    "1": [([0.38596330, 2.28852008, 2.66023156, 2.68573670], [2.81239215, 2.50851636, 0.96990553, 0.00000036]),
          ([0.00000000, 0.96383341, 2.06944427, 2.80371859], [2.69494445, 2.40720707, 1.29024241, 0.36676650])],
    "2": [([0.22264788, 1.15341785, 1.21574961, 1.21752333], [1.35316897, 1.31014224, 0.40075679, 0.00000002]),
          ([0.22044039, 1.02504506, 1.17519202, 1.21978978], [1.35161765, 1.24874992, 0.41352228, 0.00058914])],
    "4": [([0.12937551, 0.38144327, 0.36237725, 0.46589037], [0.46866228, 0.39292684, 0.09252408, 0.00000000]),
          ([0.12932051, 0.37749631, 0.36153420, 0.46618790], [0.46862935, 0.38665964, 0.09198877, 0.00000000])],
}
FAILS = []
T0 = time.time()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ---------- model ----------
def grids(muA, muB):
    """Largest count kept per slot: Ms when the slot is the only lit one, Mp when two are lit."""
    Ms = [grid_max([muA[i] + muB[i], muA[i], muB[i]]) for i in range(4)]
    Mp = [grid_max([muA[i], muB[i]]) for i in range(4)]
    return Ms, Mp


def layout(Ms, Mp):
    """Outcome index: 0 is all dark, then slot i alone with n = 1..Ms[i], then slots i < j with
    n_i = 1..Mp[i], n_j = 1..Mp[j] (row major)."""
    off = 1; single = []; pair = {}
    for i in range(4):
        single.append((off, Ms[i])); off += Ms[i]
    for i in range(4):
        for j in range(i + 1, 4):
            pair[i, j] = (off, Mp[i], Mp[j]); off += Mp[i] * Mp[j]
    return single, pair, off


def prob_matrix(muA, muB, Ms, Mp):
    single, pair, N = layout(Ms, Mp)
    P = np.zeros((16, N))
    for k, (a, b) in enumerate(PAIRS):
        if a == b:
            p = pmf_rows([muA[a] + muB[a]], Ms[a])[0]
            o, m = single[a]
            P[k, 0] = p[0]; P[k, o:o + m] = p[1:]
            continue
        pa = pmf_rows([muA[a]], Ms[a])[0]; pb = pmf_rows([muB[b]], Ms[b])[0]
        P[k, 0] = pa[0] * pb[0]
        o, m = single[a]; P[k, o:o + m] = pa[1:] * pb[0]
        o, m = single[b]; P[k, o:o + m] = pb[1:] * pa[0]
        i, j = min(a, b), max(a, b)
        pi, pj = (pa, pb) if a < b else (pb, pa)
        o, mi, mj = pair[i, j]
        P[k, o:o + mi * mj] = np.outer(pi[1:mi + 1], pj[1:mj + 1]).ravel()
    return P, np.clip(1.0 - P.sum(axis=1), 0.0, None)


def lp_time(muA, muB, S):
    """Best randomized decoder. Returns (t, decoder dict): zero, single[i][n-1], pair["ij"][ni-1][nj-1]."""
    muA = np.asarray(muA, float); muB = np.asarray(muB, float)
    Ms, Mp = grids(muA, muB)
    P, tail = prob_matrix(muA, muB, Ms, Mp)
    t, d = lp_rows(P, tail, np.asarray(S, float).ravel())
    single, pair, _ = layout(Ms, Mp)
    dec = {"zero": float(d[0]), "single": [d[o:o + m].tolist() for o, m in single],
           "pair": {f"{i}{j}": d[o:o + mi * mj].reshape(mi, mj).tolist() for (i, j), (o, mi, mj) in pair.items()}}
    return t, dec


def recheck_time_mp(muA, muB, S, dec, full=False):
    """Worst |q - s| of a stored decoder in mpmath; mass outside the stored grids counts against."""
    S = np.asarray(S, float)
    flat = [dec["zero"]] + [v for row in dec["single"] for v in row] + [v for D in dec["pair"].values() for r in D for v in r]
    if min(flat) < 0 or max(flat) > 1:
        raise ValueError("decoder entries must lie in [0, 1]")
    Q = np.zeros((4, 4))
    with mp.workdps(MP_DPS):
        f = lambda v: mp.mpf(float(v))
        zero = f(dec["zero"]); single = [[f(v) for v in row] for row in dec["single"]]
        worst = mp.mpf(0)
        for a, b in PAIRS:
            if a == b:
                p = _mp_pmf(f(muA[a]) + f(muB[a]), len(single[a]))
                q = p[0] * zero + mp.fdot(p[1:], single[a]); mass = mp.fsum(p)
            else:
                pa = _mp_pmf(f(muA[a]), len(single[a])); pb = _mp_pmf(f(muB[b]), len(single[b]))
                q = pa[0] * pb[0] * zero + pb[0] * mp.fdot(pa[1:], single[a]) + pa[0] * mp.fdot(pb[1:], single[b])
                mass = pa[0] * pb[0] + pb[0] * mp.fsum(pa[1:]) + pa[0] * mp.fsum(pb[1:])
                i, j = min(a, b), max(a, b)
                pi, pj = (pa, pb) if a < b else (pb, pa)
                D = dec["pair"][f"{i}{j}"]
                if len(D) and len(D[0]):
                    mi, mj = len(D), len(D[0])
                    wj = pj[1:mj + 1]
                    q += mp.fdot(pi[1:mi + 1], [mp.fdot([f(v) for v in row], wj) for row in D])
                    mass += mp.fsum(pi[1:mi + 1]) * mp.fsum(wj)
            s = f(S[a, b])
            worst = max(worst, q + (1 - mass) - s, s - q)
            Q[a, b] = float(q)
        worst = float(worst)
    return (worst, Q) if full else worst


# ---------- designs ----------
def lp_err(muA, muB, S):
    try:
        return lp_time(muA, muB, S)[0]
    except RuntimeError:
        return 1.0            # solver failure counts as "does not meet eps": the cost can only go up


def finish(muA, muB, S, source, store_decoder=True):
    """Scale the shape to the smallest cost with LP error <= eps - MARGIN, solve, recheck in mpmath."""
    muA = np.asarray(muA, float); muB = np.asarray(muB, float)
    c = budget_near(lambda c: lp_err(c * muA, c * muB, S), 1.0, EPS - MARGIN, rel=1e-8)
    while True:
        A, B = c * muA, c * muB
        t, dec = lp_time(A, B, S)
        m, Q = recheck_time_mp(A, B, S, dec, full=True)
        if m <= EPS:
            break
        c *= 1.0 + 1e-6
    Ms, Mp = grids(A, B)
    e = {"muA": A.tolist(), "muB": B.tolist(), "cost": float(A.max() + B.max()), "lp_error": float(t),
         "mp_error": float(m), "source": source, "grid": {"single": Ms, "pair": Mp}, "q": Q.tolist()}
    if store_decoder:
        e["decoder"] = dec
    return e


def tables(family, v):
    v = np.abs(np.asarray(v, float))
    if family == "flat":
        return np.full(4, v[0]), np.full(4, v[1])
    return v[:4], v[4:]


def cost_of(family, S, cap):
    def f(v):
        A, B = tables(family, v)
        c1 = A.max() + B.max()
        if c1 <= 0:
            return 1e9
        s = budget_near(lambda c: lp_err(c * A, c * B, S), min(1.0, cap / c1), EPS - MARGIN, rel=2e-6, cap=cap / c1)
        return c1 * s if np.isfinite(s) else 1e9
    return f


def with_decoder(e, S, source):
    """Copy of a reference row with the LP decoder of exactly that table added (rechecked in mpmath)."""
    A = np.array(e["muA"]); B = np.array(e["muB"])
    t, dec = lp_time(A, B, S)
    m, Q = recheck_time_mp(A, B, S, dec, full=True)
    if m > EPS:                       # the LP came out a hair worse this time: rescale instead
        return finish(A, B, S, source)
    return dict(e, lp_error=float(t), mp_error=float(m), q=Q.tolist(), decoder=dec, source=source)


def settle(fam, S):
    """If no wider table beat the cheaper reference row, the free row IS that row, and says so."""
    narrow = min(("flat", "strict_shape"), key=lambda n: fam[n]["cost"])
    if fam[narrow]["cost"] <= fam["free"]["cost"] + 1e-9:
        fam["free"] = with_decoder(fam[narrow], S, f"{narrow} row (the wider search did not beat it)")
        if fam["free"]["cost"] > fam[narrow]["cost"]:
            fam[narrow] = {k: v for k, v in fam["free"].items() if k != "decoder"}


def one_detector_table(key):
    """Stored one detector table (x, y) of this temperature, or None (see the module docstring)."""
    try:
        r = json.load(open(SHARED))["T"][key]
    except (FileNotFoundError, KeyError):
        return None
    e = r["families"][r["best_family"]]
    return e["x"], e["y"], e["cost"]


def search(key, maxfev, old=None, neighbors=()):
    T = float(key); S = target(4, T)
    cap = 3.0 * STRICT_COST
    fam = {}
    f = cost_of("flat", S, cap)
    v0 = np.array([np.mean(old["flat"]["muA"]), np.mean(old["flat"]["muB"])] if old else FLAT[key])
    v0 = v0 * f(v0) / v0.sum()
    c, v = nm_polish(f, v0, min(maxfev, 120))
    fam["flat"] = finish(*tables("flat", v), S, "Nelder-Mead on two pulse energies", store_decoder=False)
    print(f"T={key} flat {fam['flat']['cost']:.6f}  [{time.time() - T0:.0f}s]", flush=True)
    k0 = cost_of("free", S, cap)(np.concatenate(STRICT)) / STRICT_COST
    fam["strict_shape"] = finish(k0 * np.array(STRICT[0]), k0 * np.array(STRICT[1]), S,
                                 "timecomp.py winner, scale by bisection", store_decoder=False)
    print(f"T={key} strict_shape {fam['strict_shape']['cost']:.6f}  [{time.time() - T0:.0f}s]", flush=True)
    narrow = min(("flat", "strict_shape"), key=lambda n: fam[n]["cost"])
    starts = [(fam[narrow]["muA"], fam[narrow]["muB"], f"{narrow} optimum")]
    starts += [(a, b, f"seed {i}") for i, (a, b) in enumerate(SEEDS[key])]
    od = one_detector_table(key)
    if od:
        starts.append((od[0], od[1], "one detector table"))
    if old:
        starts.append((old["free"]["muA"], old["free"]["muB"], "stored table"))
    starts += [(e["muA"], e["muB"], f"stored table of T={k}") for k, e in neighbors]
    best = None; log = []
    f = cost_of("free", S, cap)
    for A, B, name in starts:
        v0 = np.concatenate([A, B]).astype(float)
        c0 = f(v0)
        if c0 >= 1e9:                 # the shape needs more than cap photons at this temperature
            log.append({"start": name, "maxfev": maxfev, "cost": None})
            print(f"T={key} free from {name}: dropped, needs more than {cap:.0f} photons  [{time.time() - T0:.0f}s]", flush=True)
            continue
        c, v = nm_polish(f, v0 * c0 / (max(A) + max(B)), maxfev)
        log.append({"start": name, "maxfev": maxfev, "start_cost": round(c0, 6), "cost": round(c, 6)})
        print(f"T={key} free from {name}: {c0:.6f} -> {c:.6f}  [{time.time() - T0:.0f}s]", flush=True)
        if best is None or c < best[0]:
            best = (c, v, name)
    how = "Nelder-Mead on 8 pulse energies from the" if maxfev > 0 else "rescaled, no search:"
    fam["free"] = finish(*tables("free", best[1]), S, f"{how} {best[2]}")
    return {"T": T, "best_family": "free", "families": fam, "search_log": log, "floor": disjoint_floor(S, EPS)}


def run_search(args):
    if args and not args[0].isdigit():
        sys.exit("usage: --search [maxfev] [T ..]   maxfev is a whole number and comes first")
    maxfev = int(args[0]) if args else 300          # first argument is maxfev, the rest are temperatures
    keys = args[1:] or [str(t) for t in TS]
    bad = [k for k in keys if k not in SEEDS]
    if bad:
        sys.exit(f"unknown temperature {bad}; use {list(SEEDS)}")
    try:
        res = json.load(open(OUT))
    except FileNotFoundError:
        res = {"T": {}}
    res.update({"model": "arrival time, 4 slots, no background; rows of q and s are A; decoder: zero (all dark), "
                         "single[i][n-1] (only slot i lit), pair[ij][ni-1][nj-1] (slots i < j lit)",
                "eps": EPS, "margin": MARGIN,
                "status": "costs are numerical upper bounds (searched tables, LP decoder, the stored decoder of the "
                          "free row rechecked in mpmath); floors are proved and loose"})
    res["T"] = {k: v for k, v in res.pop("T").items()}
    order = [str(t) for t in TS]
    for key in keys:
        old = res["T"].get(key)
        i = order.index(key)
        nb = [(k, res["T"][k]["families"]["free"]) for k in order[max(0, i - 1):i] + order[i + 1:i + 2] if k in res["T"]]
        new = search(key, maxfev, old and old["families"], nb if maxfev > 0 else ())
        if old:                      # a new table replaces the stored one only if it is cheaper
            for f in new["families"]:
                if old["families"][f]["cost"] <= new["families"][f]["cost"] + 1e-9:
                    new["families"][f] = old["families"][f]
            new["search_log"] = old.get("search_log", []) + (new["search_log"] if maxfev > 0 else [])
        settle(new["families"], target(4, float(key)))
        res["T"][key] = new
        res["T"] = dict(sorted(res["T"].items(), key=lambda kv: float(kv[0])))
        dump_json(res, OUT)
        print(f"T={key} cost {new['families']['free']['cost']:.4f} floor {new['floor']['value']:.4f}", flush=True)


# ---------- floor ----------
def disjoint_floor(S, eps):
    """Proved lower bound on max(muA) + max(muB) (see the module docstring)."""
    S = np.asarray(S, float); K = S.shape[0]
    gA = max((abs(S[a, b] - S[a2, b]), (a, a2, b)) for b in range(K) for a in range(K) for a2 in range(K)
             if len({a, a2, b}) == 3)
    gB = max((abs(S[a, b] - S[a, b2]), (a, b, b2)) for a in range(K) for b in range(K) for b2 in range(K)
             if len({a, b, b2}) == 3)
    fa = max(0.0, -math.log(min(1.0, 1.0 - gA[0] + 2 * eps)))
    fb = max(0.0, -math.log(min(1.0, 1.0 - gB[0] + 2 * eps)))
    return {"value": floor_down(fa + fb, slack=1e-12), "raw": fa + fb, "floor_A": fa, "floor_B": fb,
            "binding_A_a_a2_b": list(gA[1]), "binding_B_a_b_b2": list(gB[1]),
            "status": "proved (disjoint support, closed form -ln(1 - gap + 2 eps) per sender); loose"}


def verify():
    """Pass criterion for a cost: the mpmath recheck of the decoder stored on disk. The fresh LP
    is only compared (HiGHS optima on these problems move by more than MARGIN between builds)."""
    res = json.load(open(OUT))
    prev = None; notes = []
    for key, r in res["T"].items():
        S = target(4, float(key))
        fl = disjoint_floor(S, EPS)
        ok = abs(fl["raw"] - r["floor"]["raw"]) < 1e-12 and r["floor"]["value"] <= fl["raw"]
        costs = {}
        for f, e in r["families"].items():
            A = np.array(e["muA"]); B = np.array(e["muB"])
            t, dec = lp_time(A, B, S)
            m = recheck_time_mp(A, B, S, e.get("decoder", dec))
            ok &= abs(A.max() + B.max() - e["cost"]) < 1e-12 and A.min() >= 0 and B.min() >= 0
            if "decoder" in e:
                ok &= m <= EPS and abs(m - e["mp_error"]) < 1e-12
            else:                     # reference row: only a fresh LP decoder can be rechecked
                ok &= m <= EPS + 1e-6
                if m > EPS:
                    notes.append(f"T={key} {f}: fresh LP decoder gives {m:.9f} (no decoder stored for this row)")
            if abs(t - e["lp_error"]) > 1e-7 or t > EPS:
                notes.append(f"T={key} {f}: fresh LP error {t:.9f}, stored {e['lp_error']:.9f}")
            ok &= r["floor"]["value"] <= e["cost"]
            costs[f] = e["cost"]
        best = r["families"][r["best_family"]]
        ok &= "decoder" in best and best["cost"] <= min(costs.values()) + 1e-12
        od = one_detector_table(key)
        if od:
            ok &= best["cost"] <= od[2] + 1e-9
        else:
            notes.append(f"T={key}: no one detector table in {SHARED}, cost <= one detector not checked")
        if prev is not None and best["cost"] > prev + 1e-9:
            notes.append(f"T={key}: cost {best['cost']:.4f} is above the cost at the next lower temperature {prev:.4f}")
        prev = best["cost"]
        check(f"T={key} cost {best['cost']:.4f} floor {r['floor']['value']:.4f}", ok,
              f"family {r['best_family']} ({best['source']}) mp_error {best['mp_error']:.10f} flat {costs['flat']:.4f} "
              f"strict_shape {costs['strict_shape']:.4f} one detector "
              f"{'none' if od is None else format(od[2], '.4f')} "
              f"muA {np.round(best['muA'], 4).tolist()} muB {np.round(best['muB'], 4).tolist()}")
    check("all six temperatures present", [float(k) for k in res["T"]] == [float(t) for t in TS])
    for n in notes:
        print("WARN  " + n)
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}  ({time.time() - T0:.1f}s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    if "--search" in sys.argv:
        run_search([a for a in sys.argv[1:] if a != "--search"])
    else:
        verify()
