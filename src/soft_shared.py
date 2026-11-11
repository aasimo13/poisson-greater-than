"""Sampled comparison on one detector: cheapest level tables found, per temperature.

Model: N ~ Poisson(x[A] + y[B]), A, B in 0..3, no background. The receiver outputs 1 with
probability d(N). Target s[A, B] = sigmoid((A - B) / T) (T = 0: 1 / 0.5 / 0); every pair must have
|q - s| <= 0.01. Cost = max(x) + max(y). Rows of q and s are A; decoder[n] is indexed by the count.
The error target is fixed at 0.01; an EPS environment variable is ignored.

Families (free is the minimum over both):
  ramp    x = g (0, 1, 2, 3), y = g (3, 2, 1, 0): the mean is g (A - B + 3), one free number
  free    x and y free, lowest level of each 0 (a common offset is undone by adding Poisson
          noise at the receiver, so it never hurts and lowers the cost)
The decoder is LP optimal for the table (soft_core.lp_shared). The free row is a LOCAL search: a
Nelder-Mead polish of the ramp, of the SEEDS below, of the stored table, of ramps with every level
moved by up to 7%, and of the level tables at which the floor MILP stopped (when
results/soft_floor_shared.json exists). Ramps with shuffled symbols were tried and dropped: they
need more than 12 times the ramp cost, as do some of the moved ramps (each dropped start is
logged). At T = 0, 0.25, 0.5 and 4 nothing beat the ramp and the free row is the ramp row. The
ramp is not the only table at its cost: at T = 4 the middle levels can move a long way for free
(x = (0, 0.165, 0.367, 0.595), y = (0.595, 0.384, 0.196, 0) costs the same 1.1908). Every cost is a searched UPPER bound, certified by the mpmath recheck of the
stored decoder, not by the LP value.
Floor: the larger of soft_core.chain_floor (which also bounds one detector) and the one detector
bound STORED by src/soft_floor_shared.py, when that file exists; this script does not solve that
MILP again.

Usage (repo root):
  python3 src/soft_shared.py                          verify results/soft_shared.json: cost, fresh LP, mpmath
                                                      recheck of the stored decoders, floor (a few seconds)
  python3 src/soft_shared.py --search [maxfev] [T ..] search again from the seeds and the stored tables, keep the
                                                      cheaper ones (minutes; maxfev 0 only reprices them)
"""
import json
import os
import sys
import time
import numpy as np
from soft_core import (target, lp_shared, recheck_shared_mp, budget_near, nm_polish, chain_floor,
                       floor_down, dump_json)

EPS = 0.01
MARGIN = 2e-8          # designs are scaled to LP error eps - MARGIN so the mpmath recheck passes strictly
TS = [0, 0.25, 0.5, 1, 2, 4]
OUT = os.environ.get("SOFT_SHARED_OUT", "results/soft_shared.json")
FLOOR = "results/soft_floor_shared.json"
JITTER_STARTS = 4
SANITY = {"0": 116.6}  # loose upper sanity bound for the ramp row (closed_forms.py F5 has the exact 116.4618)
RAMP = ([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0])
# starting tables (x, y) from earlier searches. A rerun also starts from the table already stored in
# OUT, so repeated searches can only lower the costs.
SEEDS = {
    "0": [], "0.25": [],
    "0.5": [([0, 4.54240542, 9.08481152, 13.62721999], [13.62722089, 9.08481468, 4.54240754, 0])],
    "1": [([0, 1.40721379, 2.84944190, 4.25665544], [4.25666023, 2.84944669, 1.40721869, 0])],
    "2": [([0, 0.55763782, 1.07168596, 1.64111745], [1.64111745, 1.08223521, 0.55763782, 0])],
    "4": [([0, 0.19839785, 0.39692320, 0.59538966], [0.59538965, 0.39685382, 0.19837086, 0])],
}
FAILS = []
T0 = time.time()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def lp_err(x, y, S):
    try:
        return lp_shared(x, y, S)[0]
    except RuntimeError:
        return 1.0            # solver failure counts as "does not meet eps": the cost can only go up


def finish(x, y, S, source):
    """Scale the shape to the smallest cost with LP error <= eps - MARGIN, solve, recheck in mpmath."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    c0 = 1.0
    while lp_err(c0 * x, c0 * y, S) > 0.4 and c0 < 1e4:      # get out of the flat region near no light
        c0 *= 2.0
    c = budget_near(lambda c: lp_err(c * x, c * y, S), c0, EPS - MARGIN, rel=1e-8)
    while True:
        xs, ys = c * x, c * y
        t, d = lp_shared(xs, ys, S)
        m, Q, _ = recheck_shared_mp(xs, ys, S, d, full=True)
        if m <= EPS:
            break
        c *= 1.0 + 1e-6
    return {"x": xs.tolist(), "y": ys.tolist(), "cost": float(xs.max() + ys.max()), "lp_error": float(t),
            "mp_error": float(m), "source": source, "grid": len(d) - 1, "q": Q.tolist(), "decoder": d.tolist()}


def tables(v):
    v = np.abs(np.asarray(v, float))
    x, y = v[:4], v[4:]
    return x - x.min(), y - y.min()


def cost_of(S, cap):
    """Budget of the shape v; shapes that need more than cap photons count as 1e9."""
    def f(v):
        x, y = tables(v)
        c1 = x.max() + y.max()
        if c1 <= 0:
            return 1e9
        s = budget_near(lambda c: lp_err(c * x, c * y, S), min(1.0, cap / c1), EPS - MARGIN, rel=2e-6, cap=cap / c1)
        return c1 * s if np.isfinite(s) else 1e9
    return f


def milp_tables(key):
    """Level tables at which the stages of the floor MILP stopped (relaxation optima, not designs)."""
    try:
        r = json.load(open(FLOOR))["T"][key]
    except (FileNotFoundError, KeyError):
        return []
    return [(st["x"], st["y"]) for st in r["stages"] if "x" in st][-2:]


def settle(fam):
    """If no wider table beat the ramp, the free row IS the ramp row, and says so."""
    if fam["ramp"]["cost"] <= fam["free"]["cost"] + 1e-9:
        fam["free"] = dict(fam["ramp"], source="ramp row (the wider search did not beat it)")


def search(key, maxfev, old=None):
    T = float(key); S = target(4, T)
    fam = {"ramp": finish(*RAMP, S, "ramp, scale by bisection")}
    print(f"T={key} ramp {fam['ramp']['cost']:.6f}  [{time.time() - T0:.0f}s]", flush=True)
    g = fam["ramp"]["cost"] / 6.0
    starts = [(fam["ramp"]["x"], fam["ramp"]["y"], "ramp optimum")] + [(x, y, "seed") for x, y in SEEDS[key]]
    if old:
        starts.append((old["free"]["x"], old["free"]["y"], "stored table"))
    if maxfev > 0:
        starts += [(x, y, f"floor MILP stage table {i}") for i, (x, y) in enumerate(milp_tables(key))]
        rng = np.random.default_rng(1000 + int(100 * T))
        for i in range(JITTER_STARTS):               # the ramp with every level moved by up to 7%
            starts.append((g * np.arange(4) * rng.uniform(0.93, 1.07, 4), g * np.arange(3, -1, -1) * rng.uniform(0.93, 1.07, 4),
                           f"jittered ramp {i}"))
    best = None; log = []
    for x, y, name in starts:
        x = np.asarray(x, float); y = np.asarray(y, float)
        f = cost_of(S, 12.0 * fam["ramp"]["cost"] + 2.0)
        c0 = f(np.concatenate([x, y]))
        if c0 >= 1e9:
            log.append({"start": name, "maxfev": maxfev, "cost": None})
            print(f"T={key} free from {name}: dropped, needs more than 12 times the ramp cost  [{time.time() - T0:.0f}s]", flush=True)
            continue
        s0 = c0 / ((x - x.min()).max() + (y - y.min()).max())
        c, v = nm_polish(f, np.concatenate([s0 * (x - x.min()), s0 * (y - y.min())]), maxfev)
        log.append({"start": name, "maxfev": maxfev, "start_cost": round(c0, 6), "cost": round(c, 6)})
        print(f"T={key} free from {name}: {c0:.6f} -> {c:.6f}  [{time.time() - T0:.0f}s]", flush=True)
        if best is None or c < best[0]:
            best = (c, v, name)
    how = "Nelder-Mead on 8 levels from the" if maxfev > 0 else "rescaled, no search:"
    fam["free"] = finish(*tables(best[1]), S, f"{how} {best[2]}")
    fl = chain_floor(S, EPS)
    return {"T": T, "best_family": "free", "families": fam, "search_log": log,
            "chain_floor": {"value": floor_down(fl["total"]), "raw": fl["total"],
                            "status": "proved pairwise bound for two separable channels, which one detector cannot "
                                      "beat (soft_core.chain_floor, evaluated in floats, rounded down); loose for "
                                      "one detector"}}


def run_search(args):
    if args and not args[0].isdigit():
        sys.exit("usage: --search [maxfev] [T ..]   maxfev is a whole number and comes first")
    maxfev = int(args[0]) if args else 400          # first argument is maxfev, the rest are temperatures
    keys = args[1:] or [str(t) for t in TS]
    bad = [k for k in keys if k not in SEEDS]
    if bad:
        sys.exit(f"unknown temperature {bad}; use {list(SEEDS)}")
    try:
        res = json.load(open(OUT))
    except FileNotFoundError:
        res = {"T": {}}
    res.update({"model": "one detector, N ~ Poisson(x[A] + y[B]); rows of q and s are A, decoder[n] by count",
                "eps": EPS, "margin": MARGIN,
                "status": "costs are numerical upper bounds (searched tables, LP decoder, stored decoder rechecked "
                          "in mpmath); for the floor see results/soft_floor_shared.json"})
    res["T"] = res.pop("T")
    for key in keys:
        old = res["T"].get(key)
        new = search(key, maxfev, old and old["families"])
        if old:                      # a new table replaces the stored one only if it is cheaper
            if old["families"]["free"]["cost"] <= new["families"]["free"]["cost"] + 1e-9:
                new["families"]["free"] = old["families"]["free"]
            keep = [e for e in old.get("search_log", []) if isinstance(e, dict)]
            new["search_log"] = keep + (new["search_log"] if maxfev > 0 else [])
        settle(new["families"])
        res["T"][key] = new
        res["T"] = dict(sorted(res["T"].items(), key=lambda kv: float(kv[0])))
        dump_json(res, OUT)
        print(f"T={key} cost {new['families']['free']['cost']:.4f} chain floor {new['chain_floor']['value']:.4f}", flush=True)


def specific_floor(key):
    """One detector bound from soft_floor_shared.py, or None if that file has nothing for this T."""
    try:
        r = json.load(open(FLOOR))["T"].get(key)
    except FileNotFoundError:
        return None
    return r["value"] if r and r.get("value") is not None else None


def verify():
    """Pass criterion for a cost: the mpmath recheck of the decoder stored on disk. The fresh LP
    is only compared (HiGHS optima on these problems move by more than MARGIN between builds)."""
    res = json.load(open(OUT))
    prev = None; notes = []
    for key, r in res["T"].items():
        S = target(4, float(key))
        fl = chain_floor(S, EPS)
        ok = abs(fl["total"] - r["chain_floor"]["raw"]) < 1e-8 and r["chain_floor"]["value"] <= fl["total"]
        sp = specific_floor(key)
        floor = max(r["chain_floor"]["value"], sp or 0.0)
        costs = []
        for f in ("ramp", "free"):
            e = r["families"][f]
            x = np.array(e["x"]); y = np.array(e["y"]); d = np.array(e["decoder"])
            t, _ = lp_shared(x, y, S)
            m = recheck_shared_mp(x, y, S, d)
            ok &= abs(x.max() + y.max() - e["cost"]) < 1e-12 and x.min() >= 0 and y.min() >= 0
            ok &= m <= EPS and abs(m - e["mp_error"]) < 1e-12
            if abs(t - e["lp_error"]) > 1e-7 or t > EPS:
                notes.append(f"T={key} {f}: fresh LP error {t:.9f}, stored {e['lp_error']:.9f}")
            ok &= len(d) == e["grid"] + 1 and floor <= e["cost"]
            costs.append(e["cost"])
        best = r["families"][r["best_family"]]
        ok &= best["cost"] <= min(costs) + 1e-12
        if key in SANITY:
            ok &= r["families"]["ramp"]["cost"] <= SANITY[key]
        if prev is not None and best["cost"] > prev + 1e-9:
            notes.append(f"T={key}: cost {best['cost']:.4f} is above the cost at the next lower temperature {prev:.4f}")
        prev = best["cost"]
        check(f"T={key} cost {best['cost']:.4f} floor {floor:.4f}", ok,
              f"family {r['best_family']} ({best['source']}) mp_error {best['mp_error']:.10f} ramp {r['families']['ramp']['cost']:.4f} "
              f"floor from {'soft_floor_shared.json (stored MILP bound, not solved again)' if sp and sp > r['chain_floor']['value'] else 'chain_floor'} "
              f"x {np.round(best['x'], 4).tolist()} y {np.round(best['y'], 4).tolist()}")
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
