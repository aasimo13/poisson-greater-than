"""Sampled comparison on two separable channels: cheapest level tables found, per temperature.

Model: N1 ~ Poisson(x[A]), N2 ~ Poisson(y[B]) independent, A, B in 0..3. The receiver outputs 1 with
probability d(N1, N2). Target s[A, B] = sigmoid((A - B) / T) (T = 0: 1 / 0.5 / 0); every pair must
have |q - s| <= 0.01. Cost = max(x) + max(y). Rows of q, s and of the decoder are A: channel 1
carries A, decoder[n1][n2]. The error target is fixed at 0.01; an EPS environment variable is ignored.

Families (each row is the minimum over itself and the rows above it):
  equal   x = y, lowest level 0
  free    x and y free, lowest level of each channel 0 (removing a channel's lowest level is
          undone by adding Poisson noise at the receiver, so it never hurts and lowers the cost)
The decoder is LP optimal for the table (soft_core.lp_two). Tables come from Nelder-Mead started
at the SEEDS below and at the stored tables, so every cost is a searched UPPER bound, not an
optimum, and the last digits still move between searches. A cost is certified by the mpmath
recheck of the stored decoder, not by the LP value. The floor is soft_core.chain_floor: a proved
pairwise (Neyman-Pearson) bound, evaluated in floats and rounded down; valid for any encoder and
decoder, not tight.

Usage (repo root):
  python3 src/soft_twochan.py                          verify results/soft_twochan.json: cost, fresh LP, mpmath
                                                       recheck of the stored decoders, floor (a few seconds)
  python3 src/soft_twochan.py --search [maxfev] [T ..] search again from the seeds and the stored tables, keep the
                                                       cheaper ones (minutes per T; maxfev 0 only reprices them)
"""
import json
import os
import sys
import time
import numpy as np
from soft_core import (target, lp_two, recheck_two_mp, budget_near, nm_polish, chain_floor,
                       floor_down, dump_json)

EPS = 0.01
MARGIN = 2e-8          # designs are scaled to LP error eps - MARGIN so the mpmath recheck passes strictly
TS = [0, 0.25, 0.5, 1, 2, 4]
OUT = os.environ.get("SOFT_TWOCHAN_OUT", "results/soft_twochan.json")
# loose upper sanity bounds for the equal row: decoder "1 if N1 > N2, coin on a tie" with x = y on a
# coarse grid of tables. closed_forms.py F4 has the exact T = 0 value (55.9065).
SANITY = {"0": 55.98, "0.5": 15.68}
# starting tables: equal = one table for both channels, free = (x, y), from earlier searches. A rerun
# also starts from the tables already stored in OUT, so repeated searches can only lower the costs.
SEEDS = {
    "0": {"equal": [0.00000000, 3.91202540, 13.22638513, 27.95325579],
          "free": [([0.00000000, 3.89225551, 12.51402456, 25.78210510], [0.00000000, 3.93141123, 13.75984512, 29.82872753])]},
    "0.25": {"equal": [0.00000000, 2.88289693, 9.39660451, 19.56332209],
             "free": [([0.00000000, 2.89679441, 9.17514373, 19.18918285], [0.00000000, 2.86880605, 9.58467400, 19.78267685])]},
    "0.5": {"equal": [0.00000000, 1.35322405, 3.93986914, 7.80196308],
            "free": [([0.00000000, 1.33302786, 3.68278991, 7.51299085], [0.00000000, 1.37306387, 4.18665017, 7.97631762])]},
    "1": {"equal": [0.00000000, 0.58360689, 1.47011309, 2.69872156],
          "free": [([0.00000000, 0.60340223, 1.54257385, 2.79749918], [0.00000000, 0.56341138, 1.39432393, 2.58139460])]},
    "2": {"equal": [0.00000000, 0.25478736, 0.58561008, 0.98035729],
          "free": [([0.00000000, 0.23459314, 0.56341094, 0.94956922], [0.00000000, 0.27458312, 0.60514544, 1.00703039])]},
    "4": {"equal": [0.00000000, 0.11775626, 0.26112026, 0.41302981],
          "free": [([0.00000000, 0.13079293, 0.27722034, 0.43283199], [0.00000000, 0.12036435, 0.24853345, 0.39282756])]},
}
FLOOR_STATUS = ("proved pairwise bound (soft_core.chain_floor, Neyman-Pearson chain), evaluated in floats, rounded "
                "down; valid for any encoder and decoder, not tight")
FAILS = []
T0 = time.time()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def lp_err(x, y, S):
    try:
        return lp_two(x, y, S)[0]
    except RuntimeError:
        return 1.0            # solver failure counts as "does not meet eps": the cost can only go up


def finish(x, y, S, source):
    """Scale the shape to the smallest cost with LP error <= eps - MARGIN, solve, recheck in mpmath."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    c = budget_near(lambda c: lp_err(c * x, c * y, S), 1.0, EPS - MARGIN, rel=1e-8)
    while True:
        xs, ys = c * x, c * y
        t, d, M = lp_two(xs, ys, S)
        m, Q, _ = recheck_two_mp(xs, ys, S, d, full=True)
        if m <= EPS:
            break
        c *= 1.0 + 1e-6
    return {"x": xs.tolist(), "y": ys.tolist(), "cost": float(xs.max() + ys.max()), "lp_error": float(t),
            "mp_error": float(m), "source": source, "grid": [int(M[0]), int(M[1])], "q": Q.tolist(),
            "decoder": d.tolist()}


def tables(family, v):
    v = np.abs(np.asarray(v, float))
    if family == "equal":
        x = np.concatenate([[0.0], v]); return x, x
    x, y = v[:4], v[4:]
    return x - x.min(), y - y.min()


def cost_of(family, S):
    def f(v):
        x, y = tables(family, v)
        c1 = x.max() + y.max()
        if c1 <= 0:
            return 1e9
        return c1 * budget_near(lambda c: lp_err(c * x, c * y, S), 1.0, EPS - MARGIN, rel=2e-6)
    return f


def settle(fam):
    """If no wider table beat the equal row, the free row IS the equal row, and says so."""
    if fam["equal"]["cost"] <= fam["free"]["cost"] + 1e-9:
        fam["free"] = dict(fam["equal"], source="equal row (the wider search did not beat it)")


def search(key, maxfev, old=None):
    T = float(key); S = target(4, T)
    fam = {}
    f = cost_of("equal", S)
    eq = [SEEDS[key]["equal"][1:]] + ([old["equal"]["x"][1:]] if old else [])
    c, v = min((nm_polish(f, f(np.array(e)) / (2 * max(e)) * np.array(e), maxfev) for e in eq), key=lambda r: r[0])
    fam["equal"] = finish(*tables("equal", v), S, "Nelder-Mead on 3 levels" if maxfev > 0 else "seed, rescaled, no search")
    print(f"T={key} equal {fam['equal']['cost']:.6f}  [{time.time() - T0:.0f}s]", flush=True)
    best = None
    starts = [(fam["equal"]["x"], fam["equal"]["y"], "equal optimum")] + [(x, y, "seed") for x, y in SEEDS[key]["free"]]
    if old:
        starts.append((old["free"]["x"], old["free"]["y"], "stored table"))
    for x, y, name in starts:
        x = np.asarray(x, float); y = np.asarray(y, float)
        s0 = budget_near(lambda c: lp_err(c * x, c * y, S), 1.0, EPS - MARGIN, rel=2e-6)
        c, v = nm_polish(cost_of("free", S), np.concatenate([s0 * x, s0 * y]), maxfev)
        print(f"T={key} free from {name}: {c:.6f}  [{time.time() - T0:.0f}s]", flush=True)
        if best is None or c < best[0]:
            best = (c, v, name)
    how = "Nelder-Mead on 8 levels from the" if maxfev > 0 else "rescaled, no search:"
    fam["free"] = finish(*tables("free", best[1]), S, f"{how} {best[2]}")
    fl = chain_floor(S, EPS)
    return {"T": T, "best_family": "free", "families": fam,
            "floor": {"value": floor_down(fl["total"]), "raw": fl["total"], "levels": fl["levels"],
                      "status": FLOOR_STATUS}}


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
    res.update({"model": "two channels, N1 ~ Poisson(x[A]), N2 ~ Poisson(y[B]); rows of q, s, decoder are A",
                "eps": EPS, "margin": MARGIN,
                "status": "costs are numerical upper bounds (searched tables, LP decoder, stored decoder rechecked "
                          "in mpmath); floors are proved pairwise bounds, not tight"})
    res["T"] = res.pop("T")
    for key in keys:
        old = res["T"].get(key)
        new = search(key, maxfev, old and old["families"])
        if old:                      # a new table replaces the stored one only if it is cheaper
            for f in ("equal", "free"):
                if old["families"][f]["cost"] <= new["families"][f]["cost"] + 1e-9:
                    new["families"][f] = old["families"][f]
        settle(new["families"])
        res["T"][key] = new
        res["T"] = dict(sorted(res["T"].items(), key=lambda kv: float(kv[0])))
        dump_json(res, OUT)
        print(f"T={key} cost {new['families']['free']['cost']:.4f} floor {new['floor']['value']:.4f}", flush=True)


def verify():
    """Pass criterion for a cost: the mpmath recheck of the decoder stored on disk. The fresh LP
    is only compared (HiGHS optima on these problems move by more than MARGIN between builds)."""
    res = json.load(open(OUT))
    prev = None; notes = []
    for key, r in res["T"].items():
        S = target(4, float(key))
        fl = chain_floor(S, EPS)
        ok = abs(fl["total"] - r["floor"]["raw"]) < 1e-8 and r["floor"]["value"] <= fl["total"]
        rows = []
        for f in ("equal", "free"):
            e = r["families"][f]
            x = np.array(e["x"]); y = np.array(e["y"]); d = np.array(e["decoder"])
            t, _, M = lp_two(x, y, S)
            m = recheck_two_mp(x, y, S, d)
            ok &= abs(x.max() + y.max() - e["cost"]) < 1e-12 and x.min() >= 0 and y.min() >= 0
            ok &= m <= EPS and abs(m - e["mp_error"]) < 1e-12
            if abs(t - e["lp_error"]) > 1e-7 or t > EPS:
                notes.append(f"T={key} {f}: fresh LP error {t:.9f}, stored {e['lp_error']:.9f}")
            ok &= list(d.shape) == [e["grid"][0] + 1, e["grid"][1] + 1] and r["floor"]["value"] <= e["cost"]
            if f == "equal":
                ok &= e["x"] == e["y"]
            rows.append((f, e["cost"], m))
        best = r["families"][r["best_family"]]
        ok &= best["cost"] <= min(c for _, c, _ in rows) + 1e-12
        if key in SANITY:
            ok &= r["families"]["equal"]["cost"] <= SANITY[key]
        if prev is not None and best["cost"] > prev + 1e-9:
            notes.append(f"T={key}: cost {best['cost']:.4f} is above the cost at the next lower temperature {prev:.4f}")
        prev = best["cost"]
        check(f"T={key} cost {best['cost']:.4f} floor {r['floor']['value']:.4f}", ok,
              f"family {r['best_family']} ({best['source']}) mp_error {best['mp_error']:.10f} equal {r['families']['equal']['cost']:.4f} "
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
