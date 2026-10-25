"""Independent check of drift + background designs.

For each design: fit the robust LP decoder on a 41 point shared gain grid with background,
then evaluate that fixed decoder at 401 gains across [1-d, 1+d] (between grid points too).
Reports the worst error and the worst case peak signal mean (max x + max y)(1 + d).
Does not use opt.py; only core.py's Poisson pmf and LP.
"""
import sys, json
import numpy as np
from core import labels, lp_minimax, pmf_matrix

LAB = labels(4)


def check(x, y, d, bg):
    x = np.asarray(x, float); y = np.asarray(y, float)
    gs = np.linspace(1 - d, 1 + d, 41)
    means = np.concatenate([(g * x[:, None] + g * y[None, :]).ravel() + bg for g in gs])
    lab = np.tile(LAB, len(gs))
    e, dec = lp_minimax(means, lab, return_decoder=True)
    nb = len(dec) - 1
    worst = 0.0
    for g in np.linspace(1 - d, 1 + d, 401):
        m = (g * x[:, None] + g * y[None, :]).ravel() + bg
        P = pmf_matrix(m, nb)
        err = np.where(LAB == 1, 1 - P @ dec, P @ dec)
        worst = max(worst, err.max())
    return e, worst, (x.max() + y.max()) * (1 + d)


if __name__ == "__main__":
    best = {}
    for line in open("results/bgdrift.jsonl"):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r["d"] == 0:
            continue
        fam = "free" if r["family"] == "free" and r["x"][2] < 1 else "monotone"
        key = (r["bg"], r["d"], fam)
        if key not in best or r["verified_cost"] < best[key]["verified_cost"]:
            best[key] = r
    for key in sorted(best):
        r = best[key]
        e, w, c = check(r["x"], r["y"], r["d"], r["bg"])
        print(json.dumps({"bg": key[0], "d": key[1], "family": key[2],
                          "claimed": round(r["verified_cost"], 2), "recomputed_cost": round(c, 2),
                          "err_41grid_pct": round(100 * e, 4), "err_dense_pct": round(100 * w, 4)}),
              flush=True)
