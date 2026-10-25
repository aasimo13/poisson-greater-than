"""Collect ladder-chart points: peak expected signal photons at 1% error versus background per channel or slot.

Lines: threshold (one detector, monotone design), banded (one detector), two separable channels (each channel
gets the background), arrival time (4 slots, each gets the background). Background is not counted in the cost.
Equal background values therefore do not match total background, observation time, or readout resources.
Sources:
  threshold, banded at bg 0   : results/leakage.json era designs, src/threshold_optimum.py (321.54) and src/banded.py (196.59)
  threshold, banded at bg > 0 : results/ladder_opt.jsonl (queues/qLadder.sh, src/opt.py with BG), nominal gain
  two channels                : computed here, stored 106.4 design rescaled, 2D LP, bisection
  arrival time                : results/timecomp.json (src/timecomp.py)
Lower bounds at bg 0: threshold 321.5366 (src/threshold_optimum.py), banded 195.53 (lb_pairwise2.py 4 200 1500 8),
two channels 92.293135 (src/twotype_bound.py), one-hot arrival slots 2*log(50) = 7.824046 (docs/theory.md).
The historical arrival-time local-search value 9.11 is an uncertified diagnostic, not a plotted lower bound.
Writes results/ladder.json.   Usage: python3 src/ladder_points.py
"""
import json, os
import numpy as np
from scipy.optimize import linprog
from core import labels, pmf_matrix, scale_to_target
from timecomp import analytic_arrival_floor

BGS = [0.1, 1.0, 3.0]
X0 = np.array([0.001, 4.701, 22.414, 52.805]); Y0 = np.array([0.001, 5.005, 23.512, 53.561])


def two_chan_bg(bg, eps=0.01):
    lab = labels(4)

    def err(s):
        Px = pmf_matrix(s * X0 + bg); Py = pmf_matrix(s * Y0 + bg)
        P = np.array([np.outer(Px[a], Py[b]).ravel() for a in range(4) for b in range(4)])
        H, M = P.shape
        c = np.zeros(M + 1); c[-1] = 1
        A = np.zeros((H, M + 1)); bb = np.zeros(H); T = lab == 1
        A[T, :M] = -P[T]; A[T, -1] = -1; bb[T] = -1
        A[~T, :M] = P[~T]; A[~T, -1] = -1
        r = linprog(c, A_ub=A, b_ub=bb, bounds=[(0, 1)] * (M + 1), method="highs")
        if not r.success:
            raise RuntimeError(r.message)
        return r.x[-1]

    s = scale_to_target(err, 1e-3, 1.0, target=eps, rel=1e-4)
    return s * (X0.max() + Y0.max())


if __name__ == "__main__":
    lines = {"threshold": [], "banded": [], "two_channel": [], "arrival_time": []}
    lines["threshold"].append({"bg": 0.0, "cost": 321.540, "source": "src/threshold_optimum.py (reproduce.sh check)"})
    lines["banded"].append({"bg": 0.0, "cost": 196.62, "source": "src/banded.py K=4 exact LP budget (reproduce.sh check)"})
    lines["two_channel"].append({"bg": 0.0, "cost": 106.366, "source": "results/twotype_bound.txt stored design, src/price_curve.py"})
    # one detector designs from the opt.py runs
    best = {}
    for l in open("results/ladder_opt.jsonl"):
        l = l.strip()
        if not l.startswith("{"):
            continue
        r = json.loads(l)
        key = ("threshold" if r["family"] == "monotone" else "banded", r["bg"])
        if r["verified_err"] <= 0.0100001 and (key not in best or r["verified_cost"] < best[key]["cost"]):
            best[key] = {"bg": r["bg"], "cost": r["verified_cost"], "err": r["verified_err"], "x": r["x"], "y": r["y"],
                         "source": f"results/ladder_opt.jsonl (src/opt.py {r['family']} nominal, BG={r['bg']})"}
    for (name, bg), v in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines[name].append(v)
    for bg in BGS:
        c = two_chan_bg(bg)
        print("two channel bg", bg, round(c, 3), flush=True)
        lines["two_channel"].append({"bg": bg, "cost": c, "source": "src/ladder_points.py: stored 106.4 design rescaled, 2D LP, bisection"})
    T = json.load(open("results/timecomp.json"))
    lines["arrival_time"].append({"bg": 0.0, "cost": T["winner"]["cost"], "source": "results/timecomp.json winner"})
    for b, c in sorted(T["beta_costs"].items(), key=lambda kv: float(kv[0])):
        lines["arrival_time"].append({"bg": float(b), "cost": c, "source": "results/timecomp.json beta_costs (fixed shape rescale, upper bound)"})
    floors = {"threshold": {"bg": 0.0, "cost": 321.5366, "source": "src/threshold_optimum.py certified bound"},
              "banded": {"bg": 0.0, "cost": 195.53, "source": "reproduce.sh full: lb_pairwise2.py 4 200 1500 8"},
              "two_channel": {"bg": 0.0, "cost": 92.293135, "source": "results/twotype_bound.txt"},
              "arrival_time": {"bg": 0.0, "cost": analytic_arrival_floor()["value"],
                               "formula": "2 * log(1 / (2 * error_target))",
                               "status": "analytic necessary bound for one-hot arrival slots, evaluated in floating point",
                               "source": "docs/theory.md; results/timecomp.json analytic_floor"}}
    json.dump({"error_target": 0.01, "cost": "max expected signal photons, background excluded",
               "background": "counts per counted channel or slot; 1 for threshold/banded, 2 for two_channel, 4 for arrival_time",
               "resource_caveat": "Equal background values do not match total background, observation time, or readout resources.",
               "lines": lines, "certified_floor": floors,
               "uncertified_diagnostics": {"arrival_time_pairwise_search": T["pairwise_search_diagnostic"]}},
              open("results/ladder.json", "w"), indent=1)
    for k, v in lines.items():
        print(k, [(p["bg"], round(p["cost"], 2)) for p in v])
