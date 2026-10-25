"""The "peeled band" construction for alphabet size K.

Levels are numbered from the top, j = 0..K-1.
  even j : holds x[j/2] and y[K-1-j/2]      (band FALSE)
  odd  j : holds x[K-1-(j-1)/2] and y[(j-1)/2]   (band TRUE)
Brightness values from the bottom: v_0 = 0, v_{k+1} = h(2 v_k); level j gets v_{K-1-j}.
Both transmitters at a level emit v, so a pair's mean is the sum of its two levels' values
and is dominated by the brighter item; the decoder reads off that item's band label.

For each K the script checks that EVERY pair (A,B) lands in the band of its brighter item and
that this band's label equals [A > B]; then prints the pairwise budget 2 v_{K-1} and the exact
budget after rescaling the shape with the LP decoder (core.budget_for_shape).

Usage: python3 src/banded.py [Kmax]     (default Kmax = 8; LP step for K=8 takes a few minutes)
"""
import sys, time
import numpy as np
from core import h, budget_for_shape


def build(K, hfun=h):
    """hfun(mu) = smallest nu that two point tests separate from mu (Poisson h by default)."""
    v = [0.0]
    for _ in range(K - 1):
        v.append(hfun(2 * v[-1]))
    level_x = {}; level_y = {}
    for j in range(K):
        if j % 2 == 0:
            level_x[j // 2] = j; level_y[K - 1 - j // 2] = j
        else:
            level_x[K - 1 - (j - 1) // 2] = j; level_y[(j - 1) // 2] = j
    x = np.array([v[K - 1 - level_x[a]] for a in range(K)])
    y = np.array([v[K - 1 - level_y[b]] for b in range(K)])
    return v, x, y, level_x, level_y


def check_band_rule(K, level_x, level_y):
    assert sorted(level_x) == list(range(K)) and sorted(level_y) == list(range(K))
    assert sorted(level_x.values()) == list(range(K)) and sorted(level_y.values()) == list(range(K))
    bad = 0
    for a in range(K):
        for b in range(K):
            j = min(level_x[a], level_y[b])           # brighter item = smaller level index
            band_true = (j % 2 == 1)
            if band_true != (a > b):
                bad += 1
    return bad


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(" K  band rule violations  pairwise budget 2*v_{K-1}   exact LP budget   (seconds)")
    for K in range(2, kmax + 1):
        v, x, y, lx, ly = build(K)
        bad = check_band_rule(K, lx, ly)
        t = time.time()
        P, s = budget_for_shape(x, y)
        ex = f"{P:12.2f}"
        print(f"{K:2d}  {bad:12d}            {2 * v[K - 1]:12.2f}              {ex}      {time.time() - t:6.1f}", flush=True)
        if K == 4:
            print("      K=4 shape: x =", np.round(x, 3).tolist(), " y =", np.round(y, 3).tolist())
