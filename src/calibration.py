"""When does calibrating the gain erase the drift advantage?

Model: the gain is unknown within +-D (D = 5%) but changes slowly. Every T decisions a pilot pulse
of M detected photons from a known reference measures it; the estimate's relative error is 1/sqrt(M),
so the residual uncertainty is d = min(z/sqrt(M), D) with z = 3. Cost per decision =
P(d) (worst case peak signal photons, robust decoder for +-d, from re-optimized designs) + M/T.
P(d) is interpolated linearly between the re-optimized points below (opt.py results).
"""
import numpy as np
D, Z = 0.05, 3.0
P = {  # d: (banded, threshold family), both families re-optimized for each d
 0: {0.0: (196.6, 321.54), 0.005: (200.79, 340.36), 0.01: (205.07, 359.98), 0.02: (213.7, 404.4), 0.05: (245.4, 598.2)},
 1: {0.0: (290.6, 323.97), 0.005: (297.34, 342.62), 0.01: (304.1, 362.31), 0.02: (318.7, 406.49), 0.05: (366.72, 600.54)},
}
def Pd(bg, fam, d):
    ds = sorted(P[bg]); vals = [P[bg][k][fam] for k in ds]
    return float(np.interp(d, ds, vals))
def best(bg, fam, T):
    Ms = np.concatenate([[0], np.logspace(0, 9, 4000)])
    costs = [Pd(bg, fam, min(Z / np.sqrt(M), D) if M > 0 else D) + M / T for M in Ms]
    i = int(np.argmin(costs)); return costs[i], Ms[i]
print("bg  T(decisions per calibration)  banded  threshold  saving   pilot M (banded, threshold)")
for bg in (0, 1):
    for T in [1, 100, 1e3, 1e4, 1e5, 1e6, 1e7]:
        cb, mb = best(bg, 0, T); ct, mt = best(bg, 1, T)
        print(f"{bg:<3} {T:<10.0e} {cb:8.1f} {ct:9.1f} {100*(1-cb/ct):7.1f}%   {mb:10.0f} {mt:10.0f}")
