"""Independent verification of the 201 photon candidate encoding.

Encoding  x = [100.5, 4.7, 0, 29.4],  y = [29.4, 0, 4.7, 100.5]
Decoder   say "A > B" iff N == 0 or 18 <= N <= 77 (deterministic, N = detected photon count).

Prints all 16 pair errors with scipy and with mpmath (50 digits), then the best randomized
(LP) decoder on the same encoding and the budget after rescaling the shape with the LP decoder.
Usage: python3 src/verify_candidate.py
"""
import numpy as np
import mpmath as mp
from scipy.stats import poisson
from core import labels, pair_means, lp_minimax, budget_for_shape

X = [100.5, 4.7, 0.0, 29.4]
Y = [29.4, 0.0, 4.7, 100.5]
K = 4
ACCEPT = [0] + list(range(18, 78))     # counts on which the decoder says "A > B"

mp.mp.dps = 50


def err_scipy(m, true):
    p_acc = sum(poisson.pmf(n, m) for n in ACCEPT)
    return (1.0 - p_acc) if true else p_acc


def err_mpmath(m, true):
    m = mp.mpf(m)
    p_acc = mp.fsum(mp.e ** (-m) * m ** n / mp.factorial(n) for n in ACCEPT)
    return (1 - p_acc) if true else p_acc


def main():
    lab = labels(K)
    print("pair (A,B)  mean   truth   scipy err %      mpmath err % (50 digits)")
    ws, wm, worst_pair = 0.0, mp.mpf(0), None
    for a in range(K):
        for b in range(K):
            m = X[a] + Y[b]
            t = a > b
            es, em = err_scipy(m, t), err_mpmath(mp.mpf(X[a]) + mp.mpf(Y[b]), t)
            print(f"({a},{b})   {m:7.2f}  {str(t):5s}  {100*es:.10f}   {mp.nstr(100*em, 20)}")
            if es > ws:
                ws, worst_pair = es, (a, b)
            wm = max(wm, em)
    maxmean = max(X) + max(Y)
    print()
    print(f"WORST deterministic decoder error: scipy {100*ws:.6f}%  mpmath {mp.nstr(100*wm, 12)}%  at (A,B)={worst_pair}")
    print(f"max mean = {maxmean:.1f}")

    m = pair_means(X, Y)
    e_lp, dec = lp_minimax(m, lab, return_decoder=True)
    print(f"LP (randomized) decoder on the same encoding: worst error {100*e_lp:.4f}%")
    P, s = budget_for_shape(X, Y)
    print(f"Budget after rescaling the shape with the LP decoder (core.budget_for_shape): {P:.2f} (scale {s:.5f})")
    print("(published figure 196.75; the current core.py gives about 196.69)")


if __name__ == "__main__":
    main()
