"""How much does the photon count reveal about A and B beyond the answer [A > B]?

Inputs A, B uniform on {0..K-1} (each of the 16 pairs equally likely for K = 4).
The receiver sees N ~ Poisson(x[A] + y[B]). The answer F = [A > B] is a function of (A, B), so
    I(N; A,B) = I(N; F) + I(N; A,B | F)
and I(N; A,B | F) is the leakage beyond the answer (0 for a perfectly private computation).
Also reported: per party leakage I(N; A | F), I(N; B | F), and the receiver's best chance of
guessing the exact pair (A, B), or A alone, from N, against the chance from the answer alone.
"""
import json, sys
import numpy as np
from scipy.stats import poisson
from core import labels, budget_for_shape

K = 4


def H(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def leakage(x, y, bg=0.0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    pairs = [(a, b) for a in range(K) for b in range(K)]
    F = np.array([1 if a > b else 0 for a, b in pairs])
    m = np.array([x[a] + y[b] for a, b in pairs]) + bg
    nmax = int(m.max() + 12 * np.sqrt(m.max() + 1) + 60)
    n = np.arange(nmax)
    L = poisson.pmf(n[None, :], m[:, None])                # P(N = n | pair)
    L = np.hstack([L, np.clip(1 - L.sum(1), 0, None)[:, None]])
    prior = np.full(len(pairs), 1.0 / len(pairs))
    joint = prior[:, None] * L                             # P(pair, n)
    pN = joint.sum(0)
    out = {}
    # I(N; pair) = H(N) - H(N | pair)
    HN = H(pN)
    HN_pair = float(sum(prior[i] * H(L[i]) for i in range(len(pairs))))
    I_total = HN - HN_pair
    # I(N; F)
    pNF = np.vstack([joint[F == f].sum(0) for f in (0, 1)])
    pF = pNF.sum(1)
    HN_F = float(sum(pF[f] * H(pNF[f] / pF[f]) for f in (0, 1)))
    I_F = HN - HN_F
    out["I_total_bits"] = I_total
    out["I_answer_bits"] = I_F
    out["leak_beyond_answer_bits"] = I_total - I_F
    # per party: I(N; A | F) = H(N|F) - H(N|A,F)
    for name, idx in (("A", 0), ("B", 1)):
        HN_givenAF = 0.0
        for v in range(K):
            for f in (0, 1):
                sel = np.array([(p[idx] == v) and (F[i] == f) for i, p in enumerate(pairs)])
                if not sel.any():
                    continue
                w = prior[sel].sum()
                HN_givenAF += w * H(joint[sel].sum(0) / w)
        out[f"leak_{name}_beyond_answer_bits"] = HN_F - HN_givenAF
    # best guess of the exact pair: from N, and from the answer alone
    out["guess_pair_from_N"] = float(joint.max(0).sum())
    out["guess_pair_from_answer"] = float(sum(prior[F == f].max() for f in (0, 1)))
    jA = np.vstack([joint[[p[0] == v for p in pairs]].sum(0) for v in range(K)])
    out["guess_A_from_N"] = float(jA.max(0).sum())
    out["guess_A_from_answer"] = float(sum(
        max(prior[[(p[0] == v) and (F[i] == f) for i, p in enumerate(pairs)]].sum() for v in range(K))
        for f in (0, 1)))
    out["peak_photons"] = float(m.max() - bg)
    out["distinct_levels"] = int(len(np.unique(np.round(m, 6))))
    return out


DESIGNS = {
    "threshold (best single threshold)": ([1.5, 1, 0.5, 0], [0, 0.5, 1, 1.5]),
    "banded (best known)": ([98.2976, 4.6013, 0, 28.7852], [28.7852, 0, 4.6013, 98.2976]),
    "ChatGPT candidate (deterministic)": ([100.5, 4.7, 0, 29.4], [29.4, 0, 4.7, 100.5]),
}

if __name__ == "__main__":
    res = {}
    for name, (x, y) in DESIGNS.items():
        x = np.array(x, float); y = np.array(y, float)
        if "candidate" not in name:
            P, s = budget_for_shape(x, y)       # scale to exactly 1% worst case error
            x, y = s * x, s * y
        r = leakage(x, y)
        res[name] = r
        print(name)
        for k, v in r.items():
            print(f"   {k:32s} {v:.4f}" if isinstance(v, float) else f"   {k:32s} {v}")
    import os
    json.dump(res, open(os.environ.get("LEAK_OUT", "results/leakage.json"), "w"), indent=1)
