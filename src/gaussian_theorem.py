"""Analytic Gaussian GREATER-THAN optimum for every alphabet size K >= 2.

Model: Z = x[A] + y[B] + Normal(0, sigma**2), nonnegative deterministic
encoders, any randomized receiver, error <= epsilon for EACH input pair.
The optimum of max(x) + max(y) is 4*(K-1)*sigma*norm.isf(epsilon).
This is an amplitude cost in the Gaussian control, not a photon count.
Proof and scope: docs/theory.md. No solver or novelty claim is needed.

Run: python3 src/gaussian_theorem.py --max-k 8
"""
import argparse
import json
import numbers

import numpy as np
from scipy.stats import norm


def required_gap(epsilon=0.01, sigma=1.0):
    """Minimum mean separation for a two-point equal-variance Gaussian test."""
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and strictly between 0 and 0.5")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive")
    gap = float(2 * sigma * norm.isf(epsilon))
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("parameters exceed the supported floating-point range")
    return gap


def optimal_encoders(K, epsilon=0.01, sigma=1.0):
    """Return attaining x, y and threshold; say TRUE when Z < threshold."""
    if isinstance(K, bool) or not isinstance(K, numbers.Integral) or K < 2:
        raise ValueError("K must be an integer at least 2")
    d = required_gap(epsilon, sigma)
    x = np.arange(K - 1, -1, -1, dtype=float) * d
    y = np.arange(K, dtype=float) * d
    threshold = (K - 1.5) * d
    if not np.isfinite(x).all() or not np.isfinite(threshold):
        raise ValueError("parameters exceed the supported floating-point range")
    return x, y, float(threshold)


def threshold_errors(x, y, threshold, sigma=1.0):
    """Exact-distribution floating-point error for each pair, TRUE below cut."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(threshold) or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("finite equal-length encoders and positive sigma required")
    means = x[:, None] + y[None, :]
    z = (threshold - means) / sigma
    true = np.arange(len(x))[:, None] > np.arange(len(y))[None, :]
    return np.where(true, norm.sf(z), norm.cdf(z))


def verify_construction(K, epsilon=0.01, sigma=1.0):
    """Evaluate every pair, independently of the algebraic cost expression."""
    x, y, threshold = optimal_encoders(K, epsilon, sigma)
    errors = threshold_errors(x, y, threshold, sigma)
    return {
        "K": int(K),
        "epsilon": float(epsilon),
        "sigma": float(sigma),
        "required_gap": required_gap(epsilon, sigma),
        "optimal_amplitude_cost": float(x.max() + y.max()),
        "threshold": threshold,
        "worst_pair_error": float(errors.max()),
        "attaining_x": x.tolist(),
        "attaining_y": y.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_k < 2:
        parser.error("--max-k must be at least 2")
    try:
        rows = [verify_construction(k, args.epsilon, args.sigma)
                for k in range(2, args.max_k + 1)]
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"model": "nonnegative additive encoders; Gaussian noise",
                      "theorem": "P*=4*(K-1)*sigma*Phi_inverse(1-epsilon)",
                      "cost_unit": "Gaussian signal amplitude, not photons",
                      "rows": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
