"""Evaluate actual decoded-answer feedback in the one-step spillover model.

The original mc.py answer-feedback LP is trained with the *correct* previous
comparison supplied. Here the same policy is also run with its own previous
decoded bit. A 32-state Markov chain (previous input pair, decoded bit) gives
stationary and startup probabilities without Monte Carlo sampling.

Inputs are IID uniform unless a deterministic periodic stress sequence is
explicitly requested. Counts in different windows are conditionally independent
given the input sequence. Spillover is rho times the previous *mean*, not a
simulation of molecular trajectories or reaction kinetics. All numbers are
floating-point probability calculations; this is not an interval certificate.

Run from the repository root:
    python3 src/feedback_sequence.py
    python3 -m unittest discover -s tests -p 'test_feedback_sequence.py'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.stats import poisson


ROOT = Path(__file__).resolve().parents[1]


def pair_labels(k):
    return np.array([a > b for a in range(k) for b in range(k)], dtype=int)


def poisson_rows(means, cutoff):
    """Bins 0..cutoff-1 and one exact-probability bin N >= cutoff.

    The returned decoder is constant on the final bin. Aggregating it therefore
    defines a valid policy for the full count space rather than dropping tails.
    Normalize only the tiny floating-point PMF summation error and report it.
    """
    means = np.asarray(means, dtype=float)
    pm = poisson.pmf(np.arange(cutoff), means[..., None])
    tail = poisson.sf(cutoff - 1, means)
    p = np.concatenate([pm, tail[..., None]], axis=-1)
    raw_mass = p.sum(axis=-1, keepdims=True)
    correction = float(np.max(np.abs(raw_mass - 1)))
    if correction > 1e-9:
        raise ArithmeticError(f"Poisson row normalization error {correction}")
    return p / raw_mass, float(np.max(tail)), correction


def observation_tensor(x, y, rho):
    """P[previous pair, current pair, count bin], including an overflow bin."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("x and y must be equal-length vectors with K >= 2")
    if np.any(x < 0) or np.any(y < 0) or rho < 0:
        raise ValueError("Intensities and spillover must be nonnegative")
    means = (x[:, None] + y[None, :]).ravel()
    joint_means = rho * means[:, None] + means[None, :]
    maximum = float(joint_means.max())
    cutoff = int(np.ceil(maximum + 12 * np.sqrt(maximum + 1) + 60))
    p, tail, correction = poisson_rows(joint_means, cutoff)
    return p, {
        "cutoff": cutoff,
        "overflow_definition": "N >= cutoff; decoder constant on this bin",
        "maximum_overflow_probability": tail,
        "maximum_row_normalization_correction": correction,
    }


def fit_decoder(training, labels):
    """Minimax LP for P[current pair, side-information bit, count bin]."""
    training = np.asarray(training, float)
    labels = np.asarray(labels, int)
    h, z, n = training.shape
    p = training.reshape(h, z * n)
    if np.max(np.abs(p.sum(axis=1) - 1)) > 1e-10:
        raise ValueError("Training probability rows must sum to one")
    c = np.zeros(z * n + 1)
    c[-1] = 1
    a = np.empty((h, z * n + 1))
    a[:, :-1] = np.where(labels[:, None] == 1, -p, p)
    a[:, -1] = -1
    b = np.where(labels == 1, -1.0, 0.0)
    result = linprog(
        c, A_ub=a, b_ub=b, bounds=[(0, 1)] * (z * n + 1),
        method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                 "dual_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        raise RuntimeError(result.message)
    decoder = np.clip(result.x[:-1].reshape(z, n), 0, 1)
    p_true = np.einsum("jzn,zn->j", training, decoder)
    errors = np.where(labels == 1, 1 - p_true, p_true)
    return decoder, errors, float(result.fun)


def train_policies(p, labels):
    """Fit no feedback and correct-previous-answer policies on the same levels."""
    h = len(labels)
    no_side = p.mean(axis=0)[:, None, :]
    d_none, err_none, lp_none = fit_decoder(no_side, labels)
    d_none = np.repeat(d_none, 2, axis=0)
    oracle = np.stack([p[labels == z].sum(axis=0) / h for z in (0, 1)], axis=1)
    d_oracle, err_oracle, lp_oracle = fit_decoder(oracle, labels)
    return {
        "no_feedback": (d_none, err_none, lp_none),
        "correct_previous_answer": (d_oracle, err_oracle, lp_oracle),
    }


def output_probabilities(p, decoder):
    """q[i,j,z] = Pr(decoded current bit = 1 | previous i,z; current j)."""
    return np.clip(np.einsum("ijn,zn->ijz", p, decoder), 0, 1)


def transition_matrix(q, prior=None):
    """Transition on flattened states (pair i, decoded bit z), index 2*i+z."""
    h = q.shape[0]
    prior = np.full(h, 1 / h) if prior is None else np.asarray(prior, float)
    if prior.shape != (h,) or np.any(prior < 0) or abs(prior.sum() - 1) > 1e-12:
        raise ValueError("prior must be a probability vector")
    m = np.empty((2 * h, 2 * h))
    for i in range(h):
        for z in (0, 1):
            m[2 * i + z, 0::2] = prior * (1 - q[i, :, z])
            m[2 * i + z, 1::2] = prior * q[i, :, z]
    return m


def stationary_distribution(m, initial=None, tolerance=1e-14, max_iter=10000):
    """Power iteration, with a residual check; failure is explicit."""
    m = np.asarray(m, float)
    if np.any(m < -1e-14) or np.max(np.abs(m.sum(axis=1) - 1)) > 1e-11:
        raise ValueError("Transition matrix must be row stochastic")
    w = np.full(len(m), 1 / len(m)) if initial is None else np.asarray(initial, float).copy()
    if np.any(w < 0) or abs(w.sum() - 1) > 1e-12:
        raise ValueError("Initial state must be a probability vector")
    for step in range(1, max_iter + 1):
        nxt = w @ m
        if np.linalg.norm(nxt - w, ord=1) <= tolerance:
            residual = float(np.linalg.norm(nxt @ m - nxt, ord=1))
            return nxt, step, residual
        w = nxt
    raise RuntimeError("Stationary iteration failed to converge")


def conditional_errors(q, labels):
    return np.where(np.asarray(labels)[None, :, None] == 1, 1 - q, q)


def errors_given_state(q, labels, state):
    """Worst-current-pair calculation averages over the supplied history state."""
    h = len(labels)
    w = np.asarray(state).reshape(h, 2)
    return np.einsum("iz,ijz->j", w, conditional_errors(q, labels))


def oracle_initial(labels):
    h = len(labels)
    w = np.zeros((h, 2))
    w[np.arange(h), labels] = 1 / h
    return w.ravel()


def fresh_initial(x, y, cutoff):
    """First actual message has no predecessor; fit a no-spillover decoder."""
    k = len(x)
    labels = pair_labels(k)
    means = (np.asarray(x)[:, None] + np.asarray(y)[None, :]).ravel()
    p, _, _ = poisson_rows(means, cutoff)
    decoder, errors, _ = fit_decoder(p[:, None, :], labels)
    p_true = p @ decoder[0]
    w = np.column_stack([1 - p_true, p_true]) / len(labels)
    return w.ravel(), errors


def startup_trace(q, labels, m, initial, steps=12):
    w = np.asarray(initial, float).copy()
    trace = []
    for step in range(1, steps + 1):
        errors = errors_given_state(q, labels, w)
        trace.append({"subsequent_window": step,
                      "worst_current_pair_error": float(errors.max()),
                      "mean_error": float(errors.mean()),
                      "per_current_pair_error": errors.tolist()})
        w = w @ m
    return trace


def startup_future_bound(q, labels, m, initial, stationary, steps=12):
    """Bound all later IID-history-averaged errors after a finite prefix.

    Total variation contracts under a Markov kernel. Correct the computed
    stationary vector using its residual and the Dobrushin contraction factor.
    The arithmetic remains floating point, not outward-rounded certification.
    """
    w = np.asarray(initial, float).copy()
    prefix_max = 0.0
    for _ in range(steps):
        prefix_max = max(prefix_max, float(errors_given_state(q, labels, w).max()))
        w = w @ m
    tv = .5 * float(np.abs(w - stationary).sum())
    tau = .5 * float(np.abs(m[:, None, :] - m[None, :, :]).sum(axis=2).max())
    residual_tv = .5 * float(np.abs(stationary @ m - stationary).sum())
    if tau >= 1 - 1e-12:
        raise RuntimeError("One-step contraction is insufficient for the all-future bound")
    stationary_uncertainty = residual_tv / (1 - tau)
    tail_bound = float(errors_given_state(q, labels, stationary).max()) + tv + 2 * stationary_uncertainty
    return {
        "prefix_windows_checked": steps,
        "maximum_prefix_error": prefix_max,
        "tv_after_prefix_to_computed_stationary": tv,
        "dobrushin_contraction_factor": tau,
        "stationary_tv_uncertainty_bound": stationary_uncertainty,
        "all_later_windows_error_upper_bound": tail_bound,
        "all_subsequent_windows_error_upper_bound": max(prefix_max, tail_bound),
        "qualification": "Numerical TV/contraction bound, not interval arithmetic; IID inputs and the stated first-order model",
    }


def periodic_stress(q, labels, pair_indices):
    """Periodic deterministic true inputs, stochastic *actual* feedback bits.

    This is a specified sequence stress test, not a worst-over-all-sequences
    computation. The 2-state bit distribution is propagated around the cycle.
    """
    kernels = []
    for phase, j in enumerate(pair_indices):
        i = pair_indices[phase - 1]
        kernels.append(np.column_stack([1 - q[i, j], q[i, j]]))
    cycle = np.eye(2)
    for kernel in kernels:
        cycle = cycle @ kernel
    w, iterations, residual = stationary_distribution(cycle)
    phases = []
    ce = conditional_errors(q, labels)
    for phase, (j, kernel) in enumerate(zip(pair_indices, kernels)):
        i = pair_indices[phase - 1]
        err = float(w @ ce[i, j])
        phases.append({"previous_pair_index": int(i), "current_pair_index": int(j),
                       "previous_decoded_bit_distribution": w.tolist(), "error": err})
        w = w @ kernel
    return {"phases": phases,
            "maximum_phase_error": max(p["error"] for p in phases),
            "mean_cycle_error": float(np.mean([p["error"] for p in phases])),
            "stationary_iterations": iterations, "stationary_residual_l1": residual}


def encode_policy(decoder):
    """Lossless run representation; each run includes both count-bin endpoints."""
    answer = []
    for row in decoder:
        runs = []
        start = 0
        for i in range(1, len(row) + 1):
            if i == len(row) or row[i] != row[start]:
                runs.append([start, i - 1, float(row[start])])
                start = i
        answer.append(runs)
    return answer


def evaluate_actual(p, labels, decoder, fresh_state, stress_pairs):
    q = output_probabilities(p, decoder)
    m = transition_matrix(q)
    w, iterations, residual = stationary_distribution(m)
    err = errors_given_state(q, labels, w)
    ce = conditional_errors(q, labels)
    i, j, z = np.unravel_index(np.argmax(ce), ce.shape)
    from_correct = startup_trace(q, labels, m, oracle_initial(labels))
    from_fresh = startup_trace(q, labels, m, fresh_state)
    from_random = startup_trace(q, labels, m, np.full(2 * len(labels), 1 / (2 * len(labels))))
    return {
        "stationary_worst_current_pair_error": float(err.max()),
        "stationary_mean_error": float(err.mean()),
        "stationary_per_current_pair_error": err.tolist(),
        "stationary_joint_previous_pair_decoded_bit": w.reshape(-1, 2).tolist(),
        "stationary_iterations": iterations,
        "stationary_residual_l1": residual,
        "transition_max_row_sum_error": float(np.max(np.abs(m.sum(axis=1) - 1))),
        "stationary_pair_marginal_max_error": float(np.max(np.abs(w.reshape(-1, 2).sum(1) - 1 / len(labels)))),
        "worst_fixed_previous_pair_and_decoded_bit": {
            "previous_pair_index": int(i), "current_pair_index": int(j),
            "previous_decoded_bit": int(z), "error": float(ce[i, j, z]),
            "stationary_probability_of_previous_state": float(w[2 * i + z]),
            "scope": "Conditional error of a specific state, not a guarantee that an adversary can force it",
        },
        "startup_from_correct_previous_answer": from_correct,
        "startup_from_actual_fresh_first_decision": from_fresh,
        "fresh_start_future_bound": startup_future_bound(q, labels, m, fresh_state, w),
        "startup_from_independent_fair_bit": from_random,
        "periodic_bright_dark_stress": periodic_stress(q, labels, stress_pairs),
    }


def evaluate_design(name, x, y, rho, source):
    labels = pair_labels(len(x))
    k = len(x)
    p, truncation = observation_tensor(x, y, rho)
    policies = train_policies(p, labels)
    fresh_state, fresh_errors = fresh_initial(x, y, truncation["cutoff"])
    stress = [0 * k + (k - 1), (k - 2) * k + 1]
    report = {"name": name, "rho": float(rho), "x": list(x), "y": list(y),
              "peak_expected_signal_count": float(max(x) + max(y)),
              "source": source, "source_precision": "Committed three-decimal encoder vectors; no recovery of lost optimizer precision",
              "count_representation": truncation,
              "fresh_first_window_per_pair_error": fresh_errors.tolist(),
              "fresh_first_window_worst_error": float(fresh_errors.max()),
              "policies": {}}
    for policy_name, (decoder, train_errors, lp_value) in policies.items():
        evaluated = evaluate_actual(p, labels, decoder, fresh_state, stress)
        report["policies"][policy_name] = {
            "training_worst_current_pair_error": float(train_errors.max()),
            "training_lp_objective": lp_value,
            "training_per_current_pair_error": train_errors.tolist(),
            "decoder_true_probability_runs": encode_policy(decoder),
            "evaluation": evaluated,
        }
    return report


def find_tested_rescaling(x, y, rho, target=.01):
    """Return the first passing member of a specified finite scale grid.

    Each point refits the correct-answer oracle policy and evaluates its transfer
    to actual feedback. No monotonicity or global optimum is asserted. The
    criterion includes a fresh first decision and all subsequent IID windows.
    """
    labels = pair_labels(len(x))
    trials = []
    for factor in [1.0001, 1.00025, 1.0005, 1.001, 1.0025, 1.005, 1.01, 1.025]:
        sx, sy = factor * np.asarray(x), factor * np.asarray(y)
        p, representation = observation_tensor(sx, sy, rho)
        decoder, _, _ = train_policies(p, labels)["correct_previous_answer"]
        q = output_probabilities(p, decoder)
        m = transition_matrix(q)
        stationary, _, _ = stationary_distribution(m)
        initial, fresh_errors = fresh_initial(sx, sy, representation["cutoff"])
        future = startup_future_bound(q, labels, m, initial, stationary)
        bound = max(float(fresh_errors.max()), future["all_subsequent_windows_error_upper_bound"])
        trial = {"factor": factor, "peak_expected_signal_count": float(sx.max() + sy.max()),
                 "stationary_worst_current_pair_error": float(errors_given_state(q, labels, stationary).max()),
                 "fresh_start_all_windows_error_upper_bound": bound}
        trials.append(trial)
        if bound <= target - 1e-8:
            return {"method": "First passing tested scale; not an optimized sequential photon budget",
                    "target": target, "acceptance_margin": 1e-8,
                    "tested_scales": trials, "passing_candidate": {
                        **trial, "x": sx.tolist(), "y": sy.tolist(),
                        "count_representation": representation,
                        "decoder_true_probability_runs": encode_policy(decoder),
                        "fresh_first_window_worst_error": float(fresh_errors.max()),
                        "fresh_start_future_bound": future,
                    }}
    return {"method": "Finite grid search, not an impossibility test", "target": target,
            "tested_scales": trials, "passing_candidate": None}


def read_finite_json_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if np.isfinite(row.get("P", np.inf)):
            rows.append(row)
    return rows


def history_counterexample():
    """Recheck the disclosed uniform-history model against one fixed history."""
    x = [168.38, 12.934, .002, 55.917]
    y = [56.377, .002, 12.909, 167.508]
    p, _ = observation_tensor(x, y, .01)
    labels = pair_labels(4)
    d, errors, _ = train_policies(p, labels)["no_feedback"]
    q = output_probabilities(p, d)
    ce = conditional_errors(q, labels)
    return {
        "source": "results/mc_isi_small.jsonl, banded rho=0.01",
        "x": x, "y": y, "peak_expected_signal_count": max(x) + max(y),
        "worst_current_pair_error_averaged_uniform_history": float(errors.max()),
        "previous_pair": [0, 3], "current_pair": [2, 1],
        "fixed_history_error": float(ce[3, 9, 0]),
        "interpretation": "This does not violate the uniform-random-history model; it rejects an arbitrary-history interpretation of its 1% bound.",
    }


def run_experiment():
    source = "results/mc_fb_ans.jsonl"
    rows = read_finite_json_rows(ROOT / source)
    # Finite rows in their committed order: 3 cases at .01, 3 at .03, 1 at .1.
    selected = [
        ("banded_rho_0.01", rows[0], .01),
        ("threshold_rho_0.01", rows[1], .01),
        ("banded_rho_0.03", rows[3], .03),
        ("threshold_rho_0.03", rows[4], .03),
        ("threshold_rho_0.10", rows[6], .10),
    ]
    cases = []
    for name, row, rho in selected:
        report = evaluate_design(name, row["x"], row["y"], rho, source)
        report["tested_rescaling_for_actual_feedback"] = find_tested_rescaling(row["x"], row["y"], rho)
        cases.append(report)
        no = report["policies"]["no_feedback"]
        oracle = report["policies"]["correct_previous_answer"]
        actual = oracle["evaluation"]["stationary_worst_current_pair_error"]
        print(f"{name:24s} P={report['peak_expected_signal_count']:.3f} "
              f"no feedback={100*no['training_worst_current_pair_error']:.6f}% "
              f"correct answer={100*oracle['training_worst_current_pair_error']:.6f}% "
              f"actual stationary={100*actual:.6f}%", flush=True)
    return {
        "schema_version": 1,
        "method": "Probability propagation and a finite-state Markov chain; no Monte Carlo",
        "model": {
            "input_distribution": "IID uniform over 16 pairs (A,B) in {0,1,2,3}^2",
            "count_mean": "x[A_t] + y[B_t] + rho * (x[A_(t-1)] + y[B_(t-1)])",
            "counts": "Conditionally independent Poisson draws given the true input sequence",
            "training": "Minimax error over current pairs, with previous true comparison supplied correctly",
            "actual_feedback": "Transfer that fitted policy unchanged; replace supplied correct answer with receiver's actual previous decoded bit",
            "worst_case_scope": "Worst current-pair error averaged over specified history distribution; not worst over arbitrary sequences",
            "limitations": ["No reaction kinetics or molecule trajectory correlations", "Only one previous window contributes spillover", "No encoder-shape or closed-loop policy optimization; a finite scale grid is evaluated", "Floating-point LP and probability arithmetic", "Results are conditional on selected stored rounded encoders", "Specified periodic stress sequence is not exhaustive adversarial analysis"],
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "pair_order": [[a, b] for a in range(4) for b in range(4)],
        "history_counterexample": history_counterexample(),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/feedback_sequence.json")
    args = parser.parse_args()
    report = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
