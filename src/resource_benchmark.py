"""Compare photons, elapsed time and resolved observations under one hardware budget.

All times are physical when ``peak_rate`` is in detected signal counts / second.
The defaults use dimensionless units: R=1, R*deadline=200, b/R in {0,.01}.
Sources and integration are paused during nonoverlapping readouts. An adaptive
receiver can stop both sources after a readout; gating/feedback latency and energy
must be included in readout_time and energy_per_readout. No hardware energy is
assumed. This is an ideal gated-source model, not a detector implementation.

Fixed threshold and peeled-band shapes are held fixed and their exposure and
decoder are recalibrated. OOK uses four serial, separately resolved bit slots.
The adaptive banded policy is optimized on a selected finite time/count grid.
Its objective is uniform-input mean wall time; every input pair has its own
error constraint. Independent per-hypothesis propagation verifies the policy.

Run: python3 src/resource_benchmark.py --out results/resource_benchmark.json
Fast smoke run: python3 src/resource_benchmark.py --quick --out /tmp/resources.json
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq, linprog
from scipy.sparse import coo_matrix
from scipy.special import logsumexp
from scipy.stats import poisson

from banded import build
from core import e2, labels, lp_minimax, pair_means, pmf_matrix

K = 4
LAB = labels(K)
PAIRS = [(a, b) for a in range(K) for b in range(K)]
TAIL = 1e-12
TRANSITION_CUTOFF = 1e-14
ERROR_BUFFER = 2e-6


def shape_rates(name, peak_rate=1.0):
    """One fixed encoding family, each transmitter obeying peak rate R."""
    if peak_rate <= 0:
        raise ValueError("peak_rate must be positive")
    if name == "threshold":
        x = np.arange(K - 1, -1, -1, dtype=float)
        y = x[::-1].copy()
    elif name == "banded":
        _, x, y, _, _ = build(K)
    elif name == "banded_bg1":
        # Previously searched nominal bg=1 candidate, first record of
        # results/bgdrift.jsonl at base commit08b0ef5. Refit its rates/exposure
        # here; this is not a new globally optimized background-rate design.
        x = np.array([145.264, 10.132, .002, 47.199])
        y = np.array([47.144, .002, 10.133, 145.294])
    else:
        raise ValueError(name)
    scale = peak_rate / max(x.max(), y.max())
    return x * scale, y * scale


def smallest_exposure(error_fn, target, initial=1.0):
    lo, hi = 0.0, float(initial)
    while error_fn(hi) > target:
        hi *= 2
        if hi > 1e12 * initial:
            raise RuntimeError("no feasible exposure found")
    for _ in range(45):
        mid = (lo + hi) / 2
        if error_fn(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def error_from_decoder(P, decoder, lab=LAB):
    p_true = P @ decoder
    return np.where(np.asarray(lab) == 1, 1 - p_true, p_true)


def _metrics(name, errors, active, checks, signal_photons, background_photons,
             active_deadline, max_checks, readout_time, wall_deadline):
    active, checks = np.asarray(active), np.asarray(checks)
    photons = np.asarray(signal_photons)
    wall = active + readout_time * checks
    latest_wall = active_deadline + max_checks * readout_time
    return {
        "name": name,
        "input_pairs": [list(p) for p in PAIRS],
        "error_by_pair": np.asarray(errors).tolist(),
        "worst_pair_error": float(np.max(errors)),
        "expected_active_time_by_pair": active.tolist(),
        "expected_readouts_by_pair": checks.tolist(),
        "expected_wall_time_by_pair": wall.tolist(),
        "expected_signal_photons_by_pair": photons.tolist(),
        "expected_background_photons_by_pair": np.asarray(background_photons).tolist(),
        "uniform_mean_active_time": float(active.mean()),
        "uniform_mean_readouts": float(checks.mean()),
        "uniform_mean_wall_time": float(wall.mean()),
        "uniform_mean_signal_photons": float(photons.mean()),
        "max_expected_signal_photons": float(photons.max()),
        "max_expected_wall_time": float(wall.max()),
        "active_time_deadline": float(active_deadline),
        "max_readouts": int(max_checks),
        "wall_time_deadline": float(latest_wall),
        "common_wall_deadline": float(wall_deadline),
        "meets_wall_deadline": bool(latest_wall <= wall_deadline + 1e-9 * max(1, wall_deadline)),
    }


def fixed_shared(name, peak_rate=1.0, background_rate=0.0, epsilon=.01,
                 readout_time=0.0, wall_deadline=200.0):
    if background_rate < 0 or readout_time < 0:
        raise ValueError("background_rate and readout_time must be nonnegative")
    x, y = shape_rates(name, peak_rate)
    signal = pair_means(x, y)
    rates = signal + background_rate
    horizon = smallest_exposure(lambda t: lp_minimax(rates * t, LAB),
                                epsilon - ERROR_BUFFER, 100 / peak_rate)
    _, decoder = lp_minimax(rates * horizon, LAB, return_decoder=True)
    errors = error_from_decoder(pmf_matrix(rates * horizon), decoder)
    out = _metrics("fixed_" + name, errors, np.full(16, horizon), np.ones(16),
                   signal * horizon, np.full(16, background_rate * horizon),
                   horizon, 1, readout_time, wall_deadline)
    out.update(x_rates=x.tolist(), y_rates=y.tolist(), decoder=decoder.tolist(),
               status="achievable numerical design; shape fixed, decoder and exposure fitted")
    return out


def comparison_errors_from_bit_errors(bit_error):
    """Exactly enumerate comparator errors after four independent bit decisions."""
    # A[z,w] is P(decoded number=w | actual number=z), for independent slots.
    A = np.ones((4, 4))
    for actual in range(4):
        for decoded in range(4):
            for bit in (0, 1):
                v, w = (actual >> bit) & 1, (decoded >> bit) & 1
                A[actual, decoded] *= (1 - bit_error[v]) if v == w else bit_error[v]
    errors = []
    for a, b in PAIRS:
        joint = np.outer(A[a], A[b])
        errors.append(float(joint[(np.arange(4)[:, None] > np.arange(4)[None, :]) != (a > b)].sum()))
    return np.asarray(errors)


def four_bit_ook(peak_rate=1.0, background_rate=0.0, epsilon=.01,
                 readout_time=0.0, wall_deadline=200.0):
    """Four serial slots with the full comparator error budget.

    Fit the largest symmetric bit error q whose 16 explicitly enumerated
    comparator errors are <= epsilon-ERROR_BUFFER, then fit the binary minimax
    count decoder at q. Re-evaluate all comparator errors using the actual
    binary LP errors. This improves on an epsilon/4 union-bound allocation, but
    is only the symmetric equal-duration bit-decoder family, not an unrestricted
    four-slot optimum. The randomized decoder may randomize at zero counts.
    """
    if peak_rate <= 0 or background_rate < 0 or readout_time < 0:
        raise ValueError("invalid physical parameters")
    bit_target = brentq(lambda q: comparison_errors_from_bit_errors([q, q]).max()
                       - (epsilon - ERROR_BUFFER), 0., .5, xtol=1e-14)
    slot = smallest_exposure(
        lambda t: e2(background_rate * t, (peak_rate + background_rate) * t),
        bit_target, 6 / peak_rate)
    means = np.array([background_rate, peak_rate + background_rate]) * slot
    _, decoder = lp_minimax(means, np.array([0, 1]), return_decoder=True)
    bit_error = error_from_decoder(pmf_matrix(means), decoder, [0, 1])
    errors = comparison_errors_from_bit_errors(bit_error)
    if errors.max() > epsilon:
        raise RuntimeError("four-slot comparator verification failed")
    ones = np.array([int(a).bit_count() + int(b).bit_count() for a, b in PAIRS])
    active = 4 * slot
    out = _metrics("four_serial_bit_slots", errors, np.full(16, active), np.full(16, 4),
                   ones * peak_rate * slot, np.full(16, background_rate * active),
                   active, 4, readout_time, wall_deadline)
    out.update(slot_duration=float(slot), bit_error_zero=float(bit_error[0]),
               bit_error_one=float(bit_error[1]), bit_error_union_bound=float(4 * max(bit_error)),
               symmetric_bit_error_target=float(bit_target),
               bit_decoder=decoder.tolist(),
               status="achievable four-slot design; equal-duration symmetric bit family fitted to all 16 comparator errors")
    return out


def _posterior(rates, t, nmax):
    logp = poisson.logpmf(np.arange(nmax)[None, :], rates[:, None] * t)
    norm = logsumexp(logp, axis=0)
    post = np.exp(logp - norm[None, :])
    return post, np.exp(norm - np.log(len(rates)))


def propagate_policy(rates, horizon, policy):
    """Independent forward calculation for each hypothesis, with no LP flows.

    The overflow policy continues to the horizon and then says FALSE. Overflow
    is conservatively charged as error under either label, full horizon and all
    readouts. This uses full convolution probabilities (no transition cutoff).
    """
    rates = np.asarray(rates, float)
    policy = np.asarray(policy, float)
    steps, nmax, actions = policy.shape
    if actions != 3 or np.max(np.abs(policy.sum(2) - 1)) > 1e-9:
        raise ValueError("policy must have normalized stop-false/stop-true/continue probabilities")
    if np.max(policy[-1, :, 2]) > 1e-12:
        raise ValueError("terminal policy must stop")
    dt = horizon / steps
    errors, active_times, checks, lost_mass = [], [], [], []
    for i, rate in enumerate(rates):
        increments = poisson.pmf(np.arange(nmax), rate * dt)
        alive = increments.copy()
        lost = max(0.0, 1 - alive.sum())
        stops = np.zeros(2)
        elapsed, reads = 0.0, 0.0
        for k in range(steps):
            action_mass = alive[:, None] * policy[k]
            by_action = action_mass.sum(0)
            stops += by_action[:2]
            ended = by_action[:2].sum()
            elapsed += (k + 1) * dt * ended
            reads += (k + 1) * ended
            if k + 1 < steps:
                nxt = np.convolve(action_mass[:, 2], increments)[:nmax]
                lost += max(0.0, by_action[2] - nxt.sum())
                alive = nxt
        lost = max(lost, max(0.0, 1 - stops.sum()))
        errors.append(float(stops[1 - LAB[i]] + lost))
        active_times.append(float(elapsed + horizon * lost))
        checks.append(float(reads + steps * lost))
        lost_mass.append(float(lost))
    return {"errors": np.array(errors), "active_times": np.array(active_times),
            "checks": np.array(checks), "lost_mass": np.array(lost_mass)}


def adaptive_banded(peak_rate=1.0, background_rate=0.0, epsilon=.01,
                    readout_time=0.0, wall_deadline=200.0, steps=8,
                    minimum_active_time=None, shape_name="banded"):
    """Optimize one discrete stopping grid and independently validate its policy.

    Reference probability is the uniform mixture of all 16 hypotheses. At
    (t,n), the hypothesis/reference likelihood ratio is bounded by 16, avoiding
    the ill conditioning of a single reference Poisson rate. For Poisson rates,
    count and active time are sufficient even after conditioning on survival
    under a common stopping policy. Reference occupancy flows are consequently
    linear and every hypothesis error is a linear weighted stop mass.
    """
    if steps < 1 or readout_time < 0 or wall_deadline <= 0:
        raise ValueError("invalid time grid")
    horizon = wall_deadline - steps * readout_time
    if horizon <= 0 or (minimum_active_time is not None and horizon < minimum_active_time):
        return {"name": "adaptive_" + shape_name, "steps": steps, "feasible": False,
                "reason": "insufficient active time after all readout pauses"}
    x, y = shape_rates(shape_name, peak_rate)
    signal = pair_means(x, y)
    rates = signal + background_rate
    if lp_minimax(rates * horizon, LAB) > epsilon - ERROR_BUFFER:
        return {"name": "adaptive_" + shape_name, "steps": steps, "feasible": False,
                "reason": "fixed endpoint cannot meet the numerical error margin"}
    dt = horizon / steps
    nmax = int(poisson.isf(TAIL, rates.max() * horizon)) + 2
    ninc = min(nmax, int(poisson.isf(TAIL, rates.max() * dt)) + 2)
    increments = poisson.pmf(np.arange(ninc)[None, :], rates[:, None] * dt)
    posteriors, mixtures = zip(*[_posterior(rates, (k + 1) * dt, nmax) for k in range(steps)])
    nodes, nv = steps * nmax, steps * nmax * 3
    node = np.arange(nodes)
    rows = np.repeat(node, 3).tolist()
    cols = np.arange(nv).tolist()
    vals = np.ones(nv).tolist()
    for k in range(steps - 1):
        post = posteriors[k]
        for inc in range(ninc):
            origins = np.arange(nmax - inc)
            pr = increments[:, inc] @ post[:, :nmax - inc]
            keep = pr >= TRANSITION_CUTOFF
            origins, pr = origins[keep], pr[keep]
            rows.extend(((k + 1) * nmax + origins + inc).tolist())
            cols.extend(((k * nmax + origins) * 3 + 2).tolist())
            vals.extend((-pr).tolist())
    Aeq = coo_matrix((vals, (rows, cols)), shape=(nodes, nv)).tocsr()
    beq = np.zeros(nodes)
    beq[:nmax] = mixtures[0]
    # Ratios P_i(N=n,t)/P_mix(N=n,t) = 16*posterior_i.
    Aerr = np.zeros((16, nv))
    for k, post in enumerate(posteriors):
        for i in range(16):
            action = 1 - LAB[i]
            Aerr[i, (k * nmax + np.arange(nmax)) * 3 + action] = 16 * post[i]
    objective = np.zeros(nv)
    for k in range(steps):
        for action in (0, 1):
            objective[(k * nmax + np.arange(nmax)) * 3 + action] = (k + 1) * (dt + readout_time)
    bounds = [(0, None)] * nv
    for n in range(nmax):
        bounds[((steps - 1) * nmax + n) * 3 + 2] = (0, 0)
    start = time.monotonic()
    result = linprog(objective, A_ub=Aerr, b_ub=np.full(16, epsilon - ERROR_BUFFER / 2),
                     A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs",
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    if not result.success:
        return {"name": "adaptive_" + shape_name, "steps": steps, "feasible": False,
                "reason": "LP: " + result.message}
    flow = np.maximum(result.x.reshape(steps, nmax, 3), 0)
    mass = flow.sum(2)
    policy = np.zeros_like(flow)
    policy[:, :, 2] = 1
    _, final_decoder = lp_minimax(rates * horizon, LAB, return_decoder=True, nmax=nmax)
    policy[-1, :, 0] = 1 - final_decoder[:nmax]
    policy[-1, :, 1] = final_decoder[:nmax]
    policy[-1, :, 2] = 0
    used = mass > 1e-13
    policy[used] = flow[used] / mass[used, None]
    policy[-1, :, 2] = 0
    policy[-1] /= policy[-1].sum(1)[:, None]
    check = propagate_policy(rates, horizon, policy)
    if check["errors"].max() > epsilon:
        raise RuntimeError("independent policy verification failed: " + str(check["errors"].max()))
    out = _metrics("adaptive_" + shape_name, check["errors"], check["active_times"], check["checks"],
                   signal * check["active_times"], background_rate * check["active_times"],
                   horizon, steps, readout_time, wall_deadline)
    out.update(feasible=True, steps=steps, check_interval_active=dt, x_rates=x.tolist(), y_rates=y.tolist(),
               count_cap=nmax - 1, tail_error_charged_by_pair=check["lost_mass"].tolist(),
               overflow_policy="continue to the horizon and return FALSE; all overflow mass conservatively charged as error",
               transition_cutoff=TRANSITION_CUTOFF,
               lp_mean_wall_time=float(result.fun), solve_seconds=time.monotonic() - start,
               policy_action_order=["stop_false", "stop_true", "continue"],
               policy=policy.tolist(),
               status="numerical stopping-grid optimum, independently verified achievable policy; fixed shape")
    return out


def mean_energy(design, energy_per_signal_photon=1.0, active_power=0.0, energy_per_readout=0.0):
    """Parameterized electrical accounting; coefficients must be measured/supplied.

    energy_per_signal_photon is effective electrical energy per EXPECTED DETECTED
    signal photon, including source and link inefficiency. Readout energy includes
    the entire pause, gate/control, reset and decision cost. No default is a claim
    about real hardware; the default simply returns a photon-count objective.
    """
    return (energy_per_signal_photon * design["uniform_mean_signal_photons"]
            + active_power * design["uniform_mean_active_time"]
            + energy_per_readout * design["uniform_mean_readouts"])


def readout_energy_break_even(design, reference):
    """E_read ceiling for design to beat reference, when design uses more reads.

    Ceiling = photon_coefficient*c_photon + active_time_coefficient*P_active.
    This uses uniform-input MEAN quantities consistently, not a peak/mean mix.
    """
    extra = design["uniform_mean_readouts"] - reference["uniform_mean_readouts"]
    if extra <= 0:
        return None
    return {
        "extra_mean_readouts": extra,
        "photon_coefficient": (reference["uniform_mean_signal_photons"] - design["uniform_mean_signal_photons"]) / extra,
        "active_time_coefficient": (reference["uniform_mean_active_time"] - design["uniform_mean_active_time"]) / extra,
        "formula": "E_read < photon_coefficient*c_photon + active_time_coefficient*P_active",
    }


def benchmark(quick=False):
    epsilon, peak, deadline = .01, 1.0, 200.0
    backgrounds = [0.0] if quick else [0.0, .01]
    overheads = [0.0] if quick else [0.0, 10.0, 25.0, 50.0]
    grids = [2] if quick else [2, 4, 8]
    scenarios = []
    for bg in backgrounds:
        shape_names = ["threshold", "banded"] + (["banded_bg1"] if bg > 0 else [])
        fixed_base = [fixed_shared(s, peak, bg, epsilon, 0, deadline) for s in shape_names]
        ook_base = four_bit_ook(peak, bg, epsilon, 0, deadline)
        for overhead in overheads:
            fixed = []
            for base in fixed_base + [ook_base]:
                copy = dict(base)
                active, reads = np.array(base["expected_active_time_by_pair"]), np.array(base["expected_readouts_by_pair"])
                wall = active + overhead * reads
                latest = base["active_time_deadline"] + overhead * base["max_readouts"]
                copy.update(expected_wall_time_by_pair=wall.tolist(), uniform_mean_wall_time=float(wall.mean()),
                            max_expected_wall_time=float(wall.max()), wall_time_deadline=latest,
                            meets_wall_deadline=bool(latest <= deadline + 1e-9))
                fixed.append(copy)
            adaptive = []
            for shape_index in range(1, len(shape_names)):
                for m in grids:
                    shape_name = shape_names[shape_index]
                    print(f"background={bg:g}, readout={overhead:g}, {shape_name} adaptive grid={m}", flush=True)
                    adaptive.append(adaptive_banded(peak, bg, epsilon, overhead, deadline, m,
                                                   minimum_active_time=fixed_base[shape_index]["active_time_deadline"],
                                                   shape_name=shape_name))
            feasible = [d for d in fixed if d["meets_wall_deadline"]] + [d for d in adaptive if d.get("feasible")]
            best = min(feasible, key=lambda d: d["uniform_mean_wall_time"]) if feasible else None
            scenario = {"peak_signal_rate_per_transmitter": peak, "background_rate": bg,
                        "readout_pause": overhead, "common_wall_deadline": deadline, "epsilon": epsilon,
                        "fixed": fixed, "adaptive": adaptive,
                        "fastest_tested_feasible": None if best is None else {
                            "name": best["name"], "steps": best.get("steps"),
                            "uniform_mean_wall_time": best["uniform_mean_wall_time"]},
                        "ook_readout_energy_break_even_vs_fixed_band": readout_energy_break_even(fixed[-1], fixed[1])}
            scenarios.append(scenario)
    return {
        "schema_version": 1,
        "base_commit": "08b0ef546e9492177b93a61b0ff2b09539a4be1d",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "assumptions": [
            "Independent, constant-rate Poisson arrivals during active observation; perfect counter and fully off sources.",
            "Each transmitter obeys the same peak detected signal rate R; background is a rate b, not fixed counts per window.",
            "Sources and counting pause during every nonoverlapping readout; ideal coordinated gating is assumed.",
            "All schemes share the same wall-clock deadline; infeasible rows are excluded from the winner comparison.",
            "Adaptive objective is uniform-input mean wall time; all 16 conditional errors are separately constrained and checked.",
            "Adaptive grids and encoder shapes are a limited menu; the fastest tested scheme is not a global architecture optimum.",
            "Background scenarios include the previously searched bg=1 shape from results/bgdrift.jsonl first record, normalized to the rate cap and refitted; it is not globally optimized for this rate model.",
            "Four OOK slots run serially with one counter; equal-duration symmetric bit decoders are fitted to all 16 comparison-error constraints.",
            "Readout latency includes gating/reset/decision delays; electrical energy remains parameterized, not measured.",
            "Numerical truncation loss is charged as error in independent propagation; LP tolerances prevent exact-rational certification."
        ],
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results/resource_benchmark.json")
    args = parser.parse_args()
    result = benchmark(args.quick)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for s in result["scenarios"]:
        print("SCENARIO", s["background_rate"], s["readout_pause"], s["fastest_tested_feasible"])
        for d in s["fixed"] + [a for a in s["adaptive"] if a.get("feasible")]:
            print(" ", d["name"], d.get("steps", ""), "error", round(d["worst_pair_error"], 8),
                  "photons max/mean", round(d["max_expected_signal_photons"], 3), round(d["uniform_mean_signal_photons"], 3),
                  "wall mean/max", round(d["uniform_mean_wall_time"], 3), round(d["wall_time_deadline"], 3),
                  "feasible", d["meets_wall_deadline"])
    print("Wrote", path)


if __name__ == "__main__":
    main()
