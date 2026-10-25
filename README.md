# Computing A > B with fewer photons

Two separate light sources can let one photon counter decide whether A > B using 39% fewer detected photons than the best possible threshold scheme. The trick is to reorder each source's brightness levels so the pairs fall into alternating bands, which moves the hard decisions into dim light where photon noise is small. The threshold baseline is proved optimal, and the new design is certified within 0.55% of the best any scheme can do. This holds in an idealized model (perfect detector, no stray light, sources that switch fully off); stray light shrinks the saving to about 10% at one stray count per window.

![The 16 input pairs for each design on a square root photon scale, with the decoder's answer shaded and the close calls marked](figures/fig_bands.png)

Run `./reproduce.sh quick` to check 33 headline numbers in about two minutes.

## The problem

Two transmitters hold integers A and B in {0..K-1}. Each one sends out a nonnegative light intensity, x[A] and y[B], without seeing the other. A single photon counter sees N, a Poisson count with mean x[A] + y[B] + background. The receiver may use any randomized function of N to decide whether A > B (ties count as false). The worst case error over all K squared pairs must be at most 1%. The cost is P, the largest expected number of detected photons over all pairs, which is max(x) + max(y). The question is how small P can be, and what the encoders look like. The headline case is K = 4, where a threshold style design needs about 321.5 photons and a "peeled band" design with an alternating decoder needs about 196.6, against a certified lower bound of 195.53.

The original 201 photon candidate came from a ChatGPT conversation and was independently verified here (src/verify_candidate.py). Released under the MIT License (see LICENSE).

## Results

Status words: "verified" means recomputed here with exact style arithmetic (mpmath, 50 digits). "Proved" means a short argument plus a numerical evaluation of a one dimensional function. "Certified" means a lower bound from a solver dual bound, valid up to floating point in the h table and the MILP tolerances (not an exact rational certificate). "Numerical upper bound" means a design found by search whose error was checked by LP; it shows P is achievable, not that it is optimal.

| Result (K = 4 unless stated) | Value | Status |
| --- | --- | --- |
| 201 photon candidate, x=[100.5,4.7,0,29.4], y reversed, decoder "N==0 or 18<=N<=77": worst error | 0.959542% at (A,B)=(2,0), max mean 201 | verified (scipy and mpmath) |
| Same encoding with the best randomized (LP) decoder: worst error | 0.9178% | numerical |
| Same shape rescaled to exactly 1% with the LP decoder | 196.70 (196.69 gives 1.00007%, 196.70 gives 0.99991%) | exact for the shape |
| Peeled band construction, exact LP budget, K = 3,4,5,6,7,8 | 57.56, 196.62, 534.62, 1295.39, 2936.78, 6389.06 | numerical upper bound |
| Peeled band, pairwise budget 2 v_(K-1), K = 4 | 196.33 | arithmetic from h (not an achievable budget by itself) |
| Lower bound for ANY encoder and decoder, K = 3 | 57.43 | certified |
| Lower bound for ANY encoder and decoder, K = 4 | 195.53 | certified |
| Best single threshold decoder over all encoders, "low counts mean true" | 321.540 (corner u = 1.5, s = 107.18) | proved, lower bound certified to 321.5366 |
| Best single threshold decoder, "high counts mean true" | 452.319 | proved |
| Threshold optimum, K = 3 to 10 (low true) | 126.05, 321.54, 603.09, 971.21, 1425.53, 1966.96, 2594.72, 3308.93 | proved |
| Lower bound over monotone encoders (x nonincreasing, y nondecreasing), any decoder | 320.32 | certified |
| Lower bound for the average photon cost, any encoder and decoder | 65.50 | certified |
| Best average cost design found | 65.8 | numerical upper bound |
| Gaussian control (r = x[A] + y[B] + N(0,1)), exact MILP optimum for K = 3,4,5 | 4d, 6d, 8d with d = 4.6527 (the threshold design) | solver optimal |
| Gaussian control, peeled band chain K = 4 | 14d | arithmetic |
| Gain drift, shared gain within +-5%, best banded shape | 245.4 (monotone shapes: 598.2) | numerical upper bound |
| Error target 0.1%, 0.3%, 1%, 3%, 10%: banded vs threshold optimum | 354.7 vs 567.0, 262.3 vs 448.4, 196.5 vs 321.5, 137.0 vs 210.2, 74.8 vs 97.9 | numerical (banded); corner formula on a grid (threshold) |
| Drift plus background, banded vs monotone (bg 0.1 / 1 / 3 counts) | +-2%: 251.8 vs 405.1, 318.7 vs 406.5; +-5%: 288.8 vs 598.4, 366.7 vs 600.5, 435.4 vs 605.8 | numerical upper bound |
| Calibration every 10,000 decisions, +-5% drift (src/calibration.py) | saving 41.8% with no background, 15.1% at 1 count | model on interpolated numerical results |

The remaining drift, background and error target runs are in results/ (raw JSON lines). results/bgdrift.jsonl designs can be rechecked with src/check_bgdrift.py. That check finds that a few designs are at up to 1.0009% error when the decoder fitted on a 41 point gain grid is tested at 401 gains, so those drift numbers are good to roughly a tenth of a percent in error, not exact. The values in the table above are the rescaled ones: the affected banded designs were scaled up 0.05% until the dense check passed at 1%.

Where the first round and this repo differ: the first round reported 196.75 for the 201 photon shape after LP rescaling. The exact value is 196.70 (see the table); 196.75 came from a looser search and also passes. core.py was not changed to force a match.

## Privacy: what the count leaks

The banded design does not leak less than the threshold design; it leaks more about each input. With inputs uniform and each design at its 1% budget (src/leakage.py):

| Design | Leak beyond the answer, I(N; A,B \| answer) | Leak about A alone | Best guess of A from the count |
| --- | --- | --- | --- |
| Threshold, 321.5 photons | 1.574 bits | 0.332 bits | 43.8% (same as from the answer alone) |
| Banded, 196.6 photons | 1.575 bits | 0.620 bits | 62.2% |

The threshold count depends only on A minus B; the banded count gives pairs their own levels (zero photons means A = 2, B = 1). Exact computation, checked by a 400,000 trial simulation.

## Molecular communication

The same question for receivers that count molecules (src/mc.py). Cost is the worst case expected molecules arriving; every design meets 1% worst case error; all values are best found by search (2 or 3 starts) and rechecked, not proved, except the threshold row with no spillover.

| Spillover from the previous message | One type, threshold | One type, banded | Two types that cancel 1:1 | Two types counted separately |
| --- | --- | --- | --- | --- |
| 0 | 321.5 | 196.6 | 201.8 | 106.4 |
| 1% | 327.6 | 335.9 | not run | not run |
| 3% | 350.8 | none found | not run | not run |
| 10% | 720.8 | none found | 817.8 | 285.8 |

- Cancelling molecule types (as in Long et al. 2025, arXiv 2502.20484) lose information: counting both types always sees at least as much as their difference.
- Spillover is modeled as a fraction of a uniformly random, unknown previous message. "None found" means no searched design reached 1% below 20,000 molecules.
- With decision feedback (the receiver also sees its previous answer, or, as a best case, the exact previous pair), the banded design wins only at 1% spillover: 319.3 vs 326.2 with the previous answer, 287.4 vs 325.5 with the exact previous pair. From 3% on, the threshold design wins in every case (results/mc_fb*.jsonl).
- Not modeled: reaction kinetics, nonlinear testbed responses, and spillover from more than one previous message.

## Quick start

    python3 -m pip install -r requirements.txt
    ./reproduce.sh quick        # about 2 minutes, 33 checks; prints each headline number next to its expected value, PASS or FAIL
    ./reproduce.sh full         # lists the long jobs with their exact commands
    ./reproduce.sh full --run   # runs them one after another (hours); queue scripts write to results/rerun/

Run everything from the repo root. The scripts use nice -n 15 in reproduce.sh; set NICE="" to turn that off. Python 3.13, numpy 2.5.3, scipy 1.18.1 (HiGHS), highspy 1.15.1, mpmath 1.4.1, matplotlib 3.11.2.

## What each script does

| File | What it does |
| --- | --- |
| src/core.py | Poisson LP decoder (worst case error), two point error e2, h(mu), budget for a shape. The error target is the EPS environment variable, default 0.01. |
| src/verify_candidate.py | Checks the 201 photon candidate: 16 pair errors with scipy and with mpmath at 50 digits, the LP decoder on the same encoding, and the rescaled budget. Seconds. |
| src/threshold_optimum.py | Best single randomized threshold decoder for any encoder: minimizes s*(u) g(u) over u, certifies it by adaptive bisection, cross checks with the LP. Argument "table" gives K = 3 to 10. About 10 s for K = 4, about 1 minute for the table. |
| src/banded.py | The peeled band construction for K = 2 to 8, checks the band rule for every pair, prints pairwise and exact LP budgets. Seconds. |
| src/gaussian_milp.py | Gaussian control solved as an exact MILP, plus the band chain cost. About 1 minute with K = 5. |
| src/make_hgrid.py | Regenerates data/hgrid.npy and data/hgrid_ext.npy from core.h and compares them with the shipped files. About 20 s. |
| src/lb_pairwise2.py | Certified lower bound through the pairwise relaxation as a MILP (needs h on a grid; HGRID, MONO, OBJ and USE_HIGHS environment variables select the variants). K = 3 takes about 1 minute, K = 4 takes tens of minutes per stage. |
| src/opt.py | Re-optimizes shapes under gain drift, background and average cost, with a robust LP decoder (Nelder Mead; environment variables BG and EXTRA). |
| src/check_bgdrift.py | Refits and retests the drift plus background designs. |
| src/leakage.py | Information the photon count leaks beyond the answer, per design. Seconds. |
| src/mc.py | Molecular receivers (one shared type, cancelling types, separately counted types), spillover (ISI) and decision feedback (FEEDBACK), with shape search. Environment variables ISI, FEEDBACK, EXTRA, NSTART. Minutes to an hour per run. |
| src/mc_check.py | Rechecks the stored best molecular and feedback designs. Seconds (quick) or minutes (all). |
| src/calibration.py | Cost of calibrating the gain with pilot pulses versus living with drift. Seconds. |
| src/figure_bands.py | Draws figures/fig_bands.png and .svg. |
| queues/*.sh | The exact command lists of the original long runs (drift, background, error targets, calibration). |
| data/ | h tables, row 0 is mu, row 1 is h(mu). |
| results/ | Raw JSON lines of every original run. |

Long job commands (also printed by ./reproduce.sh full):

    python3 src/lb_pairwise2.py 4 200 1500 8                                              # K=4 bound 195.53 (the 195.53 stage takes about 23 minutes, whole run is longer)
    USE_HIGHS=1 MONO=1 HGRID=data/hgrid_ext.npy python3 src/lb_pairwise2.py 4 330 1500 3  # monotone: 320.32
    USE_HIGHS=1 OBJ=avg HGRID=data/hgrid_ext.npy python3 src/lb_pairwise2.py 4 400 2400 2 # average: 65.50
    python3 src/opt.py free common 0.05 max 11 1 400                                      # banded under 5% drift: 245.4
    python3 src/opt.py monotone common 0.05 max 12 1 400                                  # monotone: 598.2
    python3 src/opt.py free nominal 0 avg 7 4 500                                         # average: 65.8

## How the lower bound works

Any scheme must separate every true pair mean m_t from every false pair mean m_f: restricting the decoder to those two hypotheses gives a two point test, so max(m_t, m_f) >= h(min(m_t, m_f)), where h(mu) is the smallest nu with two point minimax error 1% between Poisson(mu) and Poisson(nu). The MILP chooses, for each true and false pair, which one is higher, and uses a convex hull underestimator of the h staircase on each segment (h is increasing, so h on a table cell is at least its left value). The solver's dual bound is then a lower bound on P for every encoder and decoder.

## Limitations

- Simulation and numerics only. No hardware, no real detector, no dead time, afterpulsing or finite bandwidth.
- Floating point throughout. The "certified" numbers are solver dual bounds over a tabulated h (brentq at 1e-10, a 1e-6 margin and MILP tolerances), not exact rational certificates.
- The pairwise relaxation is a necessary condition. The gap between the 195.53 lower bound and the 196.62 design for K = 4 is not closed.
- Designs from opt.py come from Nelder Mead on a small number of starts. They are upper bounds only.
- The novelty search (is this construction or bound already known) is incomplete.
- The first round scripts that produced some numbers were lost and rewritten here. Every rewritten script reproduces the published numbers; the one difference (196.70 versus a looser 196.75) is explained above.
- The calibration model mixes a worst case peak photon cost with an averaged pilot cost, and assumes the gain is constant between calibrations. It is a modeling choice, stated so it can be argued with.
