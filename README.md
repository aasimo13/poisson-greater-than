# A near-optimal four-level comparison from one Poisson count

For A and B in {0,1,2,3}, two independently encoded light sources and one summed photon count can decide whether A > B at at most 1% error for every input pair using about **38.85% fewer expected signal photons than the optimal single-threshold scheme**. Reordering each source's brightness levels puts the input pairs into alternating decision bands and exploits the dependence of Poisson noise on the mean. A 196.62-photon construction and a recorded numerical lower bound of 195.53 bracket the optimum within about **0.6%** in this one-count model. The lower bound uses floating-point tables and solver tolerances; it is not an exact rational certificate.

The headline case assumes an ideal detector, no background light, and sources that switch fully off. Background light reduces the advantage. Photon savings alone do not establish a latency or electrical-energy saving: separately resolved observations are another resource, analyzed below. The construction's advantage also depends on alphabet size; this peeled-band family loses to the threshold benchmark by K = 6.

![The 16 input pairs for each design on a square root photon scale, with the decoder's answer shaded and the close calls marked](figures/fig_bands.png)

Run `./reproduce.sh quick` to check 33 headline numbers in about two minutes.

## The problem

Two transmitters hold integers A and B in {0..K-1}. Each independently maps its input to a deterministic nonnegative expected signal count, x[A] or y[B], integrated over one observation window. A single photon counter sees N, a Poisson count with mean x[A] + y[B] + background. The receiver may use any randomized function of N to decide whether A > B (ties count as false). The worst case error over all K squared pairs must be at most 1%. The cost P = max(x) + max(y) is the largest **expected detected signal-photon count** over input pairs; background photons are excluded from this signal budget, and P is not a hard bound on a realized count. The headline case is K = 4, where the optimal threshold scheme needs about 321.54 photons and a "peeled band" design needs about 196.62, against a recorded numerical lower bound of 195.53.

The original 201 photon candidate came from a ChatGPT conversation and was independently verified here (src/verify_candidate.py). Released under the MIT License (see LICENSE).

The [theory note](docs/theory.md) gives the threshold argument, pairwise lower-bound justification, and an analytic Gaussian optimum for every K. The [related-work note](docs/related_work.md) distinguishes this specific Poisson comparison from established function-oriented communication. Exact novelty remains unresolved.

## Continuation: photons, time, and actual feedback

The [research note](docs/research_note.md) connects the original result to two new reproducible experiments. Both retain explicit error constraints and distinguish numerical evidence from hardware measurements.

### A common clock changes the comparison

Each transmitter has peak detected signal rate R, every readout consumes a pause tau, and every design must finish by the same wall-clock deadline D. At zero background, with R*D = 200, the uniform-input mean times are:

| Readout pause R*tau | Fixed threshold | Fixed banded | Four bit slots | Adaptive banded, fastest tested grid |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160.780 | 98.314 | **22.776** | 44.504 |
| 10 | 170.780 | 108.314 | **62.776** | 65.429 |
| 25 | 185.780 | 123.314 | 122.776 | **105.975** |
| 50 | misses deadline | 148.314 | misses deadline | **124.853** |

Times are multiplied by R. For illustration only, R = 1,000,000 detected signal counts/second makes the displayed numbers microseconds, with a 200-microsecond deadline. All accepted policies have at most 1% error for every input pair. The adaptive mean-time objective uses uniform inputs; hard cases can take the full deadline. Its mean advantage does not establish an advantage in worst-case latency.

![Mean decision time against readout cost under two background rates](figures/fig_resources.png)

The four-slot result uses the full comparator error allowance and needs 22.7756 peak expected signal photons. The fixed banded/four-slot timing crossover is approximately R*tau = 25.18 for these two designs. At background rate b/R = 0.01, a stored background-retuned banded shape needs 323.603 peak expected signal photons versus 325.372 for the fixed threshold shape; the unretuned shape is much worse. These are selected encoding families, not globally optimal architectures. Sources and integration pause during readouts, and coordinated stopping is idealized with its cost charged to the readout parameters. Electrical energy is parameterized, not measured. See [methods and complete scope](docs/resource_methods.md) and [raw policies and results](results/resource_benchmark.json).

### Feedback with the receiver's own answers

A 32-state Markov calculation replaces the original correct-previous-answer side information with the receiver's actual previous decoded bit. At 1% spillover, the stored banded design changes from 0.999974% oracle error to 1.002228% stationary actual-feedback error. A tested 0.25% signal increase restores a fresh-start/all-subsequent-window numerical bound below 1% under IID uniform inputs. The repaired banded and threshold budgets are 320.134 and 326.205, respectively; these are tested feasible candidates, not optimized sequential budgets.

History remains consequential: an alternating bright/dark input sequence produces 9.226% maximum phase error for the original banded feedback design at 1% spillover. This does not contradict an IID-history-averaged guarantee. See [feedback methods](docs/feedback_methods.md) and [raw policies and results](results/feedback_sequence.json).

## Results

Status words: "verified" means numerically recomputed, including mpmath at 50 decimal digits for the original candidate. "Proved" refers to an analytic argument or reduction; numerical evaluations retain their tolerances. "Certified" means a lower bound from a solver dual bound, valid up to floating point in the h table and MILP tolerances, not an exact rational certificate. "Numerical upper bound" means a design whose error was checked by LP; it shows an achievable budget to the stated numerical accuracy, not global optimality.

| Result (K = 4 unless stated) | Value | Status |
| --- | --- | --- |
| 201 photon candidate, x=[100.5,4.7,0,29.4], y reversed, decoder "N==0 or 18<=N<=77": worst error | 0.959542% at (A,B)=(2,0), max mean 201 | verified (scipy and mpmath) |
| Same encoding with the best randomized (LP) decoder: worst error | 0.9178% | numerical |
| Same shape rescaled to 1% with the LP decoder | numerical boundary about 196.6943; 196.70 is a feasible rounded-up budget (196.69 gives 1.00007%, 196.70 gives 0.99991%) | numerical boundary for this fixed shape |
| Peeled band construction, numerical LP budget, K = 3,4,5,6,7,8 | 57.56, 196.62, 534.62, 1295.39, 2936.78, 6389.06 | numerical upper bound |
| Peeled band, pairwise budget 2 v_(K-1), K = 4 | 196.33 | arithmetic from h (not an achievable budget by itself) |
| Lower bound for ANY encoder and decoder, K = 3 | 57.43 | certified |
| Lower bound for ANY encoder and decoder, K = 4 | 195.53 | certified |
| Best single threshold decoder over all encoders, "low counts mean true" | 321.540 (corner u = 1.5, s = 107.18) | proved, lower bound certified to 321.5366 |
| Best single threshold decoder, "high counts mean true" | 452.319 | proved |
| Threshold search, K = 3 to 10 (low true) | 126.05, 321.54, 603.09, 971.21, 1425.53, 1966.96, 2594.72, 3308.93 | analytic reduction; grid/local search, adaptive numerical certificate run for K = 4 |
| Lower bound over monotone encoders (x nonincreasing, y nondecreasing), any decoder | 320.32 | certified |
| Lower bound for the average photon cost, any encoder and decoder | 65.50 | certified |
| Best average cost design found | 65.8 | numerical upper bound |
| Gaussian control (r = x[A] + y[B] + N(0,1)), all K >= 2 | 2(K-1)d with d = 2 Phi^-1(0.99) = 4.6527; attained by the threshold design | analytic proof in docs/theory.md; small-K MILP cross-checks |
| Gaussian control, peeled band chain K = 4 | 14d | arithmetic |
| Gain drift, shared gain within +-5%, best banded shape | 245.4 (monotone shapes: 598.2) | numerical upper bound |
| Error target 0.1%, 0.3%, 1%, 3%, 10%: banded vs threshold optimum | 354.7 vs 567.0, 262.3 vs 448.4, 196.5 vs 321.5, 137.0 vs 210.2, 74.8 vs 97.9 | numerical (banded); corner formula on a grid (threshold) |
| Drift plus background, banded vs monotone (bg 0.1 / 1 / 3 counts) | +-2%: 251.8 vs 405.1, 318.7 vs 406.5; +-5%: 288.8 vs 598.4, 366.7 vs 600.5, 435.4 vs 605.8 | numerical upper bound |
| Calibration every 10,000 decisions, +-5% drift (src/calibration.py) | saving 41.8% with no background, 15.1% at 1 count | model on interpolated numerical results |

The remaining drift, background and error target runs are in results/ (raw JSON lines). results/bgdrift.jsonl designs can be rechecked with src/check_bgdrift.py. That check finds that a few designs are at up to 1.0009% error when the decoder fitted on a 41 point gain grid is tested at 401 gains, so those drift numbers are good to roughly a tenth of a percent in error, not exact. The values in the table above are the rescaled ones: the affected banded designs were scaled up 0.05% until the dense check passed at 1%.

Where the first round and this repo differ: the first round reported 196.75 for the 201 photon shape after LP rescaling. The numerical boundary is about 196.6943, and 196.70 is a feasible budget rounded up to two decimals; 196.75 came from a looser search and also passes. core.py was not changed to force a match.

## Privacy: what the count leaks

The banded design leaks more about each individual input conditional on the true comparison answer. Joint information beyond that answer is nearly unchanged, while total information about the pair is slightly lower. The photon saving therefore does not imply that the observation retains less task-irrelevant information. With independent uniform inputs and each design at its numerical 1% budget (src/leakage.py):

| Design | Leak beyond the answer, I(N; A,B \| answer) | Leak about A beyond the answer, I(N; A \| answer) | Best guess of A from the count |
| --- | --- | --- | --- |
| Threshold, 321.5 photons | 1.574 bits | 0.332 bits | 43.8% (same as from the answer alone) |
| Banded, 196.6 photons | 1.575 bits | 0.620 bits | 62.2% |

The threshold mean depends only on A minus B. The banded construction has more distinct mean levels, although some pairs still share a level. (A,B) = (2,1) is the unique zero-mean pair; observing zero photons does not identify it with certainty, because positive Poisson means can also produce a zero count. The table is computed directly from the count distributions. The originally reported 400,000-trial cross-check is not part of the committed reproduction suite.

## Molecular communication

The same question for receivers that count molecules (src/mc.py). Cost is max(x) + max(y), the worst expected current-message signal arrivals, excluding spillover. The error constraint is at most 1% for each current input pair, **averaged over an independent uniform previous input pair**. It is not a guarantee for every input history. Values are best found by search (2 or 3 starts) and rechecked, not proved, except the threshold row with no spillover.

| Spillover from the previous message | One type, threshold | One type, banded | Two types that cancel 1:1 | Two types counted separately |
| --- | --- | --- | --- | --- |
| 0 | 321.5 | 196.6 | 201.8 | 106.4 |
| 1% | 327.6 | 335.9 | not run | not run |
| 3% | 350.8 | none found | not run | not run |
| 10% | 720.8 | none found | 817.8 | 285.8 |

- For the same two independent accumulated counts, observing both is at least as informative as observing their difference. This toy subtraction model omits the reactions during propagation studied by Long et al. (2025), arXiv 2502.20484; it does not establish a performance ordering for their full physical system.
- Spillover is modeled as a fraction of a uniformly random, unknown previous message. "None found" means no searched design reached 1% below 20,000 molecules.
- The original FEEDBACK=answer experiment supplies the **correct previous comparison answer**; FEEDBACK=genie supplies the exact previous pair. These are side-information experiments. At 1% spillover the reported banded versus threshold budgets are 319.3 versus 326.2 with the correct answer, and 287.4 versus 325.5 with the exact pair. From 3% on the searched threshold designs win (results/mc_fb*.jsonl). Actual decoded-answer feedback is evaluated in the continuation below.
- Not modeled: reaction kinetics, nonlinear testbed responses, and spillover from more than one previous message.

## Quick start

    python3 -m pip install -r requirements.txt
    ./reproduce.sh quick        # about 2 minutes, 33 checks; prints each headline number next to its expected value, PASS or FAIL
    ./reproduce.sh extensions   # 24 tests, Gaussian theorem, resource/feedback studies, and figure; outputs to results/rerun/extensions/
    ./reproduce.sh full         # lists the long jobs with their exact commands
    ./reproduce.sh full --run   # runs them one after another (hours); queue scripts write to results/rerun/

Run everything from the repo root. The scripts use nice -n 15 in reproduce.sh; set NICE="" to turn that off. The original environment was Python 3.13, numpy 2.5.3, scipy 1.18.1 (HiGHS), highspy 1.15.1, mpmath 1.4.1, matplotlib 3.11.2. The continuation results also record their actual runtime: Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. They use the existing NumPy/SciPy/Matplotlib dependencies; no new dependency is required. Numerical solvers can choose different equally good policies across versions, so compare validated errors and objective values rather than policy bytes.

## What each script does

| File | What it does |
| --- | --- |
| src/core.py | Poisson LP decoder (worst case error), two point error e2, h(mu), budget for a shape. The error target is the EPS environment variable, default 0.01. |
| src/verify_candidate.py | Checks the 201 photon candidate: 16 pair errors with scipy and with mpmath at 50 digits, the LP decoder on the same encoding, and the rescaled budget. Seconds. |
| src/threshold_optimum.py | Best single randomized threshold decoder for any encoder: minimizes s*(u) g(u) over u, certifies it by adaptive bisection, cross checks with the LP. Argument "table" gives K = 3 to 10. About 10 s for K = 4, about 1 minute for the table. |
| src/banded.py | The peeled band construction for K = 2 to 8, checks the band rule for every pair, prints pairwise and exact LP budgets. Seconds. |
| src/gaussian_milp.py | Gaussian control solved as an exact MILP, plus the band chain cost. About 1 minute with K = 5. |
| src/gaussian_theorem.py | Attaining encoders and every conditional error for the analytic all-K Gaussian theorem. |
| src/resource_benchmark.py | Matched source-rate, background-rate and wall-deadline comparison, with a stopping-policy LP and independent per-pair propagation. |
| src/feedback_sequence.py | Actual decoded-answer feedback, startup/stationary probability propagation, tested repairs and specified adverse-history tests. |
| src/figure_resources.py | Resource-comparison figure from saved numerical results. |
| tests/ | 24 focused tests for the new proofs, per-pair constraints, probability propagation, timing and accounting. |
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

- Analytic proofs and numerical models only. No hardware, no real detector, no dead time, afterpulsing or finite-bandwidth detector model. The resource extension uses a parameterized readout pause and ideal coordinated source gating.
- Floating point throughout. The "certified" numbers are solver dual bounds over a tabulated h (brentq at 1e-10, a 1e-6 margin and MILP tolerances), not exact rational certificates.
- The pairwise relaxation is a necessary condition. The gap between the 195.53 lower bound and the 196.62 design for K = 4 is not closed.
- Designs from opt.py come from Nelder Mead on a small number of starts. They are upper bounds only.
- The novelty search (is this construction or bound already known) is incomplete.
- The first round scripts that produced some numbers were lost and rewritten here. Every rewritten script reproduces the published numbers; the one difference (196.70 versus a looser 196.75) is explained above.
- The calibration model mixes worst case expected signal photons with amortized pilot photons and assumes constant gain between calibrations. The +/-3/sqrt(M) pilot interval is statistical; calibration failures are not allocated part of the 1% decision-error budget, so these percentages are illustrative accounting results rather than end-to-end equal-error guarantees.
