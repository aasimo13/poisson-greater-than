# When does a photon-efficient comparison become a faster decision?

Research continuation, 25 October 2026. Original repository snapshot:
`08b0ef546e9492177b93a61b0ff2b09539a4be1d`.

## Main finding

The four-level nonmonotone encoding remains a strong result for one summed
Poisson observation: approximately 196.62 peak expected signal photons versus
321.54 for the optimal threshold class, with a recorded numerical lower bound
near 195.53. Extending the model to include a source-rate limit, detector
readout pauses and a common deadline changes which tested design is preferable.

Four separate bit slots are fastest when readout is inexpensive. At larger
readout cost, an adaptive banded receiver has the lowest uniform-input mean
decision time among the tested designs. Each accepted policy is checked against
the 1% error limit for every input pair. Its mean advantage does not imply a
lower worst-case latency.

The new feedback experiment also resolves an assumption in the molecular
extension. Replacing a supplied correct previous answer with the receiver's
actual previous decoded bit causes small but measurable error-budget overruns
under independent uniform inputs. Modest tested intensity increases repair
those cases. Particular input histories remain much more difficult.

These are analytic and numerical findings in specified models, not measurements
of an optical chip or molecular device. The work does not establish an
electrical-energy saving, a globally optimal protocol, or novelty priority.

## 1. Preserve the original result and make its scope explicit

The original task has independent inputs `A,B in {0,1,2,3}` and independently
applied, deterministic nonnegative encoders. Their codebooks can be jointly
designed offline. A receiver observes `N ~ Poisson(x[A]+y[B])` and decides
whether `A>B`, with ties false. Its error must be at most 1% for every one of
the 16 pairs. The cost is `max(x)+max(y)`, a maximum over **expected** detected
signal counts, not a hard cap on a realized Poisson observation.

The original 201-photon construction reproduces with worst error
0.959541929022% in the high-precision check. The peeled construction reaches
approximately 196.62, about 38.85% below the threshold benchmark. The recorded
195.528066 lower bound leaves a relative gap under 0.6%. Its mixed-integer
program uses a necessary pairwise-testing relaxation and floating-point
tables/solver tolerances. The long four-level bound was inspected but was not
rerun in this continuation; the original quick suite recomputes the three-level
bound and other headline values.

The [theory note](theory.md) now explains both threshold envelopes and the
pairwise lower bound. An analytic Poisson-affinity guard also closes the tiny
interval next to equal means that the numerical threshold subdivision omits.
The larger-alphabet table is correctly described as an analytic reduction
followed by a numerical search, rather than claiming the same explicit
interval certificate for every row.

The saving is specific to this observation resource. The simple four-slot
construction uses at most `4 log(400) = 23.9659` expected photons at a 1% error
bound, because it resolves four observations. The new resource study tightens
that baseline within a symmetric-bit, equal-slot family to approximately
22.7756 by fitting the actual comparison error instead of a conservative
four-bit union bound.

## 2. Put every design on the same clock

The [resource experiment](resource_methods.md) uses a common peak detected
signal rate `R` for each transmitter, a known background rate `b`, a readout
pause `tau`, and a wall-clock deadline `D`. Sources and integration pause during
readout; coordinated stopping and gating are idealized, with their costs
included in the readout parameters. Uniform input probabilities define the
mean-time objective. They do not replace the 16 separate error constraints.

For the displayed cases, `R=1`, `R*D=200`, and `R*tau` takes four values. A
physical rate can scale these times: choosing one million detected signal
counts per second makes the table entries microseconds. This is a unit
conversion, not a claim about a particular detector.

### No-background timing results

| Readout pause R*tau | Fixed threshold | Fixed banded | Four bit slots | Best tested adaptive banded |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160.780 | 98.314 | **22.776** | 44.504, up to 8 reads |
| 10 | 170.780 | 108.314 | **62.776** | 65.429, up to 4 reads |
| 25 | 185.780 | 123.314 | 122.776 | **105.975**, up to 2 reads |
| 50 | misses deadline | 148.314 | misses deadline | **124.853**, up to 2 reads |

Entries are uniform-input mean wall times multiplied by R. The maximum wall
time for an adaptive policy can reach 200, whereas the fixed banded scheme
finishes at its displayed time for every pair. Missing the deadline excludes a
design from the comparison; it does not imply the underlying communication
problem is impossible.

![Matched-rate decision times and readout costs](../figures/fig_resources.png)

For the two fixed alternatives alone, the fitted four-slot design has wall
time `22.7756/R + 4*tau`, and the fixed banded design has wall time
`98.3144/R + tau`. Their crossover is therefore approximately

\[
R\tau=25.18.
\]

This is a comparison of two specified designs. Other bit protocols, unequal
slot lengths, additional detectors or jointly decoded counts may improve it.

### Adaptive stopping is evaluated for each input pair

The adaptive source shape stays fixed. After each scheduled count observation,
the policy can stop with either answer or continue. Available grids have 2, 4
or 8 observations, with active horizon `H=D-M*tau` for `M` checks. The policy
is fitted using an occupancy-flow linear program under a mixture of the 16
Poisson hypotheses. A separate convolution calculation then propagates each
hypothesis through the exported policy, reevaluating errors and costs.

At zero readout cost the 8-check policy uses about 44.504 mean active-time
units, 24.318 mean signal photons, and 50.003 maximum expected signal photons.
It improves the fixed banded scheme but the four-slot design still wins this
tested zero-overhead comparison. At readout pause 25, the 2-check adaptive
policy improves mean latency by about 13.7% over the four-slot design.

The guarantee is a numerically checked feasible policy on a selected finite
grid. It is not a proof of continuous-time sequential optimality. Explicit
overflow behavior is supplied, with overflow probability conservatively charged
as error and full-duration operation. The largest charged overflow probability
in the reported sweep is below `2.5e-13`; accepted errors are below 1%.

## 3. Background is a rate, and the encoding must match it

Changing exposure duration also changes background counts. Holding a fixed
number of background counts per window would conceal that dependence.

At `b/R=0.01`, the original peeled shape needs about 279.430 active-time units,
so it misses the 200-unit deadline even before readout. A previously searched
background-aware shape restores feasibility when refitted to this rate model:

| Fixed source shape | Active time multiplied by R | Maximum expected signal photons |
| --- | ---: | ---: |
| Threshold | 162.686 | 325.372 |
| Original peeled band | 279.430 | 558.861 |
| Stored background-retuned band | **161.818** | **323.603** |
| Four bit slots | **31.187** | **31.187** |

The retuned candidate is taken from the original stored background experiment;
it has not been globally optimized for this new model. Including it prevents
the original shape's failure from being misread as a failure of every
nonmonotone encoding. Its signal-photon saving against the tested threshold
shape is only about 0.54% in this selected rate scenario.

Electrical energy is left as a parameterized accounting rule:

\[
E_i=c_\gamma\mathbb E_i[N_{\rm signal}]
  +P_{\rm active}\mathbb E_i[T_{\rm active}]
  +E_{\rm read}\mathbb E_i[J].
\]

The coefficients must be measured or supplied. `E_read` includes electronics
and control during each pause, so it is not double-counted as active exposure
power. The exported break-even calculations use mean quantities consistently;
per-pair arrays permit a consistent worst-case energy calculation as well.

## 4. Actual feedback makes a small IID correction; histories matter more

The earlier `FEEDBACK=answer` mode supplied the true previous comparison. The
[new experiment](feedback_methods.md) feeds back the receiver's actual output.
Its 32-state Markov chain tracks the previous true pair and decoded bit, with
conditionally independent Poisson observations and one previous window of
mean-based spillover. No molecular trajectories or reaction kinetics are
modeled.

| Stored design | Spillover | Correct-answer error | Actual-feedback stationary error |
| --- | ---: | ---: | ---: |
| Banded | 1% | 0.999974% | 1.002228% |
| Threshold | 1% | 0.999883% | 1.000256% |
| Banded | 3% | 0.999868% | 1.002165% |
| Threshold | 3% | 0.999884% | 1.001359% |
| Threshold | 10% | 0.999997% | 1.008619% |

These are small target overruns under IID uniform inputs. A finite grid of
intensity scales supplies tested repairs, validated for the initial
no-spillover decision and all subsequent IID-history-averaged windows using a
total-variation contraction bound. At 1% spillover, the banded candidate passes
at signal budget 320.13434, a 0.25% increase. The threshold candidate passes
at 326.20462, a 0.01% increase. These are feasible tested points, not optimized
sequential budgets.

The history assumption remains more consequential. For a sequence alternating
input pairs `(0,3)` and `(2,1)`, actual-feedback maximum phase error is 9.226%
for the original banded design and 1.292% for its threshold counterpart at 1%
spillover. The specified stress sequence is not an exhaustive adversarial
search, but it demonstrates why an IID-history-averaged guarantee cannot be
advertised as a guarantee for arbitrary input streams.

## 5. A stronger Gaussian control and a narrower interpretation

The [Gaussian theorem](theory.md#4-gaussian-control-an-optimum-for-every-k)
now covers every alphabet size, not just the small solver cases. For additive
Gaussian noise with standard deviation `sigma` and conditional error limit
`epsilon`, the optimal nonnegative amplitude cost is

\[
P^*(K)=4(K-1)\sigma\Phi^{-1}(1-\epsilon).
\]

Every pair of levels within either transmitter must be distinguishable,
forcing a minimum range; opposing equally spaced encoders attain the bound.
This is an amplitude result for the Gaussian control, not a photon count or a
claimed new theorem in the literature.

Together with the leakage calculation, it clarifies the mechanism. The useful
encoding fits the task to the physical noise. It does not follow that it
retains less information about individual inputs. Nor does the fixed
four-level construction scale automatically: the same peeled family loses to
the threshold benchmark by six input values.

The [related-work comparison](related_work.md) explicitly addresses ChannelComp,
noise-aware modulation, the September 2026 Computing Channel article, and the
molecular-computing paper. The broad task-oriented communication principle is
established. The originality of this precise Poisson construction and bound
remains an open literature question.

## Reproduce and inspect

```sh
./reproduce.sh quick
./reproduce.sh extensions
```

The original quick suite has 33 passing numerical checks. The extension suite
has 24 focused tests and reruns the analytic construction, complete resource
study, feedback study and figure generation. Reruns go to
`results/rerun/extensions/`, preserving the committed snapshots. The original
long four-level lower-bound computation is separate.

Full unrounded policies, errors and environment metadata are in
[resource_benchmark.json](../results/resource_benchmark.json) and
[feedback_sequence.json](../results/feedback_sequence.json). These results were
generated with Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. Solver choices can
change an equally good policy across versions; numerical verification should
compare feasibility and objective values rather than demand identical policy
bytes.

The next experimental decision is to select a real detector/source interface
and measure its rate, readout latency, gating cost, background and deviations
from Poisson behavior. Those measurements locate the hardware on the computed
resource comparison. A broader protocol search and a wider novelty review
remain separate tasks.
