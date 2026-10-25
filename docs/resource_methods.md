# A common clock for the photon comparison

This benchmark extends the original one-count result into an explicitly
constrained comparison of **expected photons, elapsed time, and resolved
observations**. It is a numerical study of selected designs. It is not a hardware
measurement or an optimization over all possible communication protocols.

## Physical assumptions and shared constraints

Each transmitter has the same peak detected signal rate `R`. The background is a
known constant **rate** `b`; it contributes `b*t` expected counts during an active
exposure of duration `t`. This differs from holding background counts per window
constant while changing the window length.

The sources and integration pause during every readout. A pause of duration
`tau` includes gate/control, detector reset, and decision latency. For an adaptive
scheme, a common control signal can stop both sources or start the next exposure.
That coordinated gating is an ideal assumption, with its costs charged to the
readout parameters. There is no free continuous observation between the selected
checks. Dead time, afterpulsing, bandwidth, non-Poisson noise, imperfect source
extinction, and propagation delay outside the supplied pause are not modeled.

Every design must satisfy the same 1% conditional error limit for **each of the
16 current input pairs** and the same wall-clock deadline `D`. Uniform inputs are
used only to define the mean-time/mean-energy objective; they do not replace the
16 error constraints. A row exceeding `D` is reported but excluded from the
fastest-feasible comparison.

Default dimensionless choices are `R=1`, `R*D=200`, `b/R in {0, .01}`, and
`R*tau in {0,10,25,50}`. Times scale as `1/R`. For illustration only, choosing
`R=1,000,000 detected signal counts/second` converts a time value of 100 into
100 microseconds. That rate is not asserted to describe any real detector.

## Compared designs

### Fixed shared-count designs

The threshold shape and the original peeled-band shape are normalized to obey
each source's rate cap. Background scenarios also include the previously searched
`bg=1` nonmonotone shape from the **first record of `results/bgdrift.jsonl` at
commit `08b0ef5`**, named `banded_bg1`. Its stored intensities are converted to
rates and refitted under this new rate-based model; it is not a fresh global
optimization. Their shapes are held fixed, while exposure and a common
randomized count decoder are fitted to the error requirement. For signal rate
`s_ab = x[a] + y[b]`, the observation is

\[
N\mid(a,b)\sim\operatorname{Poisson}((s_{ab}+b)t).
\]

Their wall time is `t + tau`. Their worst expected signal-photon cost is
`max(s_ab*t)`. In background, these are achievable fixed-shape results; no global
optimality is claimed for the shape after the channel changes.

### Four serial bit slots

Two bits encode each input, using one counter and four separately resolved,
equal-duration slots. A zero emits no signal; a one emits at rate `R`. The binary
minimax decoder has equal false-positive and miss probabilities `q`, including
randomization at the threshold if needed.

The code explicitly enumerates the decoded two-bit numbers and all 16 comparison
errors. It fits the largest `q` that obeys the comparison error limit, then fits
the slot duration to `q`. Thus it uses the full comparator error allowance, rather
than requiring the unnecessarily strong union bound `4q <= 1%`.

This is an optimized **symmetric-bit, equal-duration family**, not the best possible
four-slot comparator. Unequal bit errors, unequal durations, joint decoding and
adaptive bit ordering are outside this benchmark.

With no background, the fitted design uses about **22.7756** maximum expected
signal photons and the same active time at `R=1`. The simpler deterministic
on/off rule with `log(400)` photons per bright bit remains a valid conservative
baseline at `4*log(400) = 23.9659` photons.

Wall time is `4*t_bit + 4*tau`; maximum expected signal photons are `4*R*t_bit`.
The mean is half that for uniform input pairs.

### Adaptive banded design

Each selected banded source-rate shape stays fixed. A policy can stop and return false,
stop and return true, or continue, after each of `M` selected count observations.
Default candidate grids use `M in {2,4,8}`. All include the same **wall deadline**:

\[
H=D-M\tau,\qquad \Delta=H/M.
\]

Here `H` is the active-time horizon. A design with insufficient active time is
rejected before optimization. A decision at check `k` has elapsed wall time
`k*(Delta+tau)`. The fastest tested design is selected from the feasible fixed
schemes and these adaptive grids.

The stopping problem uses an occupancy-flow linear program. Define the reference
law as the equal mixture of the 16 homogeneous Poisson-process hypotheses. At
active time `t` and count `n`, its posterior hypothesis weights are

\[
w_i(t,n)=\frac{p_i(n,t)}{\sum_j p_j(n,t)}.
\]

The likelihood ratio of hypothesis `i` to the reference law is `16*w_i`, bounded
by 16. Conditioning on survival under the common policy does not change this
ratio: conditional on a total count and active time, the allocation of arrivals
within the previous intervals is independent of the homogeneous rate.

Reference transition probabilities are therefore mixtures of the next-interval
Poisson increments with weights `w_i`. Variables are the reference mass assigned
to each of the three actions at every `(t,n)` state. Flow conservation is linear;
each hypothesis error is the incorrectly labeled stop mass weighted by `16*w_i`.
The objective minimizes uniform-input **mean wall time**.

After optimization, the code converts masses to action probabilities and runs a
separate forward calculation under **each individual hypothesis**, using direct
Poisson convolutions rather than LP flow values. It recomputes all errors,
expected stopping times, and expected readout counts. A candidate that fails the
1% limit is rejected. Expected signal photons are `s_ab * E_i(active time)`, valid
for the bounded stopping time with constant active-time rate.

Counts are truncated beyond a Poisson tail of approximately `1e-12`; very small
LP transitions are omitted. The independent verifier uses full convolution
probabilities. An overflow policy continues to the deadline and returns false;
the verifier conservatively charges all overflow probability as error, the full
active horizon, and all readouts. Small numerical safety margins are applied to
the requested error. These are floating-point validations, not exact rational
certificates or an optimum over continuous observation times and all encoders.

## Results under the common deadline

The table reports **uniform-input mean wall time**, with `R=1`, `b=0`, and
`D=200`. A dash means that the maximum wall time violates the common deadline.
Adaptive results choose the fastest tested grid.

| Readout pause | Fixed threshold | Fixed banded | Four bit slots | Best tested adaptive banded |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160.780 | 98.314 | **22.776** | 44.504 (8 checks available) |
| 10 | 170.780 | 108.314 | **62.776** | 65.429 (4 checks available) |
| 25 | 185.780 | 123.314 | 122.776 | **105.975** (2 checks available) |
| 50 | — | 148.314 | — | **124.853** (2 checks available) |

All accepted designs have independently evaluated worst-pair error below 1%.
Adaptive policies can use the full 200-unit wall deadline on difficult cases;
their lower mean does not imply lower worst-case latency than the fixed banded
scheme. Some tests select different adaptive grids for latency and photon cost.

For just the two fixed alternatives, equality of their wall times occurs at

\[
\tau_\times=(98.314\ldots-22.7756\ldots)/3\approx25.1796.
\]

Thus separately resolved bits win at inexpensive readout; the shared count can
win when readout is costly. Adaptive stopping supplies a further mean-latency
option under the common deadline, subject to the ideal gating assumption.

At background rate `b/R=.01`, the unchanged peeled shape requires about **279.430**
active time and **558.861** peak expected signal photons, missing `D=200` even
before a readout pause. However, the stored background-retuned `banded_bg1`
candidate recovers feasibility: about **161.818** active time and **323.603**
peak expected signal photons, compared with **162.686** and **325.372** for the
fixed threshold shape. Thus using a nominal encoding in a changed channel gives
a misleadingly pessimistic result for nonmonotone encodings as a class. The
retuned candidate's photon saving is small in this selected rate scenario, and
neither shape is claimed optimal there. The four-bit family needs about
**31.1869** active time and peak expected signal photons. At pause 50 no tested
background-case scheme meets the deadline; that is a limitation of this menu,
not an impossibility theorem for all protocols. Background **rate** accounting
matters because a longer exposure also accumulates more background.

## Electrical energy remains a parameterized comparison

For each pair, use the same accounting rule:

\[
E_i=c_\gamma\,\mathbb E_i[N_{\rm signal}]
       +P_{\rm active}\,\mathbb E_i[T_{\rm active}]
       +E_{\rm read}\,\mathbb E_i[J].
\]

`c_gamma` is effective electrical energy per expected **detected** signal photon,
including source/link inefficiency; `P_active` is other power during active
integration; `E_read` includes electronics throughout the pause, gate/control,
reset and decision. These coefficients require hardware measurements or supplied
assumptions. A photon count by itself is not a total-energy measurement.

The exported readout break-even expression uses uniform-input means throughout:

\[
E_{\rm read}<
\frac{c_\gamma(\bar N_{\rm ref}-\bar N_{\rm design})
      +P_{\rm active}(\bar T_{\rm ref}-\bar T_{\rm design})}
     {\bar J_{\rm design}-\bar J_{\rm ref}},
\]

when the denominator is positive. This avoids mixing peak photon cost with mean
controller cost. Per-pair arrays are saved so a worst-case total-energy objective
can also be evaluated consistently.

## Reproduction and checks

```
python3 src/resource_benchmark.py --out results/resource_benchmark.json
python3 -m unittest discover -s tests -p 'test_resource_benchmark.py' -v
```

The output includes every conditional error, expected time, expected photons,
readout count, endpoint decoder and adaptive policy table, plus runtime versions.
Tests check the known no-background budget, scaling with physical rates, exact
four-bit comparison probabilities, a genuine wall-deadline exclusion, the forward
propagator against a single-terminal-count calculation, the stopping-policy error
and deadline constraints, and consistent mean-energy break-even accounting.
