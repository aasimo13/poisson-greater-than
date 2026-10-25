# Actual decoded-answer feedback

This experiment extends the one-step spillover model in `src/mc.py`. It distinguishes a receiver supplied with the correct previous comparison from a receiver that feeds back its own previous decoded bit. The encoder vectors are the committed, three-decimal values in `results/mc_fb_ans.jsonl`; their original optimizer precision is not available.

## Reproduce

```sh
python3 src/feedback_sequence.py
python3 -m unittest discover -s tests -p 'test_feedback_sequence.py' -v
```

The script writes `results/feedback_sequence.json`, including unrounded probability results, lossless run representations of the fitted decoders, startup trajectories, stationary distributions, numerical residuals, and tested rescalings. A different output path can be selected with `--output`.

## Model and guarantees

Let $J_t=(A_t,B_t)$ be an input pair, let $F_j=[A>B]$, and define $\mu_j=x_A+y_B$. Under the IID experiment, each pair is independently uniform over the 16 possibilities. Conditional on the true input sequence,

\[
N_t\sim\operatorname{Poisson}(\mu_{J_t}+\rho\mu_{J_{t-1}}),
\]

and counts in different windows are independent. Only the immediately preceding window contributes spillover. These are stipulated stochastic-channel assumptions, not a model of molecule trajectories or reaction kinetics. Correlated arrival processes and longer channel memory are outside the experiment.

The photon/molecule budget is $\max x+\max y$, the current signal contribution used in the original repository. It does not include the additional carryover arriving in that window, detector cost, or feedback energy.

The original correct-answer policy solves a linear program using

\[
P(N_t=n,Z_{t-1}=z\mid J_t=j)
=\frac1{16}\sum_{i:F_i=z}\Pr(N_t=n\mid J_{t-1}=i,J_t=j).
\]

The LP minimizes the worst error over the **current** input pair. Previous inputs are averaged. Its fitted policy is $d_z(n)=\Pr(\widehat F_t=1\mid z,n)$.

The actual-feedback experiment uses this fitted policy unchanged but sets $z=\widehat F_{t-1}$, the receiver's actual previous output. This evaluates transfer of an oracle-trained policy; it does not optimize a sequential policy.

### Exact finite-state probability propagation

The Markov state is $(J_{t-1},\widehat F_{t-1})$, with 32 states. Define

\[
q_{ijz}=\sum_n \Pr(N_t=n\mid i,j)d_z(n).
\]

Then the transition from $(i,z)$ to $(j,1)$ is $q_{ijz}/16$, and to $(j,0)$ is $(1-q_{ijz})/16$. If $w_{iz}$ is the state distribution before the current message, the error for current pair $j$ is

\[
e_j=\sum_{i,z}w_{iz}\bigl[F_j(1-q_{ijz})+(1-F_j)q_{ijz}\bigr].
\]

This is direct probability propagation, without Monte Carlo sampling. The calculation uses floating-point Poisson probabilities and an LP solver, so “exact finite-state” describes the method, not an exact-arithmetic numerical certificate.

Counts $0,\ldots,C-1$ are represented explicitly, with a final bin for $N\ge C$. The decoder is constant on that final bin. Its probability is evaluated with the Poisson survival function, so no probability is deliberately discarded. Tiny PMF summation errors are normalized and reported.

### Startup and all subsequent windows

The fresh-start experiment has no spillover in the first window. A separate no-spillover decoder processes that first actual message, producing a joint distribution over its true pair and decoded bit. The feedback policy is then used in subsequent windows. Two other initializations are also reported: a correct previous answer and an independent fair previous bit.

The script propagates 12 subsequent windows and computes the stationary distribution. To bound later windows, it uses total-variation contraction under the transition matrix. The Dobrushin coefficient is

\[
\tau=\frac12\max_{s,s'}\|M_{s,:}-M_{s',:}\|_1.
\]

For a computed stationary vector $\widetilde w$, its distance to the true invariant law is at most

\[
\frac{\|\widetilde wM-\widetilde w\|_1}{2(1-\tau)}.
\]

The finite-prefix errors, the remaining distance to stationarity, and this residual correction yield a bound on every subsequent IID-history-averaged current-pair error. The first-window error is included separately. These remain floating-point bounds, not interval-arithmetic certificates. The code refuses the bound if its one-step contraction check fails.

## Results on the stored encoders

Errors below are percentages. Every comparison in a row uses the **same encoder vectors** and signal budget. The no-feedback column therefore is not the separately reoptimized no-feedback budget from the earlier README.

| Stored shape | Spillover | Signal budget | No feedback, refitted | Correct previous answer | Actual feedback, stationary |
|---|---:|---:|---:|---:|---:|
| Banded | 1% | 319.336 | 1.156506% | 0.999974% | 1.002228% |
| Threshold | 1% | 326.172 | 1.011316% | 0.999883% | 1.000256% |
| Banded | 3% | 590.429 | 1.338989% | 0.999868% | 1.002165% |
| Threshold | 3% | 339.160 | 1.103612% | 0.999884% | 1.001359% |
| Threshold | 10% | 442.676 | 2.246096% | 0.999997% | 1.008619% |

The distinction is measurable but modest for these cases. Oracle-trained policies slightly exceed the 1% target once their own decoded outputs are fed back. These results do not demonstrate catastrophic error propagation under IID inputs.

### Tested repairs by scaling

For each stored shape, the script tests a finite list of scale factors. At each factor it refits the correct-answer policy, evaluates actual feedback, and checks the fresh first decision plus the bound on all subsequent IID windows. It returns the first tested scale passing with a numerical acceptance margin. No monotonicity or global optimum is claimed.

| Shape and spillover | Passing scale | Passing signal budget | Fresh-start/all-windows numerical error bound |
|---|---:|---:|---:|
| Banded, 1% | 1.0025 | 320.13434 | 0.998215% |
| Threshold, 1% | 1.0001 | 326.2046172 | 0.999835% |
| Banded, 3% | 1.0025 | 591.9050725 | 0.997176% |
| Threshold, 3% | 1.0005 | 339.32958 | 0.998995% |
| Threshold, 10% | 1.0050 | 444.88938 | 0.995903% |

Thus these selected policies can be repaired with tested budget increases of 0.01–0.5%. This does not establish optimal feedback budgets or optimal shapes.

## History dependence is a separate, larger issue

The earlier no-feedback banded design at 1% spillover has a current signal budget of 335.888 using its stored rounded levels. Its worst current-pair error averaged over uniform previous pairs is 0.999467%. Holding the previous pair at $(0,3)$ and the current pair at $(2,1)$ gives 9.561118% error. This reproduces the review's counterexample without contradicting the disclosed random-history model.

The new script also tests a deterministic sequence alternating $(0,3)$ and $(2,1)$, while propagating the actual decoded-bit probabilities around the cycle. For the original oracle-trained encoders, the maximum phase errors are 9.226285% for the 1%-spillover banded case and 1.291698% for its threshold counterpart. At 3% spillover they are 14.866299% and 2.126367%, respectively. These are specified stress sequences, not maximizations over all possible histories.

The finite scale repairs above are validated for IID inputs. They do not convert the design into an arbitrary-sequence 1% guarantee. The difference between averaged histories and particular histories is much larger here than the correction caused by replacing correct-answer side information with actual feedback.

## Validation

The eight tests cover explicit overflow mass, the zero-spillover control, direct Poisson-CDF checks of a deterministic threshold, Markov row sums and IID pair marginals, agreement between correct initialization and oracle training, a known periodic stochastic-bit rule, the disclosed history counterexample, and coverage of a long forward run by the all-future bound. LP outputs are also reevaluated from the probability matrices rather than treating the solver objective alone as the measured error.
