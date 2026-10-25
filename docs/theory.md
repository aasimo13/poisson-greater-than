# GREATER-THAN through a noisy summed observation

This note states the model and separates analytic arguments from numerical
evidence. It also gives an all-alphabet proof for the Gaussian control. These
are mathematical and computational results for specified observation models;
no hardware performance or priority of invention is asserted. Context and
primary literature are in [related_work.md](related_work.md).

## 1. Model and cost

Two transmitters know inputs `A` and `B` in `{0,...,K-1}`. They independently use
deterministic nonnegative encoders `x[A]` and `y[B]`. The receiver observes one
random variable and outputs `F = 1[A > B]`; ties are false. The receiver may
randomize using private randomness independent of the inputs. The two codebooks
may be designed jointly offline. Randomized transmitters, shared-randomness
switching between codebooks, and input-dependent cooperation between the
transmitters are outside this model.

For the ideal counting model,

\[
N\mid(A=a,B=b)\sim\operatorname{Poisson}(x_a+y_b).
\]

The numbers `x` and `y` are expected detected signal counts integrated over
the measurement window, not rates. For a decoder `q_n = Pr(output=1 | N=n)`,
every one of the `K^2` pairs must satisfy

\[
\Pr(\widehat F\ne 1[a>b]\mid a,b)\le\epsilon.
\]

No input prior is needed for that requirement. The objective is

\[
P=\max_{a,b}\mathbb E[N\mid a,b]
  =\max_a x_a+\max_b y_b.
\]

It is a worst-case **expected** count, not a hard upper limit on a realized
Poisson count. A background mean `beta` changes the observation mean to
`x_a+y_b+beta`; the signal-only cost still excludes `beta`. The main bounds
below concern zero background. Rates, exposure duration, emitted photons,
detector losses, control energy, dead time and switching are additional
physical resources that require their own model.

## 2. A two-point test supplies a necessary condition

Write `e2(mu,nu)` for the smallest possible maximum of the two conditional
errors in testing `Poisson(mu)` against `Poisson(nu)`. For `mu < nu`, the
likelihood ratio increases with the count, so a threshold with possible
randomization at its boundary attains this minimax error. Define

\[
h_\epsilon(\mu)=\inf\{\nu>\mu:e_2(\mu,\nu)\le\epsilon\}.
\]

Any decoder that works for all input pairs also works when restricted to any
one true pair and any one false pair. Their means must therefore obey

\[
\max(m_t,m_f)\ge h_\epsilon(\min(m_t,m_f)).
\]

This is necessary, but it does not ensure that a *single* decoder works for
all pairs simultaneously. Pair-specific optimal tests can conflict. The
common decoder is checked separately by the linear program in
[`core.py`](../src/core.py).

The function `h` is increasing: adding an independent Poisson count is a
common stochastic transformation of both hypotheses and cannot improve their
distinguishability. In fact, for `c >= 0`,
`h(mu+c) >= h(mu)+c`. This monotonicity permits the lower staircase used by
[`lb_pairwise2.py`](../src/lb_pairwise2.py): on a table cell `[g_i,g_(i+1)]`,
`h(mu) >= h(g_i)`. Lower convex-hull facets of this staircase remain valid
underestimators. The mixed-integer formulation permits either orientation
of every true/false mean pair. Its minimization **dual bound** then bounds
the original problem from below.

The stored four-level run has a dual bound of `195.52806605660902`; the peeled
construction checked with a common decoder uses approximately `196.62`.
The resulting relative gap is about `0.56%`. The cap `P <= 200` used for the
lower-bound search does not discard a better solution, since a feasible
design below 200 is known. See [recorded output](../results/lb_k4.jsonl).

These are numerical certificates: the `h` table, root solving, rounding margin
and solver tolerances use floating point. They are not exact rational or
interval-arithmetic certificates. A feasible point of the pairwise relaxation
is not an achievable design unless it also passes the common-decoder check.

## 3. Exact reduction for the best single-threshold encoder

### 3.1 Low counts mean true

Let `s` be the largest true mean and `r > s` the smallest false mean, and put
`delta = r-s`. A low-true threshold has its largest conditional errors at
these two means, because its output probability is monotone in the Poisson
mean. Its optimized error is therefore `e2(s,r)`.

Compare false pair `(a,a)` with true pair `(a+1,a)`, and false pair `(a,a)`
with true pair `(a,a-1)`. These inequalities force

\[
x_a-x_{a+1}\ge\delta,\qquad y_{a+1}-y_a\ge\delta.
\]

Thus the brightness order is forced by the threshold rule. In particular,

\[
P=x_0+y_{K-1}\ge (x_0+y_0)+(K-1)\delta
 \ge r+(K-1)\delta=Kr-(K-1)s.
\]

For `K >= 3`, nonnegativity and the true pair `(1,0)` require
`s >= (K-2)delta`. This condition is also sufficient. Set

\[
c=s-(K-2)\delta\ge0,\quad
x_a=(K-1-a)\delta,\quad y_b=c+b\delta.
\]

The largest true mean is `s`, the smallest false mean is `r`, and the cost
equals the bound. Therefore, writing `u=r/s`, the exact encoder envelope is

\[
P=s\,[Ku-(K-1)],\qquad 1<u\le1+\frac1{K-2}.
\]

### 3.2 High counts mean true

Now let `s` be the largest false mean and `r>s` the smallest true mean.
The adjacent comparisons force `x` to increase and `y` to decrease by at
least `delta=r-s`. Nonnegativity requires `s >= (K-1)delta`. Moreover,

\[
P=x_{K-1}+y_0\ge r+(K-2)\delta=(K-1)r-(K-2)s.
\]

Equality is attained by

\[
c=s-(K-1)\delta,\quad x_a=a\delta,\quad
y_b=c+(K-1-b)\delta.
\]

Consequently the exact envelope is

\[
P=s\,[(K-1)u-(K-2)],\qquad 1<u\le1+\frac1{K-1}.
\]

### 3.3 What the numerical threshold certificate establishes

For either envelope define `s_*(u)` by `e2(s_*(u),u*s_*(u))=epsilon`.
It is nonincreasing in `u`. If the envelope multiplier is `g(u)`, then

\[
\min_{u\in[u_1,u_2]}s_*(u)g(u)\ge s_*(u_2)g(u_1).
\]

[`threshold_optimum.py`](../src/threshold_optimum.py) recursively subdivides
intervals until these lower bounds approach the candidate cost. For `K=4`,
the low-true result is approximately 321.540, with a numerical lower bound
321.5366; the high-true result is approximately 452.319. The algebraic
reduction is exact, while those numerical endpoints have the stated tolerance.
The script's larger-`K` table uses numerical minimization without running
that interval certificate separately for every row.

The interval subdivision starts at `u=1+1e-9`. A prefix guard in `certify()`
now checks the interval next to 1 analytically. The affinity between Poisson laws is
`exp(-(sqrt(r)-sqrt(s))^2/2)`. Cauchy-Schwarz gives
`TV <= sqrt(1-affinity^2)`, while error at most `epsilon` under both
hypotheses requires `TV >= 1-2*epsilon`. Hence

\[
s\ge\frac{-\log(4\epsilon(1-\epsilon))}{(\sqrt u-1)^2}.
\]

For `epsilon=0.01` and `1<u<1+1e-9`, this exceeds `1e19`. It cannot improve
the displayed hundreds-of-photons candidates. The guard computes
`sqrt(u)-1` stably as `(u-1)/(sqrt(u)+1)` and requires the prefix lower
bound to reach the requested certificate tolerance. This closes that
small interval; it does not eliminate the floating-point
qualifications of the implemented interval bounds elsewhere.

## 4. Gaussian control: an optimum for every K

**Theorem.** Let `K >= 2`, `0 < epsilon < 1/2`, and `sigma > 0`. In the model

\[
Z=x_A+y_B+W,\qquad W\sim\mathcal N(0,\sigma^2),\qquad x_a,y_b\ge0,
\]

allow any randomized decoder with error at most `epsilon` for every pair.
Then, with `d=2*sigma*Phi^(-1)(1-epsilon)`,

\[
\boxed{\min(\max x+\max y)=2(K-1)d.}
\]

**Proof.** Two equal-variance Gaussian laws with mean separation `Delta`
have minimax error `Phi(-Delta/(2*sigma))`: the midpoint threshold attains
equal conditional errors, and their equal-prior Bayes error lower-bounds
every test's maximum conditional error. Thus opposite-answer means must be
at least `d` apart.

Take any `a<a'`. Pairs `(a,a)` and `(a',a)` have opposite answers and share
the same `y_a`, so `abs(x_a-x_a') >= d`. This applies to every distinct pair
of `x` levels. Sorting the `K` levels yields `range(x) >= (K-1)d`.
For any `b<b'`, pairs `(b',b)` and `(b',b')` similarly imply
`abs(y_b-y_b') >= d`, hence `range(y) >= (K-1)d`. Nonnegative levels give

\[
\max x+\max y\ge\operatorname{range}(x)+\operatorname{range}(y)
\ge2(K-1)d.
\]

For attainment choose `x_a=(K-1-a)d` and `y_b=bd`. True means are at most
`(K-2)d`, false means are at least `(K-1)d`, and a threshold at
`(K-3/2)d` gives each pair error at most `epsilon`. The cost is `2(K-1)d`.
This proves both the lower bound and attainment. No assumption of monotone
encoders was used in the lower bound. QED.

For unit variance and 1% error, `d=4.652695748...`. The optimum is `4d`, `6d`
and `8d` for `K=3,4,5`, agreeing with the earlier MILP control, and the proof
also covers every larger alphabet. The Gaussian objective is a signal
amplitude cost in this model; its numeric values are **not photon counts**.

[`gaussian_theorem.py`](../src/gaussian_theorem.py) evaluates the attaining
construction and every conditional error. The tests compare it with the
separate MILP at small `K`. These numerical checks support the implementation;
the argument above, rather than a finite sweep, proves the all-`K` statement.
No claim is made that this elementary theorem is new to the literature.

## 5. Four bit slots use a different observation resource

For `K=4`, write each input in two bits and send the four bits in separate
slots. With zero background, send no photons for bit zero and an expected
`lambda=log(4/epsilon)` photons for bit one. The receiver declares one iff
at least one photon arrives. A transmitted one is missed with probability
`exp(-lambda)=epsilon/4`; a zero never becomes one. The union bound ensures
that the comparison error is at most `epsilon` for every pair. Peak expected
signal cost is

\[
4\log(4/\epsilon)=23.965858188...\quad(\epsilon=0.01).
\]

This is a feasible construction, not an optimal four-slot claim. Recovering
all four bits is sufficient but can be unnecessary for the comparison. It
does not conflict with the approximately 195.53 single-count lower bound,
because it uses four separately resolved observations. Actual latency
requires a rate and overhead model; four slots need not have the same
duration as one high-count window.

## Reproduce the analytic control

```bash
python3 src/gaussian_theorem.py --max-k 8
python3 -m unittest discover -s tests -p 'test_gaussian_theorem.py' -v
```

The code assumes the already documented NumPy/SciPy dependencies. It adds no
hardware constants and does not rerun the long Poisson lower-bound search.
