# Relation to existing work

This is a targeted comparison with primary sources inspected on 2026-10-25,
not an exhaustive novelty or patent search. The repository currently supports
a specific finite-alphabet numerical result. The broad idea of designing
signals so a shared physical channel computes a function is established.

## Primary sources

| Source | Relevant established idea | Distinction from this repository |
| --- | --- | --- |
| S. Razavikia, J. M. B. da Silva Júnior, C. Fischione, [*ChannelComp: A General Method for Computation by Communications*](https://arxiv.org/abs/2310.06532), 2023 preprint; IEEE Transactions on Communications 72, 692–706 (2024) | Optimizes signal mappings to compute functions directly over a multiple-access channel, including mappings that adapt existing modulation. | It establishes the general approach. The inspected source did not establish this repository's four-level Poisson GREATER-THAN budget or minimax lower bound. |
| S. Razavikia, C. Fischione, [*On Designing Modulation for Over-the-Air Computation—Part I: Noise-Aware Design*](https://arxiv.org/abs/2506.15950), 2025 | Tailors constellation design to the noise distribution using a max–min formulation, with Gaussian, Laplace and heavy-tailed examples. | Noise-aware representation design is prior work. This repository uses nonnegative Poisson intensities, signal-dependent noise, and a per-input comparison-error constraint. |
| S. Razavikia, C. Fischione, [*The Computing Channel: How Modulation Programs the Airwaves*](https://arxiv.org/html/2609.11145v1), September 2026 | Describes function-oriented mappings and receiver decisions; Section II explicitly includes nonnegative optical intensities and Poisson observations with background. | Especially close conceptual context. The inspected article does not supply the same GREATER-THAN construction or numerical bounds. Its broad programming principle should not be claimed as a new discovery here. |
| J. Long et al., [*Towards a Molecular Computer: Enabling Arithmetic Operations in Molecular Communication*](https://arxiv.org/html/2502.20484v1), 2025 | Encodes signs using two molecule types and uses reactions during propagation for subtraction, with computational studies. | The repository's signed difference of independent accumulated counts omits reaction kinetics and propagation detail. Its toy receiver comparison does not invalidate the physical model in this paper. |

## The precise candidate contribution

The result to investigate for originality is the following combination:

- Independent deterministic nonnegative encoders for `A,B in {0,1,2,3}`.
- One summed Poisson count and a receiver that computes `A > B`.
- At most 1% error for every current input pair in the memoryless model.
- A nonmonotone construction at approximately 196.62 peak expected detected
  signal photons, compared with approximately 321.54 for the optimal
  single-threshold class.
- A numerical lower bound near 195.53 over every encoder and randomized
  receiver in that one-count model, leaving a gap under 0.6%.

The targeted reading above did not identify this exact combination. That is
not proof of novelty. Claims of priority would require a wider search and
expert review. The all-alphabet Gaussian theorem in [theory.md](theory.md)
is presented as an elementary control proof, without a novelty claim.

## Interpretation and limits

The main experimental mechanism is an arrangement of conditional count
distributions that fits the task and Poisson noise. The leakage calculation
shows more individual-input information beyond the answer for the banded
scheme, so the saving should not be explained as a general theorem that
discarding more input information always uses fewer photons. The 2026
Computing Channel article likewise treats privacy as an additional design
objective, not an automatic consequence of function computation.

A matched resource comparison must identify the number of resolved
observations, transmitter rate limits, background rates and receiver costs.
One-count near-optimality does not extend automatically to time-slot,
multi-detector or adaptive designs. The current band construction's loss of
advantage at larger alphabets is a boundary of that construction, not an
impossibility proof for every nonmonotone encoding.
