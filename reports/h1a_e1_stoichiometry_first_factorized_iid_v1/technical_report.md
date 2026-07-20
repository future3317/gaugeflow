# E1 stoichiometry-first composition qualification

## Outcome

The exact stoichiometry-first kernel is numerically qualified, but its first
factorized one-pass learning screen is **not qualified**. Site assignment,
free-joint H1a, L1/M1, tensor conditioning, oracle work, relaxation, DFT and
DFPT remain blocked.

Only one preregistered learning check failed: final/initial IID calibration NLL
was `0.775688 > 0.75`. The sampler itself is calibrated on the train
population: count-partition TV is `0.035507`, support-size TV is `0.008257`,
element-marginal JSD is `0.001195`, atom-count preservation is `1.0`, and
there are zero invalid states or sampling failures.

## Upstream evaluation-contract correction

The H0 child split is deliberately formula/prototype/matcher-envelope
disjoint. Its matcher envelope contains reduced anonymous stoichiometry and
primitive-site count. Consequently, for every sufficiently populated node
count, train and validation have disjoint integer-partition support. The
measured conditional partition TV is exactly `1.0`; conditional support-size
TV is `0.552677`.

This is not cache corruption. It is the intended OOD split doing what it was
constructed to do. It remains the reference for formula/prototype novelty and
OOD likelihood, but it cannot be used to demand marginal-distribution TV
calibration from a prior trained only on the train split. The original
`h1a_e1_sparse_composition_prior_v1` result remains failed and unchanged.

For implementation calibration only, the new screen uses a fixed 95/5 random
partition of the qualified child-train rows. The fit/calibration index hashes
are recorded in `result.json`. Target composition, formula, structure ID,
prototype, space group, coordinates, lattice and tensor condition are not
model inputs.

## Exact representation

For node count `N`, enumerate the 1,840 positive integer partitions supported
by `N <= 20` and at most seven species:

```text
lambda_1 >= ... >= lambda_S >= 1,  sum(lambda) = N,  S <= 7.
```

The train-only base measure is a symmetric-Dirichlet-smoothed empirical prior
`p0(lambda | N)`. Conditional on `lambda`, element/count pairs are serialized
by decreasing count; equal-count pairs are ordered by increasing element
token. Each unordered composition therefore has exactly one sequence.
Autoregressive masks enforce distinct elements and the equal-count tie rule,
so identical count slots are never over-counted.

The current implementation contains no 1,840-entry learned partition lookup.
It encodes every partition with shared count-position factors and injects the
current count into every species query. This makes unseen partitions
well-defined and removes a non-sharing bottleneck found in the first screen.

## Numerical Q2

On the required WSL CUDA environment and RTX 4060 Ti:

| check | result |
|---|---:|
| catalogue size | 1,840 |
| exhaustive FP64 normalization error | `2.22e-16` |
| 50,000-draw partition TV | `0.006510` |
| sampled/recomputed log-probability error | `1.43e-6` |
| FP32/FP64 maximum error | `1.99e-6` |
| BF16/FP32 mean error | `0.002924` |
| teacher-forced latency, 256 graphs | `3.554 ms` |
| sampling latency, 256 graphs | `15.172 ms` |
| peak allocated memory | `50.63 MiB` |

All frozen Q2 checks pass. The old interleaved species/count autoregressor and
its runtime scripts have been removed; its negative results remain in the
reports and Git history.

## One-pass result

The factorized model was trained from scratch at seed 5705 on 513,155 fit
graphs for exactly 2,005 updates, one pass, BF16, with no EMA. Throughput was
`27,780 graphs/s`; peak CUDA allocation was `53.53 MiB`.

| metric | result | threshold | status |
|---|---:|---:|---|
| final/initial IID NLL | `0.775688` | `<= 0.75` | fail |
| count-partition TV | `0.035507` | `<= 0.10` | pass |
| support-size TV | `0.008257` | `<= 0.05` | pass |
| element JSD | `0.001195` | `<= 0.05` | pass |
| element recall | `1.0` | `>= 0.95` | pass |
| exact atom-count preservation | `1.0` | `1.0` | pass |
| invalid states / failures | `0 / 0` | `0 / 0` | pass |

The factorized encoder improves OOD species NLL from `12.3747` to `11.6940`
and improves the random-initialization ratio from `0.7922` to `0.7757`, but it
does not meet the frozen learning threshold. The OOD total NLL remains
dominated by the intentionally unseen partition support; OOD distribution TV
has no qualification role.

## Decision

Do not add steps, capacity, target composition, a second composition head or a
new sampler. Before another learning mechanism is proposed, the next bounded
task is a zero-training species-law/co-occurrence audit that separates
conditional-likelihood underfit from a too-strong random-initialization-ratio
criterion. Count-constrained site assignment is not authorized by this result.
