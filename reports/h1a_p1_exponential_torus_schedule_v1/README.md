# H1a exponential torus schedule v1

Status before execution: **frozen, not run**.

The corrected H1a benchmark generated a nearest-neighbour median of 1.6031 Å
against 2.6982 Å in the training reference.  The subsequent oracle-context
reverse closure held species and lattice clean yet ended near 1.8 Å, which
attributes the remaining failure to the coordinate field/path rather than only
to joint-rollout context drift.

For the previous linear variance path, the first torus Fourier mode is
`exp(-2*pi^2*t)` and is already almost erased over most uniformly sampled model
times.  This protocol changes only the coordinate heat schedule to

```text
sigma(0) = 0
sigma(t) = 0.005 * (0.5 / 0.005)^t,  t > 0.
```

Uniform model time is therefore uniform in log coordinate noise.  The endpoint
scale follows the wrapped-coordinate schedule used by DiffCSP, while the local
target remains GaugeFlow's exact wrapped-normal scaled score.  No capacity,
loss weighting, optimizer, sample budget, or tensor path is changed.

Seed 5401 is the preregistered screen.  In addition to the prior learning-curve
checks, its generated median nearest-neighbour distance must reach at least
2.0 Å.  Failure stops the cycle before seeds 5402/5403.  Success permits those
two seeds, then a separately frozen train-reference distribution benchmark.

Reference: Jiao et al., *Crystal Structure Prediction by Joint Equivariant
Diffusion*, NeurIPS 2023, arXiv:2309.04475, Appendix D (fractional-coordinate
noise schedule `sigma_min=0.005`, `sigma_max=0.5`).
