# H1a exact wrapped-score repair

**Frozen decision: failed at H1a.** The repair makes the coordinate target
learnable, but the pre-registered `uniform_log_alpha` reverse grid still fails
the local-geometry guardrail. No H1b--H6 work is authorized.

| Seed | total validation ratio | coordinate validation ratio | fraction with minimum distance >= 0.5 A | minimum distance (A) |
|---:|---:|---:|---:|---:|
| 5201 | 0.64351 | 0.56574 | 0.93750 | 0.24187 |
| 5202 | 0.65642 | 0.65190 | 0.87500 | 0.29966 |
| 5203 | 0.64416 | 0.55312 | 0.95312 | 0.22937 |

The mean coordinate ratio is `0.59026`, so replacing sampled Euclidean-lift
labels by the exact wrapped-normal score fixed the previously observed
near-zero coordinate-learning signal. All validation, finite-training,
categorical-terminal, lattice, tensor-bypass, and sampling-failure checks pass.
The aggregate and per-seed close-contact checks do not.

A subsequent analytic, neural-network-free closure test localizes the remaining
failure to the shared reverse grid. At 50 model evaluations,
`uniform_log_alpha` leaves mean quotient coordinate RMS `0.4001`, whereas
`uniform_time` gives `0.02483`; at 100 evaluations they give `0.3943` and
`0.00127`, respectively. The former grid ends with one coordinate variance
jump from `t=0.3166` to zero. This grid is appropriate for resolving the VP
schedule, but the lattice and categorical implementations already use exact
finite-interval reverse transitions. The torus coordinate step is the only
numerically integrated factor and requires uniform resolution in its linear
variance. A separately frozen diagnostic may therefore re-sample the existing
checkpoints with `uniform_time`; it cannot overwrite this failed result.
