# H1a Cartesian-covector loss integration v1

Status: **frozen before run.**

The preceding target-free attribution found that the compact Cartesian carrier
itself has a gradient norm of only `5.17`, while the physically required map
from Cartesian to fractional covectors multiplies it to `373.27`. The map is
not removable: for row coordinates `r=fL`, scores obey

```text
s_f = s_r L^T,       s_r = s_f L^{-T}.
```

This candidate therefore preserves the exact fractional score consumed by the
reverse torus process, but evaluates the training residual in the orthonormal
Cartesian covector chart. Conditional on the noisy state, `L` is fixed and
invertible, so the two positive-definite quadratic losses have exactly the same
pointwise minimizer. A batched linear solve implements `L^{-T}` without an
explicit inverse. No scale, epsilon, carrier order, loss weight, seed, or
training budget is searched.

The no-training audit uses the same 16 states and initialization as the failed
integration. It checks exact forward and target round trips, FP32/BF16
Cartesian-output gradient stability, the real analytic coordinate-loss
gradient, parameter immutability, atlas bypass, and absence of both legacy
coordinate readouts. Success permits only a separately frozen one-pass,
single-seed coordinate-pretraining experiment.
