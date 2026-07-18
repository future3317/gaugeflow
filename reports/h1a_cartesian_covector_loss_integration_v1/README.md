# H1a Cartesian-covector loss integration v1

Status: **qualified; one single-seed, one-pass coordinate pretraining may be
frozen separately.**

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

## Result

The first execution exposed a coverage error before archival: the BF16
fractional covector transform was still under autocast (`0.04033` round-trip
residual), while the runner checked only FP32. That apparent pass was rejected.
The single production path now keeps the physical `L^T` transform in FP32 and
the runner requires both precisions to pass the unchanged `2e-5` threshold.

The corrected run passes every frozen check:

| quantity | FP32 | BF16 |
|---|---:|---:|
| Cartesian output-energy gradient norm | 5.17118 | 5.95536 |
| analytic Cartesian coordinate loss | 0.12612 | 0.13034 |
| coordinate-loss gradient norm | 5.46074 | 6.37376 |
| prediction fractional round trip | 1.19e-7 | 1.19e-7 |
| target fractional round trip | 2.38e-7 | 2.38e-7 |

The FP32/BF16 output cosine is `0.99313`, relative RMSE is `0.11848`, gradient
cosine is `0.97105`, and the BF16/FP32 gradient-norm ratio is `1.15165`. The
model has exactly `4,479,161` parameters, 80 carrier channels, zero legacy
readout parameters, zero tensor-atlas candidates, and zero optimizer steps.

This qualifies the representation and loss metric only. It does not establish
that the coordinate field learns from data and does not qualify H1a.
