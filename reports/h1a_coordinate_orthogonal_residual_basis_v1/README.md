# H1a coordinate orthogonal-residual basis v1

Status: **frozen before run; no training has been performed.**

The current vector and edge coordinate branches are each locally complete but
complementary over the fixed 16-state panel. Their joint exact readout fits
better than either branch alone, yet it obtains that fit through `32.31x`
vector/edge cancellation and is unstable in BF16. This protocol changes
neither physical output span nor training target. It tests a target-free,
graph-equal block Gram--Schmidt chart of the existing combined basis.

For weighted vector and edge designs `V` and `E`, the chart first whitens `V`,
then forms the exact residual

```text
P = (V^T W^2 V)^-1 V^T W^2 E,
E_perp = E - V P,
B = [V A_v, E_perp A_e],
```

where `A_v` and `A_e` are inverse-transpose Cholesky factors. Therefore
`(W B)^T(W B)=I`, while `B=[V,E]T` for an invertible, fixed, upper-triangular
channel transform `T`. The construction reads states and model features but no
coordinate target. It preserves the exact combined span, node permutation,
translation horizontality, and Cartesian covariance because it mixes only
scalar feature channels. No ridge, rank truncation, pivoting, learned target
projection, precision search, or runtime fallback is allowed.

The audit uses the same 16 states, initialization, noise and times as the two
preceding stability audits. It fits the target only after the chart is frozen,
then checks span identity, graph-equal orthogonality, FP32/BF16 prediction and
backbone gradients, orthogonal block cancellation, and CUDA operator cost. It
performs zero optimizer steps and cannot change the failed H1a result or open a
later Gate.
