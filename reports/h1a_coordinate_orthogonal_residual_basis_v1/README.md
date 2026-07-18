# H1a coordinate orthogonal-residual basis v1

Status: **failed before training; post-hoc basis decorrelation is insufficient.
H1a remains failed.**

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

## Result

The algebraic chart passes exactly. Its weighted Gram condition number is
`1.000000004`, maximum Gram error is `4.96e-10`, and its fitted prediction
differs from the original combined-span projection by only `1.35e-10`
relatively. The orthogonal parameter norm is `3.2299`, vector/residual block
cancellation is `1.3801`, and the CUDA chart operator costs `0.0255 ms` and
`0.360 MiB` on the fixed panel. FP32 also retains the expected fit: MSE
`0.099464`, low-time endpoint RMS `0.020287 A`, and backbone gradient norm
`4.809`.

The mixed-precision qualification nevertheless fails decisively. BF16 MSE is
`9.76787` (`98.20x` FP32), endpoint RMS is `0.30036 A`, and prediction relative
RMS is `4.2604`. The BF16 backbone gradient norm is `14670.5`, or `3050.5x`
the FP32 norm, with cosine only `0.1278`. Although the stored orthogonal
coordinate is small, its effective raw coefficient norm remains `9108.38`.
Consequently, a fixed invertible readout chart cannot remove amplification of
BF16 feature perturbations; it only changes parameter coordinates after the
ill-scaled features already exist.

No optimizer step or production mutation occurred. The candidate is rejected
and the active combined head remains unchanged. A successor must alter how a
compact, scale-controlled Cartesian coordinate carrier is formed before the
final readout, rather than add another post-hoc whitening, scale, ridge,
precision switch, or solve schedule.
