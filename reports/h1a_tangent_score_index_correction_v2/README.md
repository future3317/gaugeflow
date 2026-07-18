# H1a tangent-score index correction v2

The zero-training CUDA qualification failed and no training was started.  The
direct-difference integer-lift rewrite did not reduce the edge displacement
drift (`1.43e-6 A > 7.5e-7 A`) and made the formal BF16 translation relative
RMSE `0.01081`.  Every tangent-chart, gradient, precision, parameter, carrier,
atlas-bypass and FP32 symmetry check still passed.

An exact read-only repeat of the same executable and inputs produced BF16
translation/permutation relative RMSE `0.00518/0.00949`, versus
`0.01081/7.06e-6` in the formal run, while FP32 stayed near `1e-5`.  The edge
keys and count were identical.  This run-to-run exchange identifies
nondeterministic CUDA atomic neighbor reductions as the dominant failed check;
it is not evidence against the tangent metric.  V2 is frozen failed.  The lift
rewrite must be removed.  Only a separately frozen, vectorized deterministic
segment-reduction qualification may retain the tangent candidate; thresholds
and training budget may not change.
