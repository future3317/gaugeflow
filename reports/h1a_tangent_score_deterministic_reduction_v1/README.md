# H1a tangent score with deterministic reductions v1

The zero-training CUDA qualification failed and no training was started.
Vectorized `segment_reduce` removed the previously observed execution-order
noise: identical-input FP32 and BF16 repeat errors were exactly zero, and
permutation consistency passed.  Tangent round trips, GL(3,Z)/O(3) chart
covariance, gradients, parameter/carrier contracts and atlas bypass also
passed.

BF16 translation consistency nevertheless failed at `0.03411/0.03642`
Cartesian/fractional relative RMSE, while FP32 remained
`1.19e-5/1.64e-5`.  The edge set is unchanged under the translation.  The
remaining cause is therefore precision sensitivity in the geometry-dependent
message path, not reduction order or the tangent chart.  This protocol is
failed and cannot authorize training.  Any subsequent no-training proposal
must explicitly qualify the geometry-sensitive precision boundary and its
throughput; it may not change thresholds, seeds or optimizer budget.
