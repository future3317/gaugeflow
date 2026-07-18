# H1a geometry precision boundary v2

The zero-training CUDA qualification failed on repeat determinism only, so no
training was started.  Restoring native CUDA reductions raised forward
throughput from `275.74` to `634.19 graphs/s` at `185.73 MiB`; translation,
permutation, chart, gradient and FP32/BF16 precision checks all passed.
Identical-input repeat relative RMSE was `2.18e-5`, above the frozen `1e-6`
limit, confirming the expected atomic-reduction floor.

The prior deterministic implementation performed a full `torch.sort` inside
every reduction merely to revalidate edge and batch ordering already guaranteed
by the production graph contract.  This repeated `O(E log E)` work is not part
of the mathematical method.  A bounded successor may use linear-time
target-contiguous segment reduction without runtime sorting, while retaining
all v2 thresholds and the fixed FP32 geometry boundary.  No training is
authorized by this failed result.
