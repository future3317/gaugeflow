# H1a geometry precision boundary v1

The zero-training CUDA qualification failed on efficiency only, so no training
was started.  Fixing geometry-dependent message blocks, coordinate edge
encoding and the Cartesian carrier to FP32 repaired every numerical contract:
BF16 translation relative RMSE fell to `1.53e-5/2.03e-5`, repeat error was
zero, FP32/BF16 output cosine was `0.99981`, loss-gradient cosine was
`0.99759`, all chart and permutation tests passed, and peak memory was only
`185.73 MiB`.

Forward throughput was `275.74 graphs/s`, below the frozen `500 graphs/s`
minimum.  The protocol is therefore failed and cannot authorize training.
The prior deterministic-segment candidate is no longer needed for correctness
once geometry messages use FP32 and is a plausible avoidable cost.  A bounded
successor may restore native vectorized reductions while preserving this fixed
geometry precision boundary and every existing numerical/efficiency threshold;
it may not add capacity, steps, seeds or a fallback precision branch.
