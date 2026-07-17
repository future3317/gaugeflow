# H1a stratified-time learning curve v1

**Status: frozen before execution.**

The preceding full benchmark used checkpoints trained on only 80,000 graph
presentations (0.148 of the 540,164-structure split). A read-only gradient
audit then showed that the informative coordinate-score region is concentrated
below `t=0.1`, while iid uniform minibatches and the larger element/lattice
gradients make those updates inefficient under global clipping.

This protocol makes no loss-weight, capacity, score-target, or sampler change.
It uses randomized stratified uniform times, which preserve the exact uniform
marginal objective while covering every time stratum in each minibatch. The
tensor-free forward also skips the unused three-layer Cartesian geometry-query
encoder. A no-write RTX 4060 Ti BF16 measurement found 284.45, 412.32, and
486.91 graphs/s for batches 16, 32, and 64, with peak allocated memory 0.38,
0.85, and 1.72 GiB respectively. Batch 64 is therefore the frozen efficient
choice.

The checkpoints at 1,250, 8,441, and 20,000 steps correspond to the old
80,000-graph exposure, approximately one full training-split pass, and
1,280,000 presentations (2.37 passes). Seed 5301 is screened first. Seeds 5302
and 5303 may run only if every seed-5301 screen check passes. Even a full
three-seed pass authorizes only a separately frozen benchmark against the
training distribution, with held-out structures reserved for novelty and
leakage diagnostics.
