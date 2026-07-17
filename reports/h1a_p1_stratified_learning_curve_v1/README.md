# H1a stratified-time learning curve v1

**Status: seed-5301 screen passed; remaining two seeds authorized.**

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

## Seed-5301 screen result

The screen completed 20,000 steps with approximately 500 graphs/s and 2.27 GiB
peak allocated CUDA memory. The final-to-initial validation ratios were
`0.55947` total and `0.22536` coordinate, passing the frozen `0.65` and `0.30`
bounds. All 128 reverse trajectories completed, no MASK remained, all lattices
were finite with positive volume, and every structure had minimum periodic
distance at least 0.5 A. The minimum/median/maximum nearest distances were
`0.6617 / 1.6950 / 3.0653 A`.

The validation curve shows that coordinate loss improves from `0.22601` at
initialization to `0.06063` at the old 80,000-graph exposure, `0.05383` after
approximately one split pass, and `0.05093` at 20,000 steps. The later gain is
real but small. Moreover, the generated median remains well below the roughly
2.70 A training-reference median. This result authorizes seeds 5302 and 5303
under the same protocol; it is not H1a qualification.
