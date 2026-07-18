# H1a Cartesian-covector coordinate pretraining v1

Status: **frozen before run.**

This is the only learning experiment authorized by the qualified compact
carrier and Cartesian-covector loss integration. It uses the same seed, full
`540,164`-structure training split, optimizer, BF16 precision, batch size, and
exactly one data pass as the failed historical coordinate pretraining. The
only scientific changes are the already-qualified carrier and the equivalent
Cartesian loss metric; the probability path, analytic target, fractional
reverse score, sampler, and evaluation endpoints are unchanged.

Because the loss chart changes its numerical units, the old absolute validation
loss threshold is not reused. Before training, this protocol freezes a required
`>=2x` reduction from the step-zero validation loss. The physically comparable
teacher-forced endpoint and rollout thresholds remain unchanged. Failure stops
before joint initialization and cannot be rescued by more steps or seeds.
