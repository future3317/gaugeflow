# H1a Cartesian-covector coordinate pretraining v1

Status: **completed and failed at the low-noise endpoint; joint training remains
closed.**

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

## Result

The CUDA run used all `540,164` training structures for exactly `8,441` steps
and one data pass. Peak allocated memory was `2,326.74 MiB`, steady throughput
was approximately `390--420 graphs/s`, the final logged coordinate loss was
`0.01831`, and every log and checkpoint was finite.

The fixed validation loss fell from `0.25996` at step zero to `0.02268`, a
ratio of `0.08725` that easily passes the frozen `0.5` bound. EMA train and
validation losses are `0.02024/0.02268`, so the result is not explained by
memorization or a split gap. Both 100-step rollouts pass with zero failures:
mean endpoint RMS is `0.073996 A` from `t=0.1` and `0.116031 A` from `t=0.2`.

The protocol nevertheless fails its low-noise physical criterion. At `t=0.005`
the teacher-forced endpoint RMS is `0.05672 A > 0.04 A`; prediction/target
cosine is only `0.43272` and the explained score fraction is `0.14838`.
The `t=0.1` value `0.07689 A` passes its `0.08 A` bound. The failed historical
fractional-loss pretraining achieved `0.04640 A` at `t=0.005`, so the new metric
greatly improves its own validation objective but worsens the endpoint metric.

This opposing movement is evidence for a primal/dual metric mismatch, not
permission for more training. The Cartesian-covector loss weights fractional
score error by `G^{-1}`, whereas physical endpoint displacement weights the
same error by `G`, with `G=L L^T`. A separately frozen read-only attribution
must test this hypothesis before any probability-path or loss change.
