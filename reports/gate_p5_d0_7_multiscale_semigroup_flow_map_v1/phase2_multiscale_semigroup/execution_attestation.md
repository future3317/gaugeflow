# D0.7 phase-2 execution attestation

The immutable runner hash recorded in `manifest.json` is
`2984af236b8e4c41dff0f7bcdc3ff91441f3ddc6c53bb8045bfc56635fbf2ff0`; the
pre-run protocol hash is
`99c18ec48b684cbefa4f8e4df5da306d49aa70887fd651ac14fe16051b12bf96`.

For every regular row, the executed sampler chooses one registered total span
`delta` uniformly, samples `s` uniformly in `[0, 1-delta]`, then sets
`u=s+delta/2` and `v=s+delta`. Twenty-five percent of rows instead use a direct
endpoint map with `u=v=1` and are excluded from the strict two-step losses.
This is the exact behavior in `_stratified_batch_times` and `d07_losses`.

The `rollout_midpoint` prose field in the JSON was a non-operative stale
description inherited while drafting the sampler: it mentions independently
sampled later spans, which the runner does not do. It did not affect any random
draw, loss, threshold, model, source, optimizer, or reported result. The runner
and this attestation, rather than that stale descriptive field, define the
executed midpoint-triple experiment. The original JSON is deliberately left
unchanged after the run so its manifest hash continues to identify the actual
pre-run artifact. This disclosure prevents treating the run as evidence for an
unexecuted independently-spaced triple sampler.
