# H1a pairwise reciprocal coordinate pretraining v1

Status: **frozen, not run**.

The pairwise reciprocal-torus operator passed its separate mathematical,
precision, gradient, vectorization and RTX 4060 Ti capacity gate.  This
experiment now changes exactly one mechanism relative to the failed
`h1a_coordinate_pretraining_v1`: it adds that initially-zero global periodic
residual to the unchanged local coordinate field.

Seed 5801 trains from scratch on all 540,164 train graphs exactly once.  Model
capacity, optimizer, loss, probability path, sampler, batch size, BF16 mode,
validation rows, noise streams, rollout panel, checkpoints and every threshold
remain the same.  The model has 4,482,234 parameters, only 9,072 more than the
failed baseline.  The archived low-rank reciprocal structure-factor model is
not restored and there is no runtime switch or fallback.

Passing requires coordinate validation at most `0.35`, teacher-forced endpoint
RMS at most `0.04 A` at `t=.005` and `0.08 A` at `t=.1`, bounded 100-step
rollouts from `t=.1/.2`, zero failures and zero tensor candidates.  Failure
removes the new residual from production; it cannot be rescued by another seed
or more steps.  This experiment cannot itself qualify H1a or start H1b.
