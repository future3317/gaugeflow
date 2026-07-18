# H1a pairwise reciprocal coordinate pretraining v1

Status: **completed; failed; residual removed from production**.

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

## Result

Seed 5801 completed exactly 8,441 steps and 540,164 graph presentations with
finite losses and gradients.  BF16 CUDA training peaked at `2,255.35 MiB`, all
four checkpoints were written, and tensor-free evaluation constructed zero
atlas candidates.  The fixed validation curve was monotone:

| step | coordinate validation |
|---:|---:|
| 0 | 0.74051 |
| 1,250 | 0.65706 |
| 5,000 | 0.58686 |
| 8,441 | 0.53354 |

The run nevertheless failed both primary frozen criteria.  Final coordinate
validation was `0.53354 > 0.35`, and teacher-forced endpoint RMS at `t=.005`
was `0.04494 A > 0.04 A`.  The `t=.1` teacher-forced RMS passed at `0.06384 A`.
One-hundred-step rollouts from `t=.1/.2` passed at `0.06625/0.09858 A`, with
zero failures.

Relative to the same-panel baseline, validation improved from `0.54928` to
`0.53354`, `t=.005` RMS from `0.04640` to `0.04494 A`, and rollout RMS from
`0.07210/0.11007` to `0.06625/0.09858 A`.  Raw train/validation loss was
`0.49212/0.53942`, versus `0.49768/0.55901` previously.  These are consistent
small improvements, but they do not approach the frozen qualification bound.

The fixed 128-graph branch decomposition confirms that the new branch was not
dormant.  At `t=.005`, the complete field explained `0.298` of target energy,
whereas subtracting the pair residual left only `0.085`; at `t=.2` the values
were `0.392` versus `0.189`.  Pair RMS stayed near `0.40` through `t=.2`.
Thus the more general signed pair coefficients repaired part of the
teacher-forced field but did not resolve the dominant representation or
probability-path mismatch.

The protocol therefore fails.  No second seed, extra step, joint
initialization or later Gate is run.  Per the preregistered rule, the pairwise
reciprocal module, tests and active runtime wiring are removed; the exact
implementation remains in Git history at commit `154e6c9`, and this result is
retained as evidence against adding another reciprocal output residual.
