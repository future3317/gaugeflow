# H1a coordinate-only pretraining v1

Status: **completed; failed; joint initialization is not authorized**.

Exact quotient scores close generic, real, and high-symmetry endpoints, and no
alternative score integrator improves the failed joint checkpoint.  The prior
gradient audit instead measured that element and lattice objectives dominate
the useful low-noise coordinate gradient under global clipping.  This protocol
therefore trains the unchanged 4.47M-parameter model from scratch for exactly
one full train pass using only the Rao--Blackwell coordinate DSM objective.

This is a representation/optimization qualification, not a generative result:
element, volume and shape heads are deliberately not optimized.  Passing
requires held-out coordinate loss, teacher-forced endpoint estimation, and
100-step oracle-context rollout to pass simultaneously.  It permits only a
separately frozen joint initialization experiment.  It cannot qualify H1a or
start H1b.

## Result

Seed 5705 processed all 540,164 train graphs exactly once in 8,441 optimizer
steps.  The CUDA BF16 run remained finite, used 2,216 MiB peak allocated
memory, produced all four preregistered checkpoints, and did not construct any
tensor candidates.  Fixed validation coordinate loss decreased monotonically
across checkpoints:

| step | coordinate validation |
|---:|---:|
| 0 | 1.03675 |
| 1,250 | 0.69204 |
| 5,000 | 0.58944 |
| 8,441 | 0.54937 |

The final result failed two frozen criteria:

| check | observed | threshold | result |
|---|---:|---:|:---:|
| coordinate validation | 0.54937 | <= 0.35 | fail |
| teacher-forced endpoint RMS at t=0.005 | 0.04640 A | <= 0.04 A | fail |
| teacher-forced endpoint RMS at t=0.1 | 0.06740 A | <= 0.08 A | pass |
| 100-step rollout RMS from t=0.1 | 0.07208 A | <= 0.5 A | pass |
| 100-step rollout RMS from t=0.2 | 0.11006 A | <= 1.0 A | pass |
| sampling failures | 0 | 0 | pass |

Coordinate-only optimization therefore repairs the severe local rollout
instability of the failed joint checkpoint, but it does not qualify the
full-time score regression in one data pass.  The learning curve is still
descending, yet the frozen protocol forbids extending it or selecting an
earlier checkpoint.  A post-hoc time decomposition also shows that the field
already decays near the mixed-torus end, so an analytic high-noise envelope is
not supported as the primary repair.

The subsequent state-visibility audit identifies a more specific source of
sample inefficiency: the translation-only target changes strongly under an
equivalent repeated-species row relabeling that leaves the noisy state and
unlabeled endpoint unchanged.  No joint initialization or additional training
is run.  The next admissible implementation task is to qualify a smooth joint
translation--permutation quotient target against exact small-site oracles.
