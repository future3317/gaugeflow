# H1a Cartesian-tangent coordinate pretraining v1

Status: **completed and failed; joint initialization remains closed.**

This is the single learning experiment authorized by
`h1a_geometry_precision_boundary_v3`.  Seed 5705 trains the current compact
Cartesian carrier from scratch on all 540,164 qualified training structures
for exactly 8,441 optimizer steps and one complete data pass.  The fractional
reverse drift is treated as a tangent vector,

```text
v_r = v_f L,        v_f = v_r L^-1,
```

and the geometry-sensitive path uses the one fixed FP32 contract qualified
before training.  No tensor condition, external checkpoint, extra seed, extra
step, loss search, or runtime fallback is used.

## Result

All checkpoints and training records are finite.  Peak allocated CUDA memory
is 2,859.84 MiB and the final checkpoint contains exactly 540,164 graph
presentations.  The fixed validation coordinate loss decreases monotonically
from 34.43436 to 30.46289, 26.04380, and 24.24037 at steps 0, 1,250, 5,000,
and 8,441.  Its final/initial ratio is 0.70396, above the frozen maximum 0.5.

The physically comparable endpoint and rollout results are:

| check | result | frozen bound | status |
|---|---:|---:|:---:|
| validation final/initial | 0.70396 | <= 0.5 | fail |
| teacher-forced RMS at t=.005 | 0.04207 A | <= 0.04 A | fail |
| teacher-forced RMS at t=.1 | 0.06143 A | <= 0.08 A | pass |
| rollout RMS from t=.1 | 0.06589 A | <= 0.5 A | pass |
| rollout RMS from t=.2 | 0.09861 A | <= 1.0 A | pass |
| sampling failures | 0 | 0 | pass |
| tensor-atlas candidates | 0 | 0 | pass |

At `t=.005`, prediction/target cosine is 0.58503 and the explained score
fraction is 0.33753.  EMA train and validation coordinate losses are
22.13035/24.24037, so the failed validation reduction is not explained by a
large train/validation gap.  The tangent repair improves the archived
Cartesian-covector endpoint RMS from 0.05672 A to 0.04207 A and improves both
rollouts, but it does not meet the preregistered gate.

The decision is therefore to stop before joint initialization.  The result
cannot be rescued by changing the threshold, adding seeds or steps, or adding
a second mechanism.  H1b--H6, tensor conditioning, oracle work, relaxation,
DFT, and DFPT remain closed.
