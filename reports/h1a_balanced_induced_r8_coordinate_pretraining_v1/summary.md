# H1a balanced induced R=8 result

## Decision

**Failed.** The fixed six-iteration balanced transport makes the induced
branch trainable and prevents the original one-slot winner-take-all pattern at
initialization, but it does not qualify the local operator after one exact data
pass. The coordinate validation ratio is `0.533141 > 0.5`; its improvement over
the dynamic-edge predecessor is `0.011025 < 0.02`. R=16 and further balancing,
rank, seed, step, or capacity searches are prohibited by the preregistered
decision rule.

## Frozen run

- seed: `5705`
- train structures / graph presentations: `540,164 / 540,164`
- optimizer steps / data passes: `8,441 / 1.0`
- final logged coordinate loss: `0.334851`
- peak PyTorch CUDA allocation: `6,787.70 MiB`
- tensor candidates: `0`

## Fixed evaluation

| metric | result | threshold | status |
|---|---:|---:|---|
| final / initial EMA validation | 0.533141 | <= 0.5 | fail |
| improvement over dynamic-edge predecessor | 0.011025 | >= 0.02 for material improvement | fail |
| teacher-forced RMS, t=.005 | 0.037761 A | <= 0.04 A | pass |
| teacher-forced RMS, t=.1 | 0.053899 A | <= 0.08 A | pass |
| rollout RMS from t=.1 | 0.054275 A | <= 0.5 A | pass |
| rollout RMS from t=.2 | 0.076667 A | <= 1.0 A | pass |
| sampling failures | 0 | 0 | pass |

The failure is therefore not a low-noise endpoint or integration instability.
It remains a validation-distribution representation/learning failure.

## Slot mechanism audit

The branch is causally active: replacing it by zero at the final checkpoint
increases fixed validation coordinate loss from `0.518065` to `0.946026`
(`+82.61%`). It nevertheless fails all three required non-collapse checks.

| diagnostic | frozen rule | observed |
|---|---:|---:|
| maximum global slot mass | <= 0.14 | 0.195789 |
| minimum representation effective rank | >= 2.0 | 1.351223 |
| maximum mean absolute inter-slot cosine | <= 0.95 | 0.999738 |

Layer 0 retains nearly uniform assignments (minimum normalized entropy
`0.988892`, minimum effective slots `7.82098`) while its slot representations
are almost parallel. In layers 2--3 the learned assignment logits become sharp
enough that six fixed alternating normalizations no longer attain the intended
column masses; simultaneously the deepest representation rank falls below two.
Balanced occupancy alone therefore creates neither specialized slot values nor
a better validation field.

## Scientific boundary

This closes the local-neighborhood mechanism sequence:

1. global reciprocal residual: failed;
2. hard TopK triplets: failed and discontinuous under noisy neighbor ordering;
3. unbalanced induced R=8: used but winner-take-all collapsed;
4. balanced induced R=8: mass pressure is insufficient; failed validation and
   representation non-collapse.

No R=16 run is authorized. The next permitted work is only the preregistered
middle-noise oracle curve, score-residual reciprocal-shell spectrum, and frozen
low-k linear probe. A reciprocal global carrier is implemented only if all
three independently support a low-k deficiency. H1b/H2--H6, tensor/oracle
training, relaxation, DFT, and DFPT remain prohibited.

## Figures and canonical artifacts

- `result.json`: fixed validation, score and rollout result;
- `slot_audit.json`: checkpoint/layer/time slot diagnostics and branch ablation;
- `figures/training_learning_curve.{png,pdf}`: raw training and EMA validation;
- `figures/score_and_rollout.{png,pdf}`: time-resolved score and rollout;
- `figures/slot_diagnostics.{png,pdf}`: occupancy, entropy, rank and cosine;
- `figures/manifest.json`: source SHA256 values and rendering contract.

