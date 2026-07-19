# H1a induced R=8 coordinate DSM result

## Decision

The current-layer-context R=8 induced-slot run is a failed, diagnostically
collapsed result. It does not authorize R=16, joint training, reciprocal
features, H1b--H6, tensor conditioning, oracle work, relaxation, DFT, or DFPT.

The run used seed 5705, all 540,164 training structures, 8,441 steps, batch
size 64, and exactly one graph pass. It produced zero sampling failures and
zero tensor-atlas candidates.

## Main result

| measure | result |
|---|---:|
| validation final/initial ratio | 0.545825 |
| dynamic-edge predecessor ratio | 0.544167 |
| improvement over predecessor | -0.001658 |
| teacher-forced endpoint RMS, t=.005 | 0.037551 A |
| teacher-forced endpoint RMS, t=.1 | 0.053815 A |
| rollout RMS, t=.1/.2 | 0.056066 / 0.079584 A |
| score explained fraction, t=.2/.5/.9 | 0.600964 / 0.345166 / 0.000265 |
| sampling failures | 0 |

The `t=.9` value is a high-noise DSM identifiability diagnostic, not evidence
for a missing global feature. At that time the analytic endpoint oracle itself
has RMS 2.17696 A. The actionable middle-noise result is `t=.5`, where the
analytic oracle remains essentially exact but induced slots do not improve the
learned score.

## Slot-collapse attribution

The branch is active: replacing every induced angular output by zero in an
offline diagnostic changes the fixed final validation coordinate loss from
0.530390 to 1.095359, a relative degradation of 106.52%. Nevertheless, the
assignment becomes winner-take-all in deeper blocks:

| final block | minimum effective slots over audited times | maximum slot mass |
|---:|---:|---:|
| 0 | 7.520 | 0.170 |
| 1 | 1.758 | 0.880 |
| 2 | 1.212 | 0.951 |
| 3 | 1.348 | 0.876 |

All blocks begin with high assignment entropy. Collapse appears first in
blocks 2--3 by step 1,250 and is established by step 5,000. The failure is
therefore not an initialization gradient delay: zero-optimizer CUDA
qualification measured first-step angular/context gradient norms of
6.19e-4/4.47e-5 and 6.60 effective slots. It is learned assignment collapse.

The next bounded mechanism test is balanced per-center entropic assignment at
the same R=8. It changes neither the model capacity nor the DSM objective. R=16
is stopped because increasing rank before preventing collapse would only add
unused slots.

Primary machine-readable evidence:

- `result.json`
- `slot_audit.json`
- `../h1a_edge_query_angular_kernel_comparison_v1/result.json`
