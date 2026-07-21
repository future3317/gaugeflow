# GaugeFlow-base capacity qualification

## Question and frozen rule

The audit asked whether the previously qualified 5.46M coordinate substrate
was materially under-capacity and, if so, what is the smallest sufficient
GaugeFlow-base scale. Three candidates changed width, depth and channel rank
as a coherent scaling family while retaining the same Cartesian quotient
operators. Data exposure, seed, effective batch, optimizer, precision, clocks,
noise path and evaluation samples were fixed before execution.

Eligibility required a validation coordinate ratio at most 0.297983, t=.6
explained fraction at least 0.648570, clean-side conditional-rollout normalized nearest-neighbour
W1 at most 0.380420, valid-distance fraction 1.0, finite values and zero
sampling failures. Among eligible candidates within 0.02 validation-ratio and
0.03 conditional-rollout-W1 absolute margins of the respective best candidate, the
smallest parameter count was selected.

## Result

Every candidate passed eligibility. Scaling from 5.46M to 34.28M reduced the
matched validation ratio by 0.028408 and increased t=.6 explained fraction by
0.080509. Thus the old model was measurably capacity-limited. Scaling beyond
34M produced a mixed frontier: 98M improved t=.6 explained fraction by another
0.042064 and clean-side conditional-rollout NN-W1 by 0.017593, but did not improve the normalized
validation ratio and reduced training throughput from 238.26 to 69.88
graphs/s. The 58M model likewise improved t=.6 behavior but not rollout W1 or
the normalized validation ratio.

The frozen Pareto-margin rule selects 34.28M. This is not an assertion that
98M has no representational value; it states that its measured incremental
gain is not large enough to justify a 2.85x parameter count and 3.41x training
time in the current Alex one-pass regime.

## Engineering incident

The first launch imported a stale non-editable environment package. The
accumulating candidates failed before their first optimizer step and the 34M
candidate completed one step. Those directories were deleted after confirming
the processes had stopped; the environment was reinstalled editable from the
active worktree and all official runs restarted from empty directories with an
explicit source `PYTHONPATH`. No failed-launch checkpoint or metric enters the
reported result.

The first conditional-rollout evaluation also queried a nonexistent convenience field
on `PackedAlexModelBatch`. The evaluator was corrected to derive graph count
from the lattice batch dimension and rerun from unchanged checkpoints. This
was an evaluation-interface defect, not a training or model change.

## Boundary

Every capacity rollout fixes the source atom types, lattice and node count.
Consequently this Gate selects a conditional coordinate backbone and does not
measure free joint crystal generation. The selected 34M coordinate backbone may seed the separately frozen
GaugeFlow-base joint pretraining protocol. This result alone does not qualify
the joint law, free composition/lattice/coordinates, tensor conditioning, RL,
relaxation, DFT or DFPT.
