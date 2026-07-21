# GaugeFlow-base equal-exposure capacity screen v1

Status: **qualified; select `small_34m`**.

The former current backbone had 5.46M parameters. This screen compared
34.28M, 57.68M and 97.58M candidates from scratch with seed 5705, exactly
540,164 graph presentations, effective batch 64, the same coordinate-only
clean-side objective, optimizer and evaluation panel. Physical batches were
64/32/16 with graph-weighted accumulation 1/2/4.

| candidate | validation ratio | explained fraction at t=.6 | clean-side conditional-rollout NN-W1 | graphs/s | peak MiB | eligible |
|---|---:|---:|---:|---:|---:|---|
| 34M | 0.269575 | 0.729079 | 0.148713 | 238.26 | 18,078.94 | yes |
| 58M | 0.292842 | 0.757194 | 0.149680 | 118.57 | 16,957.60 | yes |
| 98M | 0.274138 | 0.771143 | 0.131120 | 69.88 | 15,487.36 | yes |

All candidates had finite training/evaluation values, a conditional-rollout valid
minimum-distance fraction of 1.0 and zero sampling failures. The 98M candidate
gave the best mid-noise explained fraction and conditional-rollout NN-W1, but its
improvement over 34M stayed inside the frozen quality margins while its
training throughput fell to 29% of 34M. The preregistered rule therefore
selects the smallest jointly sufficient candidate, 34M.

Relative to the prior 5.46M clean-side checkpoint, 34M improves the matched
validation ratio from 0.297983 to 0.269575 and the t=.6 explained fraction
from 0.648570 to 0.729079. This establishes a real capacity benefit without
paying for an unsupported 100M production default.

The coordinate rollout starts from the coordinate prior but fixes ground-truth
atom types, lattice and node count. It is therefore a clean-side conditional
coordinate rollout, not a full-prior or free-crystal generation result. The
validation ratio divides by each architecture's own random-initialization
loss. Because that denominator differs across widths, absolute final loss is
also retained in `result.json`; it was not substituted after seeing results.
The screen qualifies capacity only. It does not qualify joint GaugeFlow-base,
tensor conditioning, RL, relaxation, DFT or DFPT.

![capacity screen](capacity_screen.png)
