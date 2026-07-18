# H1a coordinate causal audit v2

**Result: the low-noise field improved but remains incomplete.** The corrected full benchmark failed only
the nearest-neighbor distribution metric after node count, element marginal,
volume, categorical completion, and lattice validity had passed. This audit
repeats the time-resolved conditional-score calibration on all three final EMA
checkpoints. It does not modify the failed H1a decision.

If the teacher-forced low-noise score remains weak, the next bounded mechanism
must address coordinate optimization or field representation. If that score is
well calibrated while free sampling remains over-dense, the next audit instead
starts reverse trajectories from forward-noised real states to isolate
rollout-state distribution shift. No repulsion loss, extra training steps, H1b,
tensor conditioning, or oracle work is authorized by this diagnostic.

Across the three EMA checkpoints at `t=0.005`, prediction/target norm is
`0.543--0.602`, cosine is `0.611--0.615`, and explained conditional-target
fraction is `0.373--0.378`. The one-step endpoint RMS remains `0.547--0.557 A`.
At `t=0.01`, norm/cosine fall to approximately `0.46--0.49 / 0.47--0.48` and
endpoint RMS is `0.925--0.938 A`; at `t=0.02`, only about 7% of the conditional
target is explained and endpoint RMS is about 1.50 A. Stratified training thus
substantially repaired the old near-zero field but did not produce a strong
intermediate-noise denoiser.

The conditional DSM target is not itself the population marginal-score oracle,
so these numbers alone cannot distinguish irreducible posterior uncertainty
from rollout-state failure. The next frozen diagnostic is the oracle-context
forward-noise/reverse closure.
