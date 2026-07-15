# D0.8 interpretation and stopping rule

D0.8 was selected by the frozen D0.7 decision rule, not by a post-hoc model
search: its phase-1 perturbation audit had first-step amplification `3.5566`
and mean amplification `1.1749`, both above one. The sole modification was a
finite-difference quotient 1-Lipschitz excess penalty; the model, 64 fixed
sources, optimizer, 5,000 steps, sampler, D0.7 direct/rollout/semigroup terms,
and pass thresholds remained fixed.

The final perturbation audit is less expansive (`1.0154` initially and `1.0333`
on average) than D0.7. This validates that the extra term affected the intended
quantity. It does **not** validate the coordinate generator: teacher-forced
RMS (`0.12562`) and free-running 100-map RMS (`0.24745`) both fail their frozen
limits, and the terminal direct loss remains close to one. The appropriate
interpretation is that the chosen local contraction regularization trades map
fit for less local expansion at this fixed budget; it did not create a stable,
accurate semigroup.

This is the one allowed D0.8. The protocol explicitly disallows a second
constraint, a loss-weight or bound search, EMA self-distillation, a local/global
map hierarchy, more steps, P5-D1, harmonic conditioning, oracle work, and
real-tensor experiments. The cycle stops here.
