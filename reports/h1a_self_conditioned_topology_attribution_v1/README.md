# H1a self-conditioned topology attribution v1

Status: **completed; NO-GO for staged/self-conditioned topology**.

This zero-optimizer audit used the frozen two-pass EMA checkpoint, the exact
all-pair v2 topology definition, the same 512/256 train/validation panels, and
the original v2 noise stream at `t=0.4/0.5/0.6`. It compared current noisy
topology, the frozen linear probe, a quotient Tweedie endpoint topology, and
the true clean-topology oracle. The checkpoint fingerprint was unchanged.

The endpoint estimate is

```text
xhat_0 = P_Q[x_t + sigma(t) * predicted_scaled_score].
```

It reads only the noisy state and frozen EMA score. The clean endpoint is used
only after prediction to form labels and metrics. One clean-oracle Cartesian
carrier coefficient is fitted on the 512-structure train panel and reused
unchanged for all four validation fields; variants do not receive separate
readouts.

At the preregistered focus time `t=0.6`, the Tweedie field reduces topology MSE
by `0.31269` relative to the noisy field, but its AUC is `0.77003 < 0.8` and its
explained fraction is only `0.08615`. More importantly, using that topology in
the shared carrier changes coordinate residual energy by `-0.04955`, with a
structure-bootstrap 95% interval `[-0.06020,-0.03890]`; the clean oracle gives
`+0.14203`. The frozen linear probe is more predictive (`AUC=0.81964`) but is
also non-causal in this operator (`-0.05405`).

The strict decision is
`self_conditioned_topology_not_predictive_revisit_conditional_variance`.
The result is not evidence that all self-conditioning is impossible: it shows
that the current one-step quotient Tweedie estimate does not meet the frozen
topology criterion and that neither it nor the stronger frozen probe can be
plugged into the existing linear topology-to-vector carrier. It therefore
does not authorize ACF, a staged production branch, more exposure, sampler
search, H1b--H6, tensor conditioning, oracle work, relaxation, DFT, or DFPT.

The rerun reproduces the prior two-pass clean-oracle gains at
`t=0.4/0.5/0.6` as `0.04099/0.09577/0.14203`, which verifies panel and noise
alignment. Full structure-level metrics are in the adjacent CSV files.
