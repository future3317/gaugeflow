# Q0 legacy C0 no-training root-cause audit

Status: **blocked**. Q0 has no scientific pass label. Q1 authorized: **False**.

The frozen 64 CUDA source noises and 64 fixed-lift couplings were reconstructed without training. Their maximum coupling-cost reproduction error is `1.75879e-06`. Checkpoint-independent conditional-variance, representation-collision, analytic Jacobian, and analytic singular-field solver tables are complete.

The historical P5-C0/D0.4--D0.8 runners did not persist model weights. Consequently the learned embedding, learned vector/flow Jacobians, and old-checkpoint solver convergence cannot be measured. Q0 explicitly prohibits retraining, so this run is blocked rather than silently substituting a new model.

## Missing required diagnostics

- R_embed(t)
- learned reduced vector-field Jacobian
- learned flow Jacobian and log determinant
- legacy checkpoint Euler/RK4/adaptive rollout convergence

## Decision

Stop at Q0. Do not run Q1 or any later gate. Historical reports and thresholds remain unchanged.
