# Reciprocal NO-GO synthesis

No reciprocal experiment should be rerun. Two frozen-checkpoint, zero-optimizer
audits already answer complementary questions.

## Independent Bridge audit

Source:
`E:/CODE/T2C-Flow/gaugeflow-bridge/reports/h1a_reciprocal_residual_spectrum_v1/result.json`

Source SHA-256:
`81cc416467ca05494d0f9a81ba19a10934ca0b26879d79f8117ee8145e385fe4`

On the volume-normalized seed-5705 step-8441 checkpoint and 256 fixed
validation graphs, it reports:

- middle-noise low-shell excess over the atom-permutation null: `0.007755 < 0.10`;
- middle-noise held-out low-frequency explained fraction: `-0.001368 < 0.10`;
- low-frequency minus random-Fourier explained fraction: `0.000695 < 0.05`;
- low-frequency minus graph-token explained fraction: `-0.001368 < 0.05`;
- translation, permutation, O(3), and GL(3,Z) checks passed;
- optimizer steps: `0`, and checkpoint parameters were unchanged.

Its frozen decision is `NO-GO: stop the reciprocal Bridge direction and do
not integrate a production branch`.

## Main-worktree confirmatory audit

The adjacent main-worktree artifacts use the later dynamic-edge checkpoint and
add same-composition endpoint identifiability, physical reciprocal bands, and a
matched high-frequency control. All three of their own preregistered checks
also fail. This is confirmatory evidence, not permission to repeat or tune the
Bridge experiment.

## Joint interpretation

Together with the failed TopK and induced-slot studies, the evidence rejects
both stronger aggregation on the current noisy graph and a low-frequency
global Fourier correction. It does not show that clean coordination topology
is unrecoverable. The next allowed zero-training question is whether an oracle
clean topology materially explains the residual and whether a frozen probe can
predict that topology from the current noisy state. No tensor or later Gate is
authorized.
