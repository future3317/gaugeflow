# H1a exact-count parent assignment Q1

Status: **PASS**, restricted to supported-IID carriers with oracle composition.

The hidden-256 assignment prior was initialized from the one-pass Full-Alex
checkpoint and adapted for 500 steps on 264 carriers from 135 fit materials.
Only 1,502,976 parent-adaptation parameters were updated; the broad chemical
prior remained frozen.  The final checkpoint is identified by SHA-256
`c7aab55047ca47ec36255371b6e1ea8758dddb0e7342f2883948e7a85478383c`.

The first executable candidate passed 13/14 checks.  Its sole failure was the
maximum reveal-order Monte Carlo standard error (`0.60813 > 0.6`) on one
supported calibration carrier.  A preregistered evaluation-only closure used
the unchanged checkpoint, an independent seed and 1,024 reveal orders.  It
reduced the calibration/test maximum standard errors to `0.29793 / 0.11758`
without optimizer updates or threshold changes.  All 14 checks then passed.

Key final evidence:

- supported calibration/test relative NLL reduction: `0.70939 / 0.85290`;
- paired bootstrap UCB95: `-6.35646 / -6.84782`;
- sample retrieval lift: `0.64562 / 0.64741`;
- orbit-aligned site accuracy: `0.93864 / 0.94080`;
- exact-DP mean probability lift over uniform: `0.92149`;
- exact composition: `1.0`; sampling failures: `0`;
- relabel logit residual: `1.0490e-5` (threshold `2e-5`).

One earlier execution requested four `N<=6` exact-DP carriers per IID split,
although the supported calibration split contained only three.  It stopped
after training with no scientific result.  The active runner now validates the
entire exact-DP panel before optimization, and the executable protocol uses the
common `3+3` support.

Boundary: this result does not qualify unseen-action or
formula/prototype-disjoint assignment, generated composition, node count,
lattice, coordinates, joint generation, tensor conditioning, relaxation, DFT
or DFPT.  Those stress panels remain separate from the supported-IID pass.

Files:

- `result.json`: final precision-qualified evidence;
- `candidate_result.json`: unchanged trained-candidate evidence;
- `training_history.csv`: 500-step fit trace;
- `assignment_diagnostics.png`: compact training and likelihood figure.

Implementation commits: training/preflight `a58e7811791b3a303027e195e47257713cec6f07`;
precision closure `6aec0d9aac6085b25d9a7571ad8414e05efd0f67`.
