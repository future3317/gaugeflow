# H1a coordinate clean-side-information v1

This frozen single-seed screen isolates a train--inference contract defect in
the coordinate-only task. Historical coordinate training corrupted element
tokens and lattice states even though every conditional sampler audit supplied
the true elements and lattice. The repaired path corrupts coordinates only;
the architecture, probability path, optimizer, seed, validation panel and
reverse-SDE-100 rollout are unchanged.

At the matched 2,111-step (0.2501166-pass) budget, the EMA validation-coordinate
ratio improves from the historical `0.7383705` to `0.4938223`, an absolute
improvement of `0.2445482`. At `t=0.6`, score explained fraction rises from
`0.1302396` to `0.3907047`. Teacher-forced endpoint RMS is `0.04709 A` at
`t=.005` and `0.06729 A` at `t=.1`; conditional reverse-SDE-100 rollout RMS is
`0.07684/0.12153 A` from `t=.1/.2`, with zero failures. Training sustains
`265.95 graphs/s` with `4835.50 MiB` peak allocated CUDA memory. Every frozen
check passes.

This identifies corrupted observed side information as a major cause of the
coordinate-only residual. It does not reverse the archived Tweedie/ACF
causality result: deterministic state-derived topology still failed to improve
the old checkpoint's held-out residual. No topology branch, extra exposure,
tensor conditioning, oracle, relaxation, DFT or DFPT is authorized, and the
historical H1a status remains unchanged.

The numeric source of truth is `result.json`; checkpoints and training logs
remain under `E:/DATA/T2C-Flow/runs/h1a_coordinate_clean_side_information_v1`.
