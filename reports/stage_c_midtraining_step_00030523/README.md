# Stage-C 20k mid-training evaluation

This diagnostic evaluates the EMA from global step 30,523 (20,000 Stage-C
updates after the Stage-B step-10,523 initialization). The checkpoint SHA-256
is `a0b813ab0f88664ebb49da54c34f9cc60b98776dcc3b4814a7c83887538aef00`.
The evaluation ran on a separate RTX 4090 and did not alter model or optimizer
state.

The physical panel is the complete held-out MatPES calibration split: 36,990
graphs for energy, force and stress and 19,101 PBE graphs for the frozen
per-atom teacher feature. The generation panel reuses the fixed A1-v1.1
512-reference/512-generated protocol. This is a diagnostic retention
comparison, not a new Gate or an early-stop rule.

| Metric | Stage-B | Stage-C 10k | Stage-C 20k |
| --- | ---: | ---: | ---: |
| Equal-head physical composite | 0.592913 | 0.387088 | 0.325350 |
| Normalized energy RMSE | 0.114310 | 0.095521 | 0.085950 |
| Normalized force RMSE | 0.388189 | 0.331799 | 0.305305 |
| Normalized Kelvin-stress RMSE | 0.573348 | 0.430090 | 0.388771 |
| PBE teacher-feature cosine | 0.899572 | 0.917104 | 0.926391 |
| Normalized nearest-neighbour W1 | 0.544374 | 0.553292 | 0.562817 |
| Normalized volume W1 | 0.072235 | 0.062390 | 0.067634 |

From 10k to 20k, the physical composite improves by another 15.95%; normalized
energy, force and stress RMSE improve by 10.02%, 7.99% and 9.61%, respectively.
Teacher-feature cosine increases by 0.009287. The generative-retention trend is
more mixed: nearest-neighbour W1 increases by 0.009525 and volume W1 increases
by 0.005244 from its 10k value, although volume W1 remains better than Stage-B.

All 512 generated structures retain exact composition and finite positive
lattices. Their minimum-distance validity at 0.5 Angstrom and formula
uniqueness are both 1.0; terminal masks and sampling failures are zero. There
is therefore no catastrophic forgetting, but the monotone
`0.544374 -> 0.553292 -> 0.562817` nearest-neighbour trend should be watched at
later checkpoints as a physical-transfer versus generative-retention
trade-off.

The first training launcher did not retain stderr, so its stop after finite
metrics through Stage-C step 24,277 was initially unclassified. An exact resume
then reproduced the cause: LeMat material `oqmd-2964825` declares `nsites=8`
but stores 15 Cartesian positions and 15 species. The fail-closed parser
correctly rejected it. The v3 index had inspected lattice and count metadata
but not the nested position/species lengths, so its qualification was
incomplete. Training remains stopped at the complete step-20,000 checkpoint
while a geometry-complete clean index is built. No runtime parser fallback is
introduced. Future long-run protocols should checkpoint every 5,000 steps to
bound work lost to an interruption.

The machine-readable evidence is in `result.json`, produced by
`scripts/evaluate_stage_c_checkpoint.py`.
