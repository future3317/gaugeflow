# Stage-C 40k mid-training evaluation

This diagnostic evaluates the EMA from global step 50,523 (40,000 Stage-C
updates after the Stage-B step-10,523 initialization). The checkpoint SHA-256
is `cf132a7fc9d7508c3ead7c196d22b292cf7556b83c2b59fc47332adad12ff65f`.
The evaluation ran on a separate RTX 4090 and did not alter model, optimizer,
or training-stream state.

The physical panel is the complete held-out MatPES calibration split: 36,990
graphs for energy, force and stress and 19,101 PBE graphs for the frozen
per-atom teacher feature. The generation panel reuses the fixed A1-v1.1
512-reference/512-generated protocol. This is a retention diagnostic, not a
new Gate or an early-stop rule.

| Metric | Stage-B | Stage-C 10k | Stage-C 20k | Stage-C 30k | Stage-C 40k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal-head physical composite | 0.592913 | 0.387088 | 0.325350 | 0.290835 | **0.265180** |
| Normalized energy RMSE | 0.114310 | 0.095521 | 0.085950 | 0.080053 | **0.075680** |
| Normalized force RMSE | 0.388189 | 0.331799 | 0.305305 | 0.289821 | **0.274797** |
| Normalized Kelvin-stress RMSE | 0.573348 | 0.430090 | 0.388771 | 0.364273 | **0.346883** |
| PBE teacher-feature cosine | 0.899572 | 0.917104 | 0.926391 | 0.932265 | **0.936388** |
| Normalized nearest-neighbour W1 | **0.544374** | 0.553292 | 0.562817 | 0.565613 | 0.578456 |
| Normalized volume W1 | 0.072235 | **0.062390** | 0.067634 | 0.068019 | 0.071122 |

From 30k to 40k, the physical composite improves by another 8.82%; normalized
energy, force and stress RMSE improve by 5.46%, 5.18% and 4.77%, respectively.
Teacher-feature cosine increases by 0.004123. Relative to Stage-B, the physical
composite is 44.72% of its incoming value.

The local-geometry retention cost is now material: nearest-neighbour W1 rises
by 0.012842 from 30k and by 0.034082 from Stage-B. Volume W1 remains 0.001113
below Stage-B. All 512 generated structures retain exact composition, finite
positive lattices, minimum-distance validity at 0.5 Angstrom, and formula
uniqueness; terminal masks and sampling failures are zero. Therefore this is
distributional forgetting rather than a validity or sampler failure. Final
checkpoint selection must use the physical/retention Pareto frontier rather
than defaulting to the last optimizer step.

The machine-readable evidence is `result.json` (SHA-256
`383d26c47af5c48eabd41e82adacaf868e4efdd822ed69766273362d2d73f12b`).
