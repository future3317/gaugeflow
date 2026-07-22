# Stage-C 30k mid-training evaluation

This diagnostic evaluates the EMA from global step 40,523 (30,000 Stage-C
updates after the Stage-B step-10,523 initialization). The checkpoint SHA-256
is `8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9`.
The evaluation ran on a separate RTX 4090 and did not alter model, optimizer,
or training-stream state.

The physical panel is the complete held-out MatPES calibration split: 36,990
graphs for energy, force and stress and 19,101 PBE graphs for the frozen
per-atom teacher feature. The generation panel reuses the fixed A1-v1.1
512-reference/512-generated protocol. This is a retention diagnostic, not a
new Gate or an early-stop rule.

| Metric | Stage-B | Stage-C 10k | Stage-C 20k | Stage-C 30k |
| --- | ---: | ---: | ---: | ---: |
| Equal-head physical composite | 0.592913 | 0.387088 | 0.325350 | **0.290835** |
| Normalized energy RMSE | 0.114310 | 0.095521 | 0.085950 | **0.080053** |
| Normalized force RMSE | 0.388189 | 0.331799 | 0.305305 | **0.289821** |
| Normalized Kelvin-stress RMSE | 0.573348 | 0.430090 | 0.388771 | **0.364273** |
| PBE teacher-feature cosine | 0.899572 | 0.917104 | 0.926391 | **0.932265** |
| Normalized nearest-neighbour W1 | **0.544374** | 0.553292 | 0.562817 | 0.565613 |
| Normalized volume W1 | 0.072235 | **0.062390** | 0.067634 | 0.068019 |

From 20k to 30k, the physical composite improves by another 10.61%; normalized
energy, force and stress RMSE improve by 6.86%, 5.07% and 6.30%, respectively.
Teacher-feature cosine increases by 0.005874. Relative to Stage-B, the physical
composite is 49.05% of its incoming value.

The generative-retention trend remains mixed. Nearest-neighbour W1 increases by
only 0.002796 from 20k but is 0.021239 above Stage-B; volume W1 changes by
+0.000385 from 20k and remains 0.004216 below Stage-B. All 512 generated
structures retain exact composition, finite positive lattices, minimum-distance
validity at 0.5 Angstrom, and formula uniqueness. Terminal masks and sampling
failures are zero. This is not catastrophic forgetting, but it is direct
evidence of a physical-transfer versus local-geometry-retention trade-off.

The machine-readable evidence is `result.json` (SHA-256
`67a6c7f3f16482a00bae419e81192117064cc8f7d1ceef2e08f4b1a6346ec8d3`).
