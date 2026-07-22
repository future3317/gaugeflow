# Stage-C 10k mid-training evaluation

This diagnostic evaluates the EMA from global step 20,523 (10,000 Stage-C
updates after the Stage-B step-10,523 initialization). Training continued
unchanged while the evaluation ran on a separate RTX 4090. The checkpoint
SHA-256 is
`2b90b655902f4cadb8fbfd79180548bba561b68b6d320710185eebb881bb4ab7`.

The physical panel is the complete held-out MatPES calibration split used by
Stage-B: 36,990 graphs for energy, force and stress and 19,101 PBE graphs for
the frozen per-atom teacher feature. The generation panel reuses the fixed
A1-v1.1 512-reference/512-generated protocol. It is a diagnostic retention
comparison, not a new Gate or an early-stop decision.

| Metric | Stage-B | Stage-C 10k | Change |
| --- | ---: | ---: | ---: |
| Equal-head physical composite | 0.592913 | 0.387088 | -34.71% |
| Normalized energy RMSE | 0.114310 | 0.095521 | -16.44% |
| Normalized force RMSE | 0.388189 | 0.331799 | -14.53% |
| Normalized Kelvin-stress RMSE | 0.573348 | 0.430090 | -24.99% |
| PBE teacher-feature cosine | 0.899572 | 0.917104 | +0.017532 |
| Normalized nearest-neighbour W1 | 0.544374 | 0.553292 | +0.008918 |
| Normalized volume W1 | 0.072235 | 0.062390 | -0.009845 |

All 512 generated structures retain exact composition and finite positive
lattices. Their minimum-distance validity at 0.5 Angstrom and formula
uniqueness are both 1.0; terminal masks and sampling failures are zero. Element
and node-count JSD are unchanged because those variables are sampled by the
same qualified discrete laws. The small nearest-neighbour W1 increase is a
1.64% relative fluctuation, while volume W1 improves by 13.63%. At this
checkpoint, Stage-C therefore provides material physical-representation gains
without evidence of catastrophic forgetting of the A1 generative law. The
declared 50,000-step continuation remains in progress and later checkpoints
must be evaluated independently.

The machine-readable evidence is in `result.json`. It was produced by
`scripts/evaluate_stage_c_checkpoint.py`; the evaluator commit is `39eee66`.
