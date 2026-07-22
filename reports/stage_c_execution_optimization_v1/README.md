# Stage-C execution optimization

This bounded engineering cycle changed no model, loss weight, dataset
distribution, threshold, or learning protocol.  It qualified three execution
changes before the long LeMat continuation:

1. LeMat rows are converted once per Arrow row group and MatPES JSONL reads are
   ordered source-locally.  The first and last batch digests remained exact.
   On 100 local batches this reduced host materialization from 41.04 s to
   9.44 s (4.35x).
2. Gradients are reduced in bounded flat buckets.  Every rank owns the same
   AdamW and EMA state and applies the globally summed gradient locally, so the
   old full-model broadcast is absent.  On two GPUs the synchronization/update
   microbenchmark decreased from 92.24 to 57.34 ms/step (37.8%); rank replicas
   and the two-strategy final model digest were exact.
3. The additive Stage-C objective is evaluated by three fixed roles: LeMat
   structure, MatPES physical transfer, and Alex generative replay.  One GPU
   computes each weighted objective term and the gradients are summed once.
   The tested loss is still exactly
   `0.4 L_LeMat + 0.3 L_MatPES + 0.3 L_Alex`.

Real three-stream profiles with 64 graphs per stream gave:

| execution | graphs/s | critical step | decision |
|---|---:|---:|---|
| legacy two-GPU data parallel | 138.19 | 1.389 s | replaced |
| optimized two-GPU data parallel | 147.00 | 1.306 s | valid fallback-free diagnostic |
| optimized three-GPU data parallel | 87.01 | 2.207 s | rejected |
| three-role stream parallel | 170.31 | 1.127 s | production choice |

Thus the production choice is 23.2% faster than the old runner and 15.9%
faster than optimized two-GPU data parallel.  Its per-rank peak allocations
were 12.18, 11.08, and 14.68 GiB on 24-GiB RTX 4090 cards.  All three final
parameter digests agreed.  A separate unit test verifies that the sum of the
three role gradients equals the original joint-objective gradient.

The profile also found malformed optional LeMat stress fields.  Since the
LeMat Stage-C branch is geometry-only and MatPES is the sole physical-label
stream, dropping otherwise valid structures would be incorrect.  The runtime
interface now parses only LeMat species, coordinates, and lattice and masks all
LeMat physical targets.  This both removes the irrelevant-data failure and
prevents unused labels from entering a generative replay batch.

The four-step interrupted/resumed CUDA smoke is exact: all 2,245 tensor leaves
and 696 scalar leaves in model, AdamW, EMA, per-role stream cursors, and RNG
state match the uninterrupted checkpoint, with zero mismatches.  These files
are execution evidence, not Stage-C learning or generation qualification.  The
long continuation still requires a separately frozen training protocol.
