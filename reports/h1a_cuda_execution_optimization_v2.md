# H1a CUDA execution optimization v2

## Decision

**Passed.** The production CUDA trainer now uses TF32 tensor-core execution
for eligible FP32 matrix multiplications, fused AdamW, and materializes the
gradient-norm scalar only at the existing synchronized logging boundary. The
model, coordinate objective, probability path, data order, batch size, seed,
checkpoint state, and exposure are unchanged.

The earlier fused-AdamW-only pilot did not qualify: its observed speedup was
only about 3%, below the frozen 10% minimum. Reusing segment lengths across
the deterministic reductions was also rejected in pilot because it did not
improve end-to-end throughput. Neither rejected change is represented as an
independent scientific improvement.

## Real-batch numerical equivalence

The archived two-pass checkpoint and the same first Alex-MP-20 batch/noise
stream were used for both backends.

| Metric | Observed | Frozen requirement |
|---|---:|---:|
| selected coordinate-loss relative difference | 0.00036985 | <= 0.0005 |
| coordinate prediction cosine | 0.99999756 | >= 0.999 |
| full-gradient cosine | 0.99997008 | >= 0.997 |
| one-step parameter-update cosine | 0.99999272 | >= 0.999 |
| parameter-update relative norm difference | 0.00021252 | <= 0.01 |
| finite loss/gradient/parameters/optimizer | yes | required |

## Alternating RTX 4060 Ti benchmark

Each repeat used batch size 64, 8 warmup steps, 32 synchronized measurement
steps, the same checkpoint, data order, and noise seed. Reference and optimized
runs were alternated.

| Backend | Repeat 1 | Repeat 2 | Repeat 3 | Median graphs/s |
|---|---:|---:|---:|---:|
| FP32-highest + unfused AdamW reference | 203.50 | 224.69 | 224.93 | 224.69 |
| TF32-high + fused AdamW | 262.73 | 262.25 | 258.11 | 262.25 |

The median speedup is **1.1672x (16.72%)**. Peak allocated CUDA memory changes
from 4224.55 MiB to 4222.29 MiB (ratio 0.9995), satisfying the <=1.05 limit.

This is an execution qualification, not a new H1a training result and not a
Gate pass. It does not authorize added exposure, tensor conditioning, H1b-H6,
oracle work, relaxation, DFT, or DFPT.
