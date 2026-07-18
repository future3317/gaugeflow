# H1a factorized Cartesian angular moments v1

Status: **zero-training CUDA qualification passed; learning was not part of
this protocol.**

This mechanism augments the unchanged `192/32/4`, 16-RBF coordinate backbone
with a 64-dimensional persistent scalar edge state and eight channels of
first/second Cartesian angular moments.  It constructs no explicit triplet
index.  Expanding the contractions gives the same low-order angular sum as an
explicit `(i -> j, k -> j)` reference, while the implementation uses only
edge-leading products and target-contiguous node segment reductions.

The production model adds 466,944 parameters and has 4,948,281 parameters in
total.  Every new output projection is initialized to exact zero, so the
initial function equals the preceding coordinate model without a legacy
dispatch or checkpoint fallback.  Once those projections are activated, the
internal angular coefficient path has gradient norm `0.08217`.

On an idle RTX 4060 Ti 16 GB with PyTorch `2.5.1+cu124`, the fixed 16-graph
mixed-BF16/fixed-FP32 probe obtained:

| check | result | frozen bound | status |
|---|---:|---:|:---:|
| forward throughput | 320.84 graphs/s | >= 300 | pass |
| peak allocated memory | 182.86 MiB | <= 1,536 MiB | pass |
| BF16/FP32 output relative RMS | 0.01430 | <= 0.05 | pass |
| BF16/FP32 output cosine | 0.999898 | >= 0.99 | pass |
| BF16/FP32 gradient ratio | 1.00463 | 0.5--2.0 | pass |
| BF16/FP32 gradient cosine | 0.999363 | >= 0.9 | pass |
| tensor-free atlas candidates | 0 | 0 | pass |

The isolated CPU suite additionally matches the explicit low-order triplet
sum in FP64 and passes reflection-inclusive O(3), node/edge permutation,
translation, GL(3,Z), finite-gradient and full-denoiser tests.  This result
authorizes exactly one separately frozen, seed-5705, one-pass coordinate-only
learning experiment.  It does not qualify H1a generation, tensor conditioning
or later Gates.
