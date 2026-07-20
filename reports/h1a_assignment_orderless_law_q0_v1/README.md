# Orderless remaining-count assignment Q0

Decision: **PASS**.

| metric | value |
|---|---:|
| complete distribution normalization error | 4.441e-16 |
| subset-DP vs order brute force | 1.041e-17 |
| sample exact-count fraction | 1.000000 |
| relabel marginal error | 0.000e+00 |
| FP64 / FP32 neural equivariance | 1.110e-15 / 4.768e-07 |
| residual-stabilizer error | 2.980e-07 |
| BF16 output cosine | 0.999974 |
| RTX 4090 forward latency / peak memory | 5.067 ms / 99.048 MiB |

This is a mathematical/software qualification only. It performs no learning
and does not qualify assignment or connect generated composition.

The reported CUDA memory is explicitly a no-grad forward measurement. An
initial software attempt retained simultaneous FP32 and BF16 autograd graphs
and therefore measured 720.057 MiB; that attempt is preserved in
`attempt_1_autograd_measurement_result.json`. The frozen 512 MiB acceptance
threshold was not changed.
