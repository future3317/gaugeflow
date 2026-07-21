# H1a deterministic assignment reduction v1 — PASS

This no-training CUDA qualification was preregistered before execution and ran
once from commit `1bdea8206a08d8d0575c8a24d843c4a79f5708b5` using the frozen IID-v1
checkpoint.  It verifies the numerical implementation needed before a new
assignment checkpoint may be trained.

All frozen checks pass on an RTX 4090:

- exact repeated forward/path evaluations: bitwise residual `0.0`;
- maximum node-relabel raw-logit residual: `8.94e-6 <= 2e-5`;
- maximum fixed-path relabel residual: `3.81e-6 <= 2e-5`;
- finite-gradient fraction: `1.0`;
- 128-graph forward/backward step: `36.35 ms`;
- peak allocated CUDA memory: `1,523.03 MiB`.

The active implementation uses target-major complete-pair edges, constructs
replicated orderless graphs directly in target-major order, and performs
linear-time `segment_reduce` sums without atomic `index_add_`.  Strict FP32
matmul precision is used for symmetry-critical assignment training and
evaluation.  No model capacity or loss was changed.

This PASS authorizes freezing a successor assignment protocol only.  It does
not qualify assignment likelihood, `p(N)`, lattice, coordinates, joint
generation, tensor conditioning, relaxation, DFT, or DFPT.
