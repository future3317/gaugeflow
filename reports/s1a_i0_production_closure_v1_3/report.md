# S1a-I0 production trainer/reverse-sampler closure

## Decision

`s1a_tensor_free_production_v1_3` passed its frozen implementation-closure
gate on the WSL CUDA environment. This qualifies the software path, not the
scientific S1a generator on a real train/validation split.

## What changed

The element process remains the exact finite-step absorbing-mask bridge. The
coordinate process remains a Cartesian-isotropic wrapped Brownian process on
the periodic translation quotient, trained with a metric-correct fractional
score. The lattice process remains cosine VP, but the denoiser now predicts the
clean log volume and projected trace-free log shape. The finite-step VP
posterior consumes those clean-state predictions directly.

This removes the ill-conditioned high-noise reconstruction
`(x_t + sigma_t^2 score) / alpha_t`. It does not clip log shape, add Cholesky
jitter, lower the terminal noise, or restore an old ODE fallback.

## Frozen history

- v1 used uniform-time reverse nodes and failed during lattice Cholesky.
- v1.1 used uniform log-alpha nodes. Fixed loss decreased, but the raw lattice
  score remained insufficiently calibrated and the rollout failed closed.
- v1.2 allowed 3,000 bounded memorization steps with the same method. Fixed
  loss worsened and the rollout again failed closed, ruling out a simple
  training-budget explanation.
- A runner/config optimizer mismatch affecting v1--v1.2 is documented in
  `reports/s1a_i0_production_closure_v1_1/protocol_runner_mismatch.md`; the
  failure conclusions remain valid, while v1.3 reads all values from JSON.

## v1.3 result

| Metric | Result | Gate |
|---|---:|---:|
| CUDA training steps | 300 | 300 |
| Initial fixed loss | 19.003906 | -- |
| Final fixed loss | 1.540639 | -- |
| Fixed loss ratio | 0.081070 | <= 0.95 |
| Sampling failures | 0 | 0 |
| Terminal MASK count | 0 | 0 |
| Device | NVIDIA GeForce RTX 4060 Ti | CUDA required |
| Torch | 2.5.1+cu124 | pinned environment |

The exact machine-readable result is `results.json`. A deterministic rerun in
`gradient_diagnostic.json` records finite, nonzero terminal gradient norms for
the element, coordinate, volume, and shape heads: 0.7204, 0.6885, 0.03173, and
0.001240, respectively.

## Remaining boundary

The production trainer, EMA/checkpoint recovery, empirical training-split node
count prior, P1 blueprint, CIF writer, and joint reverse sampler now exist. A
real-data S1a run, decoded validity/uniqueness/novelty evaluation, and the full
230-space-group/Wyckoff blueprint sampler have not been qualified. Tensor
conditioning, oracle promotion, relaxation, DFT, and DFPT remain disabled.
