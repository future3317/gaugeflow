# H1a scaled variable-projection stability v1

Status: **failed before training; scaled variable projection is rejected. H1a
remains failed.**

This audit asks whether the only remaining causally supported combination is
numerically safe before spending a 1,024-step memorization budget. It uses the
same first 16 fixed states, seed, noise, model capacity, path and affine
coordinate head as the failed unscaled variable-projection experiment. The
only candidate operation is the already-qualified power-of-two `1024x`
function-preserving readout parameterization.

The exact head is solved once in graph-equal FP64 least squares. The audit then
compares FP32 and CUDA BF16 prediction and nonlinear-backbone gradients without
an optimizer step. This distinction is necessary because scaling the stored
head parameters does not change the effective function weights seen by the
backbone. A small stored solution norm is insufficient if cancellation,
forward loss or gradients remain unstable.

The algebraic pieces pass. The power-of-two chart preserves the initialized
function within `5.96e-7`, the captured design reconstructs production within
`2.98e-7`, rank is `225/225`, and the scaled solution norm is `8.894`. FP32
execution also reproduces the exact panel fit at MSE `0.099467` with a finite
backbone gradient norm of `3.889`.

BF16 execution fails decisively. Coordinate MSE is `10.9886`, or `110.47x` the
FP32 value; BF16/FP32 prediction relative RMS is `4.4899`. The backbone gradient
norm is `23,468.3`, `6,033.9x` FP32, and its cosine with the FP32 gradient is
`-0.1572`. The solved vector and edge contributions have norms `272.59` and
`271.00` while their sum has norm only `16.83`, a `32.31x` component-to-total
cancellation ratio. Scaling changes the stored solution norm but leaves an
effective unscaled norm of `9,107.83`; it cannot repair this cancellation.

No optimizer step was executed, all model parameters were restored exactly,
and no production file was changed. The frozen decision therefore rejects
scaled variable projection before training. It cannot be rescued by searching
scale, ridge, solve frequency, steps, seeds, or silently switching the failed
protocol precision. A successor must first qualify a compact equivariant
basis-decorrelation mechanism. H1b--H6, tensor conditioning, oracle work,
relaxation, DFT and DFPT remain prohibited. Exact values are in `result.json`;
the frozen runner, tests and protocol remain recoverable from commit `231126d`.
