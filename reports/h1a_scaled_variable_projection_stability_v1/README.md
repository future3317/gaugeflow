# H1a scaled variable-projection stability v1

Status: **frozen before run; no training has been performed.**

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

All thresholds and the pass/fail boundary are frozen in
`configs/gates/h1a_scaled_variable_projection_stability_v1.json`. A failure
rejects the combination before training; it cannot be rescued by searching the
scale, ridge, solve frequency, steps or seeds. H1a remains failed and H1b--H6,
tensor conditioning, oracle work, relaxation, DFT and DFPT remain prohibited.
