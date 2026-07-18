# H1a coordinate branch minimality v1

Status: **frozen before run; no training has been performed.**

The preceding FP32/BF16 audit measured `32.31x` cancellation between the final
vector and edge coordinate fields. This protocol tests the simplest causal
repair: remove a redundant branch rather than add another basis, loss, or
fallback. It uses the same first 16 fixed states, model initialization, noise,
path and exact graph-equal affine solve.

Vector-only, edge-only and combined bases are evaluated from the same captured
production features. A single branch may proceed only if it retains full
one-state quotient rank, target projection, 16-state FP32 fit, low-time endpoint
accuracy, moderate solution norm, and stable BF16 prediction and backbone
gradient. If both single branches pass, the smaller vector branch is selected;
if neither passes, production remains unchanged.

Thresholds and decisions are frozen in
`configs/gates/h1a_coordinate_branch_minimality_v1.json`. The protocol performs
zero optimizer steps and cannot be rescued by changing precision, thresholds,
states, seeds or training budget. H1a remains failed; H1b--H6, tensor
conditioning, oracle work, relaxation, DFT and DFPT remain prohibited.
