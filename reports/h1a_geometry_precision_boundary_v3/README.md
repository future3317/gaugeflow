# H1a geometry precision boundary v3

The zero-training CUDA qualification passed every frozen numerical and
efficiency check.  Geometry-sensitive message blocks and the coordinate edge
encoder run in one fixed FP32 typed path; terminal scalar heads remain AMP
eligible.  Target-contiguous `torch.segment_reduce` removes CUDA atomic-order
noise without runtime sorting.

Identical-input repeat error was exactly zero.  BF16 translation relative RMSE
was `1.53e-5/2.03e-5` in Cartesian/fractional charts, output cosine against
FP32 was `0.999806`, loss-gradient cosine was `0.997593`, and all tangent,
GL(3,Z), O(3), permutation, gradient, parameter, carrier and atlas-bypass
contracts passed.  RTX 4060 Ti forward throughput was `516.03 graphs/s`, above
the frozen `500` minimum, with `185.73 MiB` peak allocation.

This result authorizes only a separately frozen seed-5705, 8,441-step,
one-complete-train-pass coordinate-only pretraining on all 540,164 training
structures.  It does not qualify H1a or authorize joint generation, H1b-H6,
tensor conditioning, oracle work, relaxation, DFT or DFPT.
