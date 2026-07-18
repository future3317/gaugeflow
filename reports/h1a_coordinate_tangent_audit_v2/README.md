# H1a coordinate tangent audit v2

Status: **completed; strict quotient rank is full, but the tangent is severely
anisotropic. H1a remains failed.**

The corrected audit accumulated the FP32 parameter Jacobian Gram matrix in
FP64.  The three translation modes have eigenvalues
`3.26e-11/4.50e-11/6.11e-11`; all 30 physical quotient directions are active.
The current vector field is therefore not mathematically rank deficient on the
fixed generic 11-site state.

It is nevertheless poorly preconditioned.  The active eigenvalues span
`2.35e-6` to `54.50`, giving condition number `2.315e7`, while the entropy
effective rank is only `2.27`.  The coordinate-edge-head gradient norm is
`0.19391`; coordinate-vector-head and vector-control-gate norms are only
`0.001283` and `0.000562`.  A direct hook confirms why: final latent vectors
have RMS `5.80e-4`, target score RMS `0.8629`, and full initial prediction RMS
only `0.00957`.

The target projection residual `2.61e-5` misses the frozen `1e-6` threshold,
but strict full rank means this is not an unreachable mathematical direction.
It is dominated by finite-precision translation-horizontal projection and the
same tiny tangent eigenvalues.  The result supports one capacity-neutral,
O(3)-invariant vector RMS preconditioner with a fixed floor; it does not support
another reciprocal/Fourier branch or a larger model.

The successor must first improve the no-training tangent spectrum, vector-to-
edge gradient balance, covariance tests and CUDA cost.  Only then may it repeat
the one-state 1,024-step test.  H1b and all tensor/later work remain stopped.
