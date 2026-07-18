# H1a latent-vector RMS preconditioner v1

Status: **failed before training and removed from production. H1a remains
failed.**

The capacity-neutral O(3)-invariant RMS factor did what it was designed to do
locally: vector-head/edge-head gradient ratio increased from `0.00662` to
`0.56077`, full quotient rank remained `30/30`, translation error was
`3.43e-7`, direct O(3) covariance error was `4.77e-7`, and the zero-vector
stratum had finite forward/backward values.  Parameter count was unchanged.

It did not repair the tangent geometry.  Condition number remained
`2.156e7 > 5e6`, entropy effective rank was `2.326 < 4`, and quotient target
projection residual was `1.30e-5 > 1e-6`.  It scales the weak vector path but
does not add well-conditioned transverse Cartesian directions.  Consequently
the one-state training authorized only on qualification was not run.

The CUDA benchmark was finite at `460.63 graphs/s`.  Its reported `3449.95 MiB`
includes the retained tangent Jacobian and is not a clean standalone model
memory number; it therefore cannot support a production memory comparison.
The independent tangent failures already reject the operator, so this
measurement caveat does not change the decision.

The implementation is removed from the active source tree.  Exact code remains
at commit `1d707d6`; no runtime flag or fallback is retained.  A future repair
must improve directional span/conditioning, not merely rescale the final latent
vectors.
