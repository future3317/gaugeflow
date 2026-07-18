# H1a Cartesian moment score head v1

Status: **failed before training and removed from production. H1a remains
failed.**

The vectorized rank-one/rank-two construction is mathematically and
computationally sound.  Its direct full-O(3) covariance error is `1.79e-7`,
node-permutation error `2.47e-7`, full-model translation error `2.64e-7`, and
forward/backward values are finite.  On the RTX 4060 Ti it achieved
`446.21 graphs/s` with `1586.43 MiB` peak allocated memory, passing both frozen
efficiency limits.

It did not change the diagnosed tangent.  Condition number remained
`2.309e7`, effective rank `2.2696`, and the moment/central-head gradient norm
ratio was only `5.88e-4`.  The multiplicative `Q m` construction is itself too
weak at initialization: adding correct Cartesian directions is not useful when
their parameter tangent is effectively disconnected.  The frozen rule forbids
rescaling or training the failed operator, so no one-state experiment ran.

The implementation is removed from the active tree; exact code remains at
commit `127bcad`.  No flag, compatibility loader or runtime fallback is kept.
Any successor must be a first-order, variance-controlled Cartesian readout or
an optimization formulation supported by a separate audit, not a larger
unqualified moment stack.
