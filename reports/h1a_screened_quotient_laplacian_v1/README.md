# H1a screened quotient-Laplacian preconditioner v1

Status: **failed before training and removed from production. H1a remains
failed.**

The candidate applied a vectorized screened inverse of the normalized periodic
graph Laplacian to the final Cartesian zero-mean score.  It added no parameters
or hidden edge-channel expansion and passed quotient rank, target projection,
affine fit, O(3), permutation and full-model translation checks.

It is computationally clean: RTX 4060 Ti coordinate training throughput was
`483.23 graphs/s` with `1618.85 MiB` peak allocation, and translation error was
`2.06e-7`.  But it did not address the diagnosed tangent.  Condition number
remained `3.455e7`, effective rank `2.24`, and required readout step norm rose
to `2484.92`; the frozen limits were `5e6`, `4.0` and `100`.

Thus the weak directions are not explained by simple graph-Laplacian spatial
modes.  The operator was not trained and is retained only in Git history and
this report.  The next mechanism should target the final affine parameter scale
directly while preserving the initial function and avoiding state-dependent
normalization.

Exact values are in `result.json`.
