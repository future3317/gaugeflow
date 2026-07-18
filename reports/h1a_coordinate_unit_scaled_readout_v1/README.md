# H1a graphwise unit-scaled coordinate readout v1

Status: **failed before training and removed from production. H1a remains
failed.**

The candidate kept the parameter count fixed and normalized every existing
vector and aggregated edge vector-field channel by an O(3)-invariant graphwise
RMS before the unchanged linear readout.  Operator O(3), permutation, finite
zero-stratum behavior, exact quotient rank, target projection, affine forward
and endpoint fitting all passed.

It produced strong but insufficient scale improvement.  The minimum-norm
readout update fell from `2079.20` to `6.14`, condition number from `3.496e7` to
`6.542e6`, and effective rank rose from `2.23` to `3.49`.  The frozen spectral
limits were `5e6` and `4.0`, so both remain failed.

The explicit `edge x 192 x 3` basis also missed the preregistered systems and
numeric guardrails: RTX 4060 Ti throughput was `392.45 < 400 graphs/s` at
`1767.70 MiB`, and full-model translation error was `3.84e-5 > 1e-5`.  The
normalization amplifies tiny periodic-graph FP32 differences.  Near misses are
not promoted after the run.

The result establishes that coordinate-basis scale is causal, but rejects this
particular expanded-basis implementation.  It was not trained and is retained
only in Git history and this report.  A successor must avoid materializing all
edge channels and improve quotient node-mode conditioning directly.

Exact values are in `result.json`.
