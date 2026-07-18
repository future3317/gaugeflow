# H1a compact Cartesian Krylov production integration v1

Status: **failed before training; the integration is removed from active
production. H1a remains failed.**

The target-free operator qualification passed for an 80-channel Cartesian
carrier. This protocol now requires a single clean production implementation:
the old `coordinate_vector_head` and `coordinate_edge_head` are removed, not
retained as dispatch or checkpoint fallbacks. The existing scalar edge encoder
feeds one learned 32-output projection, split into 16 first-moment and 16 STF
second-moment coefficients. A single 80-channel readout consumes the existing
vector stream and `(m,Qm,Q^2m)` carriers.

All moment reductions and the final coordinate carrier/readout accumulate in
FP32 under BF16 backbone autocast. This is a typed geometry reduction, not an
FP32-only model or a selectable precision fallback. The integration must have
exactly 4,479,161 parameters, no legacy readout keys, no coordinate-target
input, finite FP32/BF16 output and gradients, at least 200 graphs/s on the fixed
RTX 4060 Ti batch, and at most 2.5 GiB peak allocation.

Passing authorizes only a separately frozen single-seed fixed-state
memorization experiment. It does not qualify H1a or permit later Gates.

## Result

The clean integration satisfies its structural and runtime contracts. It has
exactly `4,479,161` parameters and 80 carrier channels, contains zero legacy
readout keys, and bypasses all tensor candidates. FP32/BF16 coordinate-output
relative RMS is `0.08780` with cosine `0.99623`; gradient norm ratio is
`1.09185` and gradient cosine `0.98573`. A 64-graph RTX 4060 Ti forward reaches
`1066.85 graphs/s` at `506.24 MiB` peak allocation.

It nevertheless fails the frozen absolute-gradient bound. The target-free
production output-energy gradient norms are `373.27` in FP32 and `407.55` in
BF16, both above `100`. This is not a mixed-precision directional failure; it
is excessive absolute Jacobian scale introduced when the normalized carrier is
coupled to the actual fractional coordinate output. No coordinate target or
optimizer step was used.

Per the preregistered decision, the integration is removed from the active tree
and remains recoverable at commit `e25f432`. The qualified standalone operator
result remains valid, but it does not authorize memorization. A new read-only
diagnostic must attribute gradient scale by carrier order and parameter group
before any revised integration is proposed.
