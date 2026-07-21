# Pair-context assignment fixed-batch overfit v2

Status: **failed**.

The target-free symmetric two-point RBF Gram is mathematically sufficient to
separate the three archived hard collision classes, but merely appending it to
the old mean-message edge feature did not improve learning. Aggregate
target-quotient probability lower bound was `0.65402`, retrieval was `0.65234`,
and aligned site accuracy was `0.85539`; exact composition and finite gradients
remained `1.0`.

The pair-context feature itself is not the failure: zero-training enumeration
shows that it reduces each archived collision class exactly to the target
parent orbit.  A later supervision audit localized the remaining defect to the
use of one fixed CIF representative.  Parent automorphisms then present an
equivariant scorer with mutually inconsistent path labels.  The next
separately frozen screen keeps this pair context and samples uniformly from the
deduplicated target parent orbit before independently sampling the reveal
order:

```text
A_rep ~ Uniform(unique(G_parent . A_target)),
Z ~ Uniform(site reveal orders).
```

The v2 failure does not authorize the IID assignment Gate or any downstream
generation, tensor, relaxation, DFT or DFPT stage.
