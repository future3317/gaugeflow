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
parent orbit.  A later optimization audit found that the one-path estimator
has too much reveal-order variance on the symmetric collision carriers.  The
same model reaches unit retrieval when the order expectation is enumerated.
The next screen therefore keeps this pair context and analytically averages
over every legal next site at each sampled prefix:

```text
sum_d E_{S_d} mean_{i not in S_d} log p(A_i | A_{S_d}, C, carrier).
```

Eight independently sampled prefix paths per carrier are packed into one
vectorized scorer call.  On the archived four-site hard carrier this estimator
reached `0.93138` quotient-probability lower bound, `0.99805` retrieval and
`0.99902` aligned site accuracy after 500 diagnostic steps; a single path
remained at the count-uniform solution.

The v2 failure does not authorize the IID assignment Gate or any downstream
generation, tensor, relaxation, DFT or DFPT stage.
