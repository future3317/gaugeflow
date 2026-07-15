# Post-run analysis: substrate-v2 decoration-only v2

The v2 report is a valid failed qualification.  Its vector-invariant scorer
reached exact proper-SO(3) quotient MAP accuracy, exact-count sampling and
species-aware periodic `StructureMatcher` rate of 1.0 for all six endpoint and
seed combinations.  It nevertheless failed the unchanged `2e-6` complete
assignment probability-vector node-relabeling threshold, with errors from
0.0276 to 3.0.

This was not converted to a pass.  The next versioned protocol, v3, retains
the panel, seeds, budgets, ablations and all thresholds, but makes two
numerical requirements explicit: float64 accumulation for the small
neighbourhood mean and a fixed bounded categorical score `20*tanh(raw/20)`.
Both prevent an arbitrary input row ordering from being amplified after the
exact categorical NLL has saturated.  v3 also records the raw scorer-logit
permutation error alongside the probability-law error.
