# Promotion-metric audit: substrate-v2 decoration-only v3

The original `manifest.json` is retained without modification.  Its aggregate
`not_passed_primary_metrics` status is a runner-selection defect: it required
all three ablations to pass, including the deliberately information-deficient
legacy and RBF-only negative controls.

The v3 protocol's promotion rule applies to the **exact-quotient
geometry-aware scorer**, namely `rbf_vector_invariant_scorer`.  Filtering the
immutable `decoration_results.csv` to that pre-declared candidate gives all six
endpoint/seed rows:

| criterion | threshold | observed |
|---|---:|---:|
| proper-SO(3) quotient MAP accuracy | >= 0.95 | 1.0 for 6/6 |
| species-aware periodic StructureMatcher rate | >= 0.90 | 1.0 for 6/6 |
| exact composition count | required | 1 for 6/6 |
| terminal masks | 0 | 0 for 6/6 |
| sampling failures | 0 | 0 for 6/6 |
| fresh-forward assignment-law relabel error | <= 2e-6 | max 2.22e-16 |

It therefore **qualifies the fixed-geometry, supplied-composition decoder
substrate only**.  It does not demonstrate composition generation, joint
unconditional crystal generation, tensor conditioning, a GaugeFlow pass, or
any DFT/DFPT result.  The negative controls remain useful evidence that RBF
alone is insufficient and that the vector-invariant path is doing real work.

The next authorized experiment is a separately versioned endpoint-ID Q1
qualification in which composition is generated rather than supplied.
