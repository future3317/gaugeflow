# Gate A11-Q0 exact assignment and residual-group audit

This is a read-only, untrained mathematical qualification for the A11-Q finite assignment law. It does not start Q1, Q2, A11-G, tensor conditioning, a full benchmark, relaxation, DFT, or DFPT.

## Contract verified

For each endpoint, Q0 enumerates the unique count-constrained chemical labelings, not permutations of artificial same-species slots. With the observed 2+2 endpoint composition, this is exactly six assignments. The neutral all-zero score probe makes this a uniform categorical law; its quoted masses are checks of group marginalization, not learned endpoint-ID accuracy.

Production quotient calculations use only `proper_so3`. `full_o3_scalar` is printed alongside it only as the A11.0 O(3)-scalar decoder diagnostic. It is not an allowed production quotient: improper operations cannot be silently removed for a rank-three polar tensor condition.

At each partial state the quotient group is recomputed as `Gamma_t = {gamma: gamma y_t = y_t}`. The all-mask state retains the geometry group; revealed species/mask tokens can only retain operations compatible with that actual current state.

## Exact-law results

| material | support | fixed-CIF p (diagnostic) | proper quotient p | full-O(3) quotient p (diagnostic) | proper quotient NLL | full quotient NLL | entropy | samples with wrong count | Q0 checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| JVASP-1180 | 6 | 0.166667 | 0.333333 | 0.333333 | 1.098612 | 1.098612 | 1.791759 | 0 | passed_exact_law_checks |
| JVASP-22673 | 6 | 0.166667 | 0.333333 | 0.333333 | 1.098612 | 1.098612 | 1.791759 | 0 | passed_exact_law_checks |

## Residual groups

- `JVASP-1180` / `proper_so3_production` / `all_mask`: |Aut(X)|=4, |Gamma_t|=4.
- `JVASP-1180` / `proper_so3_production` / `one_revealed_species`: |Aut(X)|=4, |Gamma_t|=1.
- `JVASP-1180` / `proper_so3_production` / `fully_revealed_species`: |Aut(X)|=4, |Gamma_t|=2.
- `JVASP-1180` / `full_o3_diagnostic` / `all_mask`: |Aut(X)|=4, |Gamma_t|=4.
- `JVASP-1180` / `full_o3_diagnostic` / `one_revealed_species`: |Aut(X)|=4, |Gamma_t|=1.
- `JVASP-1180` / `full_o3_diagnostic` / `fully_revealed_species`: |Aut(X)|=4, |Gamma_t|=2.
- `JVASP-22673` / `proper_so3_production` / `all_mask`: |Aut(X)|=2, |Gamma_t|=2.
- `JVASP-22673` / `proper_so3_production` / `one_revealed_species`: |Aut(X)|=2, |Gamma_t|=1.
- `JVASP-22673` / `proper_so3_production` / `fully_revealed_species`: |Aut(X)|=2, |Gamma_t|=1.
- `JVASP-22673` / `full_o3_diagnostic` / `all_mask`: |Aut(X)|=2, |Gamma_t|=2.
- `JVASP-22673` / `full_o3_diagnostic` / `one_revealed_species`: |Aut(X)|=2, |Gamma_t|=1.
- `JVASP-22673` / `full_o3_diagnostic` / `fully_revealed_species`: |Aut(X)|=2, |Gamma_t|=1.

## Relabeling consistency

- `JVASP-1180`: maximum FP32 log-probability-vector error = `0.000e+00` against the pre-registered `2.0e-06` threshold; pass=True.
- `JVASP-22673`: maximum FP32 log-probability-vector error = `0.000e+00` against the pre-registered `2.0e-06` threshold; pass=True.

## Decision

Q0 passes only the exact-enumeration and group-action implementation checks. It does not produce a learned composition, exact-assignment, StructureMatcher, terminal-mask, or sampling-validity result. Q1 remains **not started**. If Q1 is ever authorized and passes, Q2 must first test materials with distinct proper-SO(3) orbit structures before any tensor condition is restored.
