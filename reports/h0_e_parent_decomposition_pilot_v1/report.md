# H0-E-v1 parent-decomposition pilot

## Decision

`H0-E-v1_failed_stop_before_H1`.

The frozen 1,024-row Alex-MP-20 pilot failed exactly one preregistered
threshold: 125 rows exposed a nontrivial parent candidate, giving
`125 / 1024 = 0.1220703125`, below the required `0.15`. H1a/H1b and H2--H6
remain unauthorized. No tensor, oracle, relaxation, DFT, or DFPT task ran.

The independent audit passed and reproduced all 32 rows in its deterministic
rebuild panel. It therefore verifies the negative gate result rather than
reclassifying it as a software failure.

## Recomputed results

| Metric | Result | Frozen requirement | Pass |
|---|---:|---:|:---:|
| Selected rows | 1,024 | 1,024 | yes |
| Nontrivial-parent fraction | 0.1220703125 | >= 0.15 | **no** |
| Qualified/candidate fraction | 1.0 (125/125) | >= 0.90 | yes |
| Maximum periodic RMS (Angstrom) | 3.2388e-8 | <= 0.10 | yes |
| p95 periodic RMS (Angstrom) | 1.3715e-8 | <= 0.075 | yes |
| Median sector-normalized top-2 energy | 0.90811 | >= 0.80 | yes |
| Terminal space-group agreement | 1.0 | >= 0.90 | yes |
| Occurrence integrality | 1.0 | 1.0 | yes |
| Processing failures / nonfinite results | 0 / 0 | 0 / 0 | yes |
| Distinct parent / child space groups | 34 / 25 | >= 10 / >= 20 | yes |

The realized sector combinations were: 98 displacement-plus-strain, 13
strain-only, 8 displacement-only, 4 two-displacement, and 2 two-strain. Thus
113/125 qualified paths require a strain OPD; a displacement-only occurrence
test would be physically incomplete.

## Exact representation and acceleration

Concrete occurrence is evaluated in

```text
compact atomic displacement representation direct-sum Kelvin strain representation.
```

The displacement action is a node permutation plus a Cartesian `3 x 3`
rotation; no dense `3N x 3N` matrix is formed. Homogeneous strain is represented
by the six coefficients of the symmetric Hencky strain
`E = 0.5 log(A A^T)` in an orthonormal Kelvin basis. This is an isometry of the
symmetric-tensor Frobenius inner product, not an approximation. One- and
two-component subsets use the exact intersection of their affine stabilizers,
and the residual is Reynolds-projected only with that declared subgroup.

The exact periodic closest-vector solver now batches all atoms sharing a cell
and reuses one float64 QR factorization. On a fixed 64-vector skew-cell CPU
microbenchmark, 40 repetitions took 0.0787 s versus 0.2642 s for the equivalent
single-vector loop (3.36x speedup). OPD enumeration is cached per real irrep and
component group orbits are reused across all one/two-component subsets.

## Artifact integrity

- Decomposition Parquet SHA-256:
  `4f3c22ef084a15e82ac0cdd5c8a5c26c3083af4a65a59a789e1f03946e205358`
- Selection Parquet SHA-256:
  `9575768b609f758e70f94148720094b7cb9d5c13c12c455e78669b93b7d9e3b0`
- Required H0-D-v2 records SHA-256:
  `ba711a63874444cfc5ca1c9d5603cdcad65ae63799f44fd209aff979e73c940c`
- External artifact root:
  `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1`

The first post-run audit found that PyArrow had inferred a sparse schema from
the first unqualified dictionary and omitted later qualified-only columns. The
writer was corrected to construct the union of all record keys, a regression
test was added, and the unchanged frozen protocol was rerun. The hash above is
for the corrected, independently auditable artifact; the in-memory gate metrics
were identical before and after the serialization repair.

See `independent_audit.json` for the fail-closed recomputation.
