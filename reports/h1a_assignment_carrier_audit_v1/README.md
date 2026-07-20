# H1a assignment carrier audit v1

Status: **qualified in its zero-training scope**.

The audit verifies that the qualified H0-E-v4 O1 records can support one
separately frozen oracle-C count-constrained assignment Q1. It does not pass
assignment, `p(N)`, L1/M1, free joint H1a, tensor/oracle work, relaxation, DFT
or DFPT.

## Frozen checks

| quantity | result | frozen requirement |
|---|---:|---:|
| candidate carriers | 454 | 454 |
| train / val / test carriers | 358 / 43 / 53 | 358 / 43 / 53 |
| material split disjoint | true | true |
| maximum atoms | 20 | <=20 |
| maximum species | 5 | <=7 |
| maximum mixed-radix DP states | 1,053 | <=100,000 |
| median uniform target quotient probability | 0.00015873 | <0.5 |
| occupational symmetry-breaking fraction | 1.0 | >=0.1 |

Every carrier exactly reconstructs its target coloring, closes parent site
count times cell index on the child site count, and has a valid parent action
containing the identity. Distinct crystallographic operations need not induce
distinct permutations on a finite carrier: 41.8502% of catalogues have such an
action kernel. The audit therefore quotients to the faithful image
`G_parent -> S_N`, deduplicates induced permutations, and verifies exact group
closure. Production quotient likelihood independently deduplicates the unique
target labelings, so duplicate operation multiplicity cannot alter probability.

Allowed Q1 inputs are upstream composition with an explicit oracle/generated
role, species-free parent geometry, parent-space-group prior, sampled supercell
index, and the parent site-action image. Child atomic numbers, occupational
classes, target class-to-species mapping, child space group, occupational
stabilizer, CIF row order, and material ID remain target-only audit fields.

Evidence: `result.json`; SHA-256
`00256b17833da8b9bc08f639b80a27da455eca849539893b8606acb7bd7ba1b8`.
