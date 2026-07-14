# Gate A11.0 periodic unlabeled-site automorphism audit

This is a read-only geometric-identifiability audit. It starts no training and does not alter Gate A through A10.

## Method

Each endpoint is first Niggli reduced, which quotients integer lattice-basis representations. Every site is then replaced by the same dummy species before `SpacegroupAnalyzer` enumerates periodic operations. Operations are converted to explicit site permutations using only lattice and fractional coordinates. Target elements are inspected only after those permutations and site orbits have been fixed.

Two partitions are reported. `proper_so3` keeps only determinant-positive Cartesian operations, matching GaugeFlow's tensor gauge. `full_o3_scalar` also retains improper operations because the proposed A11-G distance/dot-product type head is O(3)-invariant. The latter is therefore the conservative identifiability partition for A11-G.

## Results

| material | partition | operations | site orbits | mixed chemical orbits | fixed-CIF deterministic ceiling | decision |
|---|---|---:|---:|---:|---:|---|
| JVASP-1180 | proper_so3 | 4 | 1 | 1 | 0.500 | stochastic_assignment_and_quotient_supervision_required |
| JVASP-1180 | full_o3_scalar | 4 | 1 | 1 | 0.500 | stochastic_assignment_and_quotient_supervision_required |
| JVASP-22673 | proper_so3 | 2 | 2 | 2 | 0.500 | stochastic_assignment_and_quotient_supervision_required |
| JVASP-22673 | full_o3_scalar | 2 | 2 | 2 | 0.500 | stochastic_assignment_and_quotient_supervision_required |

## Orbit-level labels

- `JVASP-1180` / `proper_so3` orbit 0: sites [0, 1, 2, 3]; N:2, In:2; mixed=True; constant-label ceiling=0.500.
- `JVASP-1180` / `full_o3_scalar` orbit 0: sites [0, 1, 2, 3]; N:2, In:2; mixed=True; constant-label ceiling=0.500.
- `JVASP-22673` / `proper_so3` orbit 0: sites [0, 3]; B:1, N:1; mixed=True; constant-label ceiling=0.500.
- `JVASP-22673` / `proper_so3` orbit 1: sites [1, 2]; B:1, N:1; mixed=True; constant-label ceiling=0.500.
- `JVASP-22673` / `full_o3_scalar` orbit 0: sites [0, 3]; B:1, N:1; mixed=True; constant-label ceiling=0.500.
- `JVASP-22673` / `full_o3_scalar` orbit 1: sites [1, 2]; B:1, N:1; mixed=True; constant-label ceiling=0.500.

## Consequence

At least one full-O(3) unlabeled site orbit contains multiple target species. A deterministic distance/dot-product decoder cannot be judged solely by fixed-CIF site accuracy on that orbit. Any successor must specify stochastic balanced assignment and automorphism-quotient supervision before training; A11-G is not authorized by this audit.
