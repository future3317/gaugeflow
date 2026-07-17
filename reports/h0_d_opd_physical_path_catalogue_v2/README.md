# H0-D affine OPD catalogue v2

Decision: **`H0-D-v2_qualified_H0-E_may_start`**.

This result supersedes without overwriting the frozen H0-D-v1 failure. It
qualifies the abstract, low-index commensurate affine OPD catalogue and its
parent-realization interface. It does not perform the H0-E Alex parent
decomposition pilot and therefore does not authorize H1 or later gates.

## Qualified object

For every one of the 230 space groups and every parent-rotation-quotiented
upper HNF with `det(B) <= 4`, the builder constructs the complete finite affine
quotient

\[
Q_{p,B}=G_p^B/T_B,
\]

including exact Seitz translation cosets. The complete physically real
irreducible representations of `Q_{p,B}` are obtained from its Cayley table.
Nonzero fixed spaces are quotiented by parent-domain conjugacy and stored with
their pointwise affine stabilizer as at most three packed 64-bit words.

The concrete parent occurrence test is deliberately deferred to H0-E: a mode
is retained there only when its real character has positive integral
multiplicity in the compact parent displacement representation.

## Equivalent acceleration

The implementation changes representation, not the physical catalogue:

- a displacement action is stored as a node permutation plus one Cartesian
  `3 x 3` rotation, never a dense `3N x 3N` matrix;
- homomorphism is checked for `generators x G`, which is a complete induction
  proof for all left factors and avoids an all-pairs matrix tensor;
- instead of enumerating every abstract subgroup, it closes the fixed-space
  lattice directly using

  \[
  \operatorname{Fix}\langle g_1,\ldots,g_k\rangle
  =\bigcap_i\ker[D(g_i)-I].
  \]

  Each intersection is evaluated as the kernel of a batched positive
  semidefinite matrix. A full-subgroup Reynolds calculation is retained only
  as an independent test, not as a runtime fallback;
- physical stabilizer intersections are exact bitwise AND operations;
- duplicate enumeration tuples are removed before the frozen path measure is
  normalized, so tuple multiplicity cannot change prior mass.

## Full build

| Quantity | Result |
| --- | ---: |
| Parent space groups | 230 |
| Canonical HNF orbits | 6,188 |
| Explicit exact branches | 230 |
| Physical-real irreps | 53,441 |
| Abstract OPD classes | 75,416 |
| Maximum affine quotient order | 192 |
| Build workers | 4 CPU processes |
| Build time | 183.19 s |

The versioned external artifacts are:

- `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v4/opd_catalogue_v2/catalogue_manifest.json`
- `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v4/opd_catalogue_v2/catalogue_records.json.gz`
- records SHA-256:
  `ba711a63874444cfc5ca1c9d5603cdcad65ae63799f44fd209aff979e73c940c`

The gzip writer fixes its timestamp and filename header, so an identical
catalogue has a stable content hash.

## Independent audit

[`audit.json`](audit.json) does not accept the builder's booleans as evidence.
It independently:

- reconstructs all 230 parents and all 6,188 affine quotients;
- verifies every Cayley-table hash and HNF orbit;
- verifies physical-real regular-representation completeness certificates;
- fully rebuilds the algebra records for SG 1, 2, 62 and 221;
- checks basis gauge, domain/enumeration relabeling, compact displacement,
  multiplicity-free measure and packed-intersection invariance;
- cross-checks the SG 221 Gamma polar-vector OPD classes against
  `spgrep-modulation` (6 classes versus 6 classes).

Every preregistered check passed. Package versions were `spglib 2.7.0`,
`spgrep 0.6.0`, `spgrep-modulation 0.3.0` and `hsnf 0.4.0`.

## Scientific boundary

H0-D-v2 is an abstract catalogue qualification. H0-E must still instantiate
the catalogue on the frozen Alex parent panel, prove positive displacement
occurrence, assign realized physical-class mass, and report coverage. H1a,
H1b, tensor conditioning, oracle work, relaxation, DFT and DFPT remain blocked.
