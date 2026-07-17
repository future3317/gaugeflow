# H0-A Alex child split qualification

## Decision

**H0-A passed under `h0_a_alex_formula_prototype_split_v1`.** This result does
not pass H0 as a whole and does not authorize H1a or H1b.

## Frozen data and split

- Source: all 675,204 Alex-MP-20 rows under `E:/DATA/T2C-Flow`.
- Split counts: train 540,164; validation 67,520; test 67,520.
- Connected components: 567; largest component: 84,581 rows.
- Target fractions: 0.8/0.1/0.1; maximum absolute fraction deviation:
  `1.1848271040948433e-06`.
- Assignment SHA-256:
  `aa3ac522da6fd8f6d2548915d6db9b1dc628eef371cd36dee62ca3bc143e7ab2`.

Each row is joined to an exact reduced-formula node and to a conservative
StructureMatcher envelope consisting of its anonymous reduced stoichiometry
and standardized primitive-site count. The resulting bipartite connected
component is the indivisible split unit. Exact spglib primitive orbit
signatures remain recorded as a finer diagnostic.

## Failed first definition and repair

The first attempt connected formulas only through exact spglib primitive orbit
signatures. It balanced the row counts and removed exact formula/prototype
overlap, but a frozen StructureMatcher audit found cross-split near duplicates:
the exact orbit signature was too fine for the matcher tolerance neighborhood.
That attempt was not promoted.

The repaired definition uses a necessary-condition envelope for
`StructureMatcher.fit_anonymous` with `primitive_cell=True` and
`attempt_supercell=False`. Two structures in different envelopes cannot be a
matcher pair. The repaired split has zero cross-split overlap for reduced
formula, exact prototype, matcher envelope and connected component. Therefore
the complete cross-split StructureMatcher candidate universe is empty; this is
an exhaustive certificate rather than a sampled zero-hit statement.

## Artifact hashes

- Split manifest:
  `746352f9fd135599fd1bec42666dd5b0eee9d9e99a296147dbb8a68c23eae53d`.
- Independent split audit:
  `b655509f522dd5c108b3e71066a2769c8b95074b466ca9244545013dfa81e373`.
- Assignment Parquet:
  `aa3ac522da6fd8f6d2548915d6db9b1dc628eef371cd36dee62ca3bc143e7ab2`.

The original source Parquet files were not modified. All parent candidates,
alternate cells, OPD paths, mode scans and cross-source joins must inherit this
child assignment.
