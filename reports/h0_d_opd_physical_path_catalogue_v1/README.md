# H0-D OPD physical-path catalogue v1

## Decision

`H0-D_failed_stop_before_H0-E`.

The frozen contract test passed, but no qualified catalogue manifest exists at
`E:/DATA/T2C-Flow/processed/gaugeflow_h0_v4/opd_catalogue_manifest.json`.
Consequently H0-D does not pass and H0-E/H1 remain unauthorized.

This is a scientific-object failure, not a package-availability failure:

- `spglib==2.7.0`, `spgrep==0.6.0`, and `spgrep-modulation==0.3.0` are present
  at the preregistered versions;
- the current production `OPDBranch` stores point-operation indices but not
  translation cosets, so it cannot certify an affine child subgroup;
- `spgrep-modulation` enumerates useful little-group isotropy branches and is
  retained as an independent cross-check, but its output alone is not a
  complete real-k-star physical catalogue;
- no evidence currently proves parent-normalizer, k-star, OPD-basis,
  unimodular-cell and domain-relabeling quotient invariance together;
- no physical class measure exists whose mass is independent of catalogue
  tuple multiplicity.

## Required repair before rerun

The next version must construct the finite affine quotient
`G_parent^B / T_B`, retain rotation and translation-coset identity, form the
full real k-star representations that actually occur in a concrete parent
Wyckoff displacement representation, and canonicalize fixed-space/stabilizer
orbits before assigning class mass. A point-group-only compatibility table or
ordinary subgroup enumeration is not an acceptable substitute.

The exact branch must be represented explicitly for all 230 parent space
groups. Distorted entries may only be activated when their displacement-irrep
occurrence is certified for the parent blueprint.

## Evidence

- Frozen protocol: `configs/gates/h0_d_opd_physical_path_catalogue_v1.json`
- Machine result: `result_manifest.json`
- Contract tests: `2 passed`

No H0-E parent-decomposition pilot, H1 training, tensor work, oracle,
relaxation, DFT or DFPT was run.
