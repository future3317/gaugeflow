# GaugeFlow data quality audit

## Technical summary

The 4,998-row tensor/CIF join is complete and unique, all projected tensors are
finite and Reynolds-consistent, and all tracked basis changes round-trip. The
current v1 split is **not formula-disjoint**: 165 reduced
formula groups affecting 672 rows cross train/validation/test.
This is a high-severity benchmark-leakage risk, so v1 results must not be
described as formula-disjoint. The frozen v1 files remain unchanged.

This issue is **high severity** for validation/test generalization claims, but
it does not explain the frozen eight-record Gate A training-panel diagnostic:
all eight Gate A IDs remain in train and Gate A asks whether the conditioning
path can control those training examples. It does block promotion to the full
4,000/499/499 benchmark until a new split protocol is activated.

## Scope

- Rows audited: 4998 (full artifact)
- Physically/data-valid rows: 4998
- Source duplicate material IDs: 0
- Duplicate IDs across protocol splits: 0
- Missing source IDs: 0
- Extra source IDs outside the frozen artifact: 0
- Missing target-cache files: 0
- Extra target-cache files: 0
- Gate A eight IDs exactly present in the frozen train split: True

## Tensor and geometry checks

- Exact-zero tensors by split: `{"test": 223, "train": 1853, "val": 221}`
- Invalid or non-finite target rows: 0
- Voigt/Cartesian round-trip tolerance: `2e-7` (FP32)
- Reynolds invariance tolerance: `5e-4`
- Niggli/basis quotient round-trip tolerance: `1e-5`
- Alternate but equivalent second Niggli representatives: 77
- The tracked cell-basis operation never acts on the Cartesian tensor.

## Split and leakage checks

- Reduced-formula overlap train/val: 94
- Reduced-formula overlap train/test: 73
- Reduced-formula overlap val/test: 10
- Union of overlapping formula groups: 165
- Rows in an overlapping formula group: 672 / 4998
- Cross-split StructureMatcher near-duplicate pairs: 56
- Formula grouping failure is sufficient to reject the `formula-disjoint` label;
  StructureMatcher pairs identify the stronger primitive/supercell/near-duplicate
  leakage class. No frozen split was modified.
- Runtime source-token counts (mentions are not necessarily model inputs):
  `{"space_group": 0, "stabilizer_rotations": 2, "target_graph": 0, "target_lattice": 0}`. Dataset records are checked
  by regression tests to contain tensor conditions and current crystal state,
  not target-CIF stabilizers, space groups, paired target graphs, or target lattices.

## Inactive repair candidate

An explicitly versioned candidate was generated without modifying or activating
v1: `artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/splits.json`.

- Status: `candidate_not_active`
- Counts: 4,000 train / 499 validation / 499 test
- Reduced-formula groups are strictly disjoint across all three splits.
- All eight Gate A IDs are retained in train.
- Seed: `20260714`; selected deterministic balance trial: 12 of 100
- Candidate SHA-256:
  `e2f5c08014b9c62836523d85e80e79e498fb9c7ba2bfc2273564e17327a12e5e`

The candidate is a remediation artifact, not a silent replacement. Activating
it requires a new protocol/version and invalidates comparisons to results that
used v1.

## Integrity outputs

- Row-level audit: `reports/data_quality_rows.csv`
- Cross-split structural matches: `reports/data_quality_cross_split_matches.csv`
- Manifest: `reports/data_quality_manifest.json`
- Manifest SHA-256: `3e1128fc1ec48a0e2ca7d53be5153a216ddc5610fc34a8aa34dc579b82d3ce1d`
- Frozen split SHA-256: `6ac07d1456490a197fc8e32bd59fa7320d374130771a4490c3e47784c05912d6`
- Computed target-cache index SHA-256: `2f7cd811e9ededd1bfff01a9832db4fe2deb447bc52000355281ac005e187d86`
- Versioned preprocessing cache manifest:
  `artifacts/tensororbit_jarvis_v1_preprocessed_v1.pt.manifest.json`
- Rebuilt cache SHA-256:
  `ef4ad9fc70a6786bb0937132ce932a1f955fe35403b8af202e6b07a06ec9b333`

Any failed row remains in the CSV with its exception. This report does not
alter the frozen split or declare Gate A passed.
