# TensorOrbit-JARVIS-v2 full-O(3) data quality report

## Technical conclusion

The newly rebuilt `tensororbit_jarvis_v2_full_o3_v2` artifact is internally
consistent and is the first local v2 artifact suitable for **future external
tensor-oracle qualification**. It is not yet permission to train GaugeFlow or
to claim real-tensor generation performance.

The audit found two material problems in the previous data lineage and fixed
both without overwriting history:

1. The old source manifest attached the real GMTNet commit
   `7a606a459ee48a320ed38450e391811fb43d5e19` to the wrong repository
   (`divelab/AIRS`). GitHub commit search resolves it to `YKQ98/GMTNet`, where
   the pinned file is `data/jarvis_diele_piezo.pkl`. A fresh download from that
   exact commit has SHA-256
   `2a57e081f0072b2ac7fca7769095adcded1d299d2cd971db5c93fd25eb66929d`,
   exactly matching the hash recorded for the old local pickle. Re-exporting
   it produced byte-identical normalized records with SHA-256
   `492278c8173cd21f6d7f4ca6ac8ee3e5da72fd8386fccb6bb2d1aae136ecd068`.
2. The earlier materialized v2 target cache used proper-SO(3) Reynolds
   projection for crystal compatibility. That is physically incomplete for a
   polar rank-three tensor. The new schema-3 build uses the full crystallographic
   O(3) point group, while retaining proper SO(3) only for the downstream
   condition orbit.

## Dataset and grain

- Upstream grain: one GMTNet/JARVIS material record per `JARVIS_ID`.
- Upstream records: 5,000 unique material IDs.
- Explicit exclusions: 2 (`JVASP-44417`, `JVASP-8639`), each recorded as absent
  from the frozen 4,998-row parent population.
- Modeling artifact: 4,998 unique material rows with one CIF, one source 3x6
  piezoelectric matrix in `C/m^2`, and one projected Cartesian 3x3x3 target.
- Frozen split: 4,000 train / 499 validation / 499 test.
- Target-cache files: 4,998, with an exact material-ID join to the split and
  CSV rows.

## Checks performed and evidence

| Quality dimension | Result | Evidence and implication |
|---|---|---|
| Source provenance | Pass after correction | Official `YKQ98/GMTNet` commit and file path resolve; fresh pickle hash matches the previously recorded hash. |
| Normalized export reproducibility | Pass | Re-exported JSON and exclusions are byte-identical to the old normalized files. |
| Completeness | Pass | 5,000 raw rows = 4,998 split rows + 2 explicit exclusions; 4,998 target files present. |
| Uniqueness and joins | Pass | No duplicate raw or split IDs; split, CSV and cache joins are exact. |
| Split leakage | Pass for v2 | Reduced-formula overlap is zero for train/val, train/test and val/test. The composition precondition therefore leaves zero possible cross-split StructureMatcher matches. |
| Tensor schema | Pass | Every cache is schema 3, finite, 3x3x3 and symmetric in its two strain indices. |
| Physical compatibility | Pass | Every target passes full-O(3) Reynolds invariance at the registered tolerance; polar inversion-compatible targets are zero. |
| Voigt conversion | Pass | Per-row source order and engineering-shear convention are explicit; Cartesian/Voigt round trips pass. |
| CIF validity | Pass | All 4,998 CIFs parse during the audit. Pymatgen reports benign finite-precision coordinate rounding on a subset. |
| Hash integrity | Pass | Split, source, CSV, target-index, row-audit and build-manifest hashes are recorded in the attestation. |

## Distribution findings

The full-O(3) projection yields 2,300 exact physical zero tensors, or 46.0% of
the artifact. This is a real symmetry property rather than missing-label
encoding and must remain distinct from a CFG null condition.

| Split | Rows | Exact zeros | Zero rate |
|---|---:|---:|---:|
| Train | 4,000 | 1,826 | 45.65% |
| Validation | 499 | 231 | 46.29% |
| Test | 499 | 243 | 48.70% |

The test zero rate is about 3.0 percentage points above train. Evaluation must
therefore report the physical-zero subset and nonzero response strata
separately; a single aggregate tensor error could hide a model that predicts
zero too often.

Relative to the legacy proper-SO(3) cache, full-O(3) compatibility changes the
target norm of 369/4,998 rows, makes 3 additional targets exactly zero, and
changes one target norm by as much as `7.0978 C/m^2` in Cartesian Frobenius
norm. Consequently, checkpoints or oracle results trained on the legacy cache
are not scientifically interchangeable with this build.

## Severity and model impact

- **Critical, resolved for the new artifact:** the source repository in the
  old manifest was wrong. The content hash happened to be correct, but the
  lineage could not be independently reproduced until the repository was
  corrected and the upstream file re-downloaded.
- **High, resolved for the new artifact:** legacy proper-SO(3) crystal
  projection omitted improper symmetry constraints. This changes hundreds of
  targets and materially affects the learning problem.
- **High, unresolved for real-tensor evaluation:** no two architecture-distinct
  external tensor oracles have yet passed matched qualification on this exact
  build. The data can be used to begin oracle qualification, not GaugeFlow
  claims.
- **Medium, requires stratified reporting:** nearly half of targets are exact
  zero and the split zero rates are not identical. Training may balance
  sampling, but validation/test must retain their natural distribution and
  report zero/nonzero metrics separately.

## What this does and does not explain

These data defects could invalidate a future real-data benchmark or tensor
oracle trained on the old cache. They do **not** explain P5-C0 or D0.3--D0.8:
those coordinate-substrate experiments use a single code-defined four-atom
endpoint and random source coordinates, with no JARVIS row or piezo target.
Their negative results remain generator/path evidence, not dataset evidence.

## Activation boundary

The new build has status `raw_build_audit_passed_oracle_still_unqualified`.
The next permitted data-dependent activity is a separately frozen qualification
of GMTNet and an architecture-distinct equivariant tensor predictor on this
exact schema-3 build. GaugeFlow full training, real-tensor fidelity claims,
relaxation, DFT and DFPT remain blocked.
