# H0 data activation v1

## Technical summary

**Decision: `H0_not_passed_stop_before_H1`.** The external data center is present and its frozen manifest is readable, but source availability is not equivalent to model-ready activation. No H1 training is authorized by this audit.

## Gate status

| Component | Status |
| --- | --- |
| H0-A | `blocked_split_not_frozen` |
| H0-B | `partial_missing_derivation_attestations` |
| H0-C | `blocked_frozen_teacher_missing` |
| H0-D | `blocked_catalogue_missing` |
| H0-E | `blocked_pilot_missing` |

## Evidence and interpretation

- Alex-MP-20 Parquet sources match the frozen data-center manifest. The source profile contains 675,204 structurally valid rows; upstream reduced-formula overlap is train--val 15621, train--test 15524, and val--test 4278. GaugeFlow's formula/prototype-disjoint child split has not been materialized.
- PhononDB contains 10,034 successful compact float64 force-constant caches with phonopy 4.3.1; remaining formal attestations: translational_zero_mode_test, degenerate_subspace_numerical_test, non_analytic_correction_attestation.
- MatPES-PBE data files are present; no frozen, hashed teacher checkpoint is activated.
- A normalized OPD path-class catalogue and the 1,000--5,000 structure parent decomposition pilot do not yet exist.

## Scope and data definition

The audit reads external files in place. It does not copy raw data into the code repository, derive new labels, train a model, run relaxation, or access tensor gates.

## Method

Dataset files are checked against the frozen root manifest by relative path, byte count, and SHA-256 recorded in that manifest. Small derivation manifests are hashed directly. Scientific qualification additionally requires the versioned split, derivation tests, teacher checkpoint, path measure, and pilot artifacts.

## Limitations and next actions

1. Freeze the Alex child split before constructing any parent/path/join artifact.
2. Add the missing PhononDB translational-mode, degenerate-subspace, and NAC attestations without changing the existing force-constant cache.
3. Qualify a frozen MatPES-PBE teacher and disagreement model.
4. Build a deduplicated OPD physical path-class measure, then run the bounded parent decomposition pilot.
5. Split H1 into H1a (P1 real-data hybrid generator) and H1b (full 230-group/Wyckoff parent blueprint); H1 passes only when both pass.
