# h0_data_activation_v3

## Technical summary

**Decision: `H0_not_passed_stop_before_H1`.** The external data center is present and its frozen manifest is readable, but source availability is not equivalent to model-ready activation. No H1 training is authorized by this audit.

## Gate status

| Component | Status |
| --- | --- |
| H0-A | `qualified` |
| H0-B | `qualified` |
| H0-C | `blocked_frozen_teacher_missing` |
| H0-D | `blocked_catalogue_missing` |
| H0-E | `blocked_pilot_missing` |

## Evidence and interpretation

- Alex-MP-20 H0-A is qualified: all source hashes match, the child-first formula/prototype split contains {'test': 67520, 'train': 540164, 'val': 67520}, its 567 connected components have zero cross-split formula, prototype, matcher-envelope and component overlap, and the exhaustive StructureMatcher candidate universe across splits is empty.
- PhononDB H0-B is qualified under its versioned derivation attestation: all 10,034 compact Hessians pass full-universe algebraic constraints, while a frozen 1024-material long-tail/stratified sample passes acoustic-mode, degenerate-subspace, q/-q conjugacy and explicit NAC checks. This is not represented as a full-universe eigendecomposition audit.
- MatPES-PBE data files are present; no frozen, hashed teacher checkpoint is activated.
- A normalized OPD path-class catalogue and the 1,000--5,000 structure parent decomposition pilot do not yet exist.

## Scope and data definition

The audit reads external files in place. It does not copy raw data into the code repository, derive new labels, train a model, run relaxation, or access tensor gates.

## Method

Dataset files are checked against the frozen root manifest by relative path, byte count, and SHA-256 recorded in that manifest. Small derivation manifests are hashed directly. Scientific qualification additionally requires the versioned split, derivation tests, teacher checkpoint, path measure, and pilot artifacts.

## Limitations and next actions

1. Preserve the qualified Alex child split and require every later artifact to inherit it.
2. Preserve the qualified PhononDB derivation and its source-confidence fields.
3. Qualify a frozen MatPES-PBE teacher and disagreement model.
4. Build a deduplicated OPD physical path-class measure, then run the bounded parent decomposition pilot.
5. Split H1 into H1a (P1 real-data hybrid generator) and H1b (full 230-group/Wyckoff parent blueprint); H1 passes only when both pass.
