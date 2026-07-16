# vNext protocol amendment: Q0.1 partial legacy completion

This versioned amendment responds to `GAUGEFLOW_Q0_CODE_REVIEW.md`, SHA-256
`3e4dc7ad9cf16ddb2bda4014862da1d1f37426934c3559524470d95a2162e6a2`.
It does not edit the controlling specification, the original Q0 config, or the
frozen Q0 run at commit `42a34c5`.

## Why a versioned amendment is necessary

Original Q0 correctly ended as `blocked`: the historical P5-C0 runner did not
save a checkpoint and retraining is forbidden. That immutable artifact cannot
be recovered. Q1, however, is a new regular finite-time affine-flow
qualification and has no dependency on the historical learned field. Making
Q1 depend forever on an unrecoverable artifact would deadlock the protocol
without improving scientific validity.

Q0.1 therefore separates execution completeness from a scientific verdict:

- `execution_status = complete_partial_legacy` means every recoverable,
  checkpoint-independent audit was executed with corrected metric semantics;
- `scientific_verdict = legacy_learned_field_unclassified` preserves the
  missing learned-field evidence;
- `q1_authorized` may become true only when every pre-registered P0 release
  check in `configs/gates/q0_1_partial_legacy_audit.yaml` passes.

No Q0 or Q0.1 state is called a scientific pass. A future Q0R reenactment, if
ever authorized, must be labelled as a new model and may not backfill the
historical checkpoint.

## Corrected metric semantics

- kNN output is named `knn_local_target_dispersion`; it is not an estimator of
  irreducible conditional variance.
- `exact_equivalence_risk` is computed over fixed-tolerance representation
  equivalence classes.
- exact collisions, quantile-defined near pairs, and alias witnesses are
  separate fields and tables.
- solver error against the analytic finite-time solution and residual to the
  limiting target are separate columns.
- coupling replay compares assignment, integer lift, translation, optimum,
  second optimum, endpoint, and velocity, and records content hashes.

## Gate order after amendment

The logical order is frozen Q0 -> Q0.1 -> P0 release checks -> Q1v2. Q2 and all
later gates retain their original order. Real tensor training, learned tensor
oracles, relaxation, DFT, and DFPT remain prohibited. Q10 still needs an
independent human unlock even if all predecessors eventually qualify.
