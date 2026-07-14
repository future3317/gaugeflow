# TensorOrbit-JARVIS-v2 activation audit

## Result

The v2 formula-grouped split is internally valid as an **inactive candidate**.
All checks are `True`; its status remains
`candidate_not_active`.  Gate A.1 continues to use v1 and its frozen eight-ID
training panel.  This audit neither activates v2 nor changes any Gate A result.

## Data and split checks

- Candidate counts: `{"train": 4000, "val": 499, "test": 499}`.
- ID join to the 4,998 audited rows: `True`.
- Formula-group overlaps train/val, train/test, val/test: `{"train_val": 0, "train_test": 0, "val_test": 0}`.
- Response-stratum metadata agrees with recomputation: `True`.
- Exact physical zero tensors by candidate split: `{"test": 242, "train": 1824, "val": 231}`.
- All target-cache IDs resolve: `True`.
- Candidate and source hashes are recorded in `activation_manifest.json`.

## StructureMatcher near-duplicate control

The exact prior `StructureMatcher` configuration is recorded in the manifest.
Its composition precondition leaves **0** cross-split same-reduced-formula
candidate groups after formula grouping; therefore the candidate has 0
cross-split near-duplicate pairs under that control.  This is stronger than
the v1 result, which had 56 pairs, but it is a data-split property, not Gate A
performance evidence.

## Activation boundary

Activating v2 requires a new versioned training/evaluation protocol and new
checkpoints.  It must not be substituted into v1 results, this causal audit,
or a 4,000/499/499 claim without that separate protocol.
