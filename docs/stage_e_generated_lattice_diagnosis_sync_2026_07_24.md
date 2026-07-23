# Stage-E Generated-Lattice Diagnosis Sync - 2026-07-24

Status: Stage-E v1 remains blocked.  This document synchronizes the latest
multi-step diagnosis so future work does not re-open already answered causes or
skip the generated-state provenance contract.

## Executive Conclusion

The Stage-E generated-lattice failure is not a single sampler bug and is not a
reason to start Stage-F.  The evidence now supports this chain:

```text
missing composition_counts
  -> old lattice adapter volume drift

full adapter shape residual
  -> counts-fixed C-new volume/NN decoupling in oracle_ca

partial/MASK orderless assignment carrier shift
  -> remaining oracle_c tail, especially target 123

single shared E-v1 lattice adapter
  -> cannot simultaneously serve clean/full and partial/MASK carrier regimes
```

The current conclusion is therefore:

```text
Stage-E v1 is frozen as blocked.
A-v2 must first qualify provenance-tracked generated-state coverage.
```

## What Was Fixed

### 1. Composition Counts Interface

Commit:

```text
838364d9 fix: preserve composition context in Stage-E lattice exposure
```

Finding:

- `composition_counts` missing from generated-side lattice exposure was a real
  interface bug.
- It explains the old adapter's volume catastrophe.
- A provenance test now confirms generated-side arms do not read clean target
  counts.
- Counts sum, padding, vocabulary mapping, per-graph batch separation,
  composition order invariance and exact-count relabel consistency were checked
  against the Stage-A/C composition semantics.

Interpretation:

```text
counts bug explains volume drift, but not the full Stage-E failure.
```

### 2. Shape Residual Scale

Commit:

```text
7b8f222d fix: scale Stage-E lattice shape residual
```

Zero-training intervention:

```text
volume = base_volume + adapter_delta_volume
shape  = base_shape  + alpha * adapter_delta_shape
alpha in {0, 0.25, 0.5, 0.75, 1.0}
```

The volume residual was left intact.  Only the five-dimensional trace-free
shape residual was scaled.  No hard clipping, retraining, sampler change,
checkpoint change or random-stream change was used.

Official smoke32 evidence for the counts-fixed adapter:

| alpha | arm | tensor RMSE | volume-W1 | NN-W1 | valid distance | max condition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | oracle_ca | 0.8468 | 0.0823 | 0.5989 | 1.000 | 6.15 |
| 0.25 | oracle_ca | 0.7862 | 0.0792 | 0.5281 | 1.000 | 5.26 |
| 0.50 | oracle_ca | 0.8077 | 0.0773 | 0.5952 | 1.000 | 7.00 |
| 0.75 | oracle_ca | 0.8133 | 0.0783 | 0.6339 | 1.000 | 9.76 |
| 1.00 | oracle_ca | 0.7178 | 0.0823 | 0.6645 | 1.000 | 12.99 |
| 0.25 | oracle_c | 1.0153 | 0.0759 | 0.7010 | 1.000 | 12.48 |
| 0.25 | free | 0.9833 | 0.3061 | 0.3292 | 1.000 | 3.59 |

Interpretation:

- `alpha=0.25` preserves volume correction and repairs the `oracle_ca`
  volume/NN decoupling.
- Full `alpha=1.0` overdrives shape and creates condition-number tails.
- This is a local diagnostic fix, not a Stage-E pass.

### 3. Lattice-Coordinate Counterfactual

The C-new lattice is not automatically unphysical at the endpoint.  Clean
coordinates placed on the C-new lattice did not immediately worsen NN.  The
failure is more likely produced during reverse sampling:

```text
shape trajectory changes periodic distances and dynamic neighbor graphs
  -> coordinate score is evaluated out of its qualified carrier distribution
```

This means coordinate exposure is not automatically authorized.  The first
failure mode to control is still the generated-state lattice/assignment carrier
path.

### 4. Orderless Partial/MASK Exposure

Commits:

```text
50a1c93d fix: expose partial assignment state in Stage-E lattice adapter training
b747613c fix: retain clean lattice exposure carrier in Stage-E adapter training
```

Protocol change:

```text
element_exposure = orderless_partial
```

Generated exposure uses nested orderless partial/MASK element states with exact
composition counts.  Clean-retention query still uses clean element tokens.

Smoke32 comparison:

| adapter | shape scale | oracle_ca volume-W1 | oracle_ca NN-W1 | oracle_c volume-W1 | oracle_c NN-W1 | free NN-W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| counts-fixed clean exposure | 0.25 | 0.079249 | 0.528087 | 0.075922 | 0.701007 | 0.329158 |
| orderless partial v2 | 0.25 | 0.157782 | 0.541020 | 0.086297 | 0.667458 | 0.319875 |
| orderless partial v2 | 0.00 | 0.149005 | 0.639191 | 0.086886 | 0.673520 | 0.321570 |

Target 123 remains pathological in `oracle_c`:

| adapter | scale | condition |
| --- | ---: | ---: |
| counts-fixed clean exposure | 0.25 | 12.483171 |
| orderless partial v2 | 0.25 | 12.530057 |
| orderless partial v2 | 0.00 | 11.041600 |

Interpretation:

- Partial/MASK exposure is a real missing carrier state.
- It slightly improves `oracle_c`.
- It damages clean/full-assignment volume retention in `oracle_ca`.
- It does not remove target 123's condition tail.
- Therefore the partial adapter is a diagnostic artifact, not a production
  candidate.

## Frozen Stage-E Answer

The current answers to the required handoff questions are:

| question | answer |
| --- | --- |
| Is the fixed adapter better than the old adapter? | Yes for old volume drift; not enough for Stage-E qualification. |
| How much did missing `composition_counts` explain? | It explains the old adapter volume catastrophe, not the remaining NN/shape failure. |
| Which path still fails? | `oracle_c` remains the decisive generated-assignment path; `free` is still not qualified. |
| Is Stage-E still blocked? | Yes. |
| Next minimal root-cause hypothesis? | v1 lacks a provenance-qualified generated-state coverage contract across generated assignment, lattice and coordinate carriers. |

## Do Not Do

- Do not start Stage-F.
- Do not claim Stage-E pass.
- Do not rerun A/B/C v1 to overwrite history.
- Do not make the partial/MASK adapter a production default.
- Do not keep scalar-tuning E-v1 adapters as the main line.
- Do not update the paper to say tensor-conditioned generation works.
- Do not delete historical worktrees, checkpoints, runs or evaluations without
  a separate read-only inventory and explicit approval.

## Next Allowed Work

The next line is A-v2 generated-state coverage, starting with provenance rather
than scale.  The first provenance layer is now partially implemented and smoke
tested; the work must continue as a correctness audit before any large training
run.

Completed provenance steps:

1. The generated-state contract in
   [`gaugeflow_base_v2_generated_state_contract.md`](gaugeflow_base_v2_generated_state_contract.md).
2. Replay/cache writer and loader around
   `GeneratedStateReplayEntry`.
3. Fail-closed behavior for stale checkpoint hashes, sampler protocol
   mismatch, target leakage and forbidden source ID overlap.
4. Synthetic four-role cache smoke.
5. Tiny real cache smoke using two real Alex train structures and four carrier
   roles.
6. Tiny real replay-cache per-role loss/gradient audit through the current
   `TensorFreeHybridDiffusion` objective.
7. Forbidden-source overlap audit against Stage-D validation/test and the
   frozen Stage-E 256-sample factorial target panel.
8. 34M generated-state replay correctness runner and 20-step tiny training
   smoke on the eight-entry real replay cache.
9. Bounded 34M 2k generated-state correctness run on the same replay contract.

Current immediate task:

1. Evaluate the 34M 2k correctness checkpoint with the same frozen
   generated-state stratification before any capacity run.
2. Do not expand to 58M/98M or multi-GPU capacity competition until the 34M
   generated-state contract has rollout evidence, not only training-interface
   evidence.

Deferred work:

1. Run the 34M 2--5k correctness experiment only after the tiny training smoke
   verifies finite loss, nonzero active gradients and actual parameter updates.
2. Defer 58M/98M or multi-GPU full training until the 34M generated-state
   contract is proven.

Large GPU capacity should be used after this provenance layer is closed, not to
continue blind E-v1 adapter tuning.

## Evidence Index

Reports:

```text
reports/stage_e_shape_residual_dose_v1/README.md
reports/stage_e_orderless_partial_exposure_v1/README.md
reports/gaugeflow_v1_freeze_2026_07_24/README.md
```

Server artifacts:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/
  stage_e_lattice_generated_exposure_jarvis_v1/adapter.pt
  stage_e_lattice_generated_exposure_jarvis_countsfix_v1/adapter.pt
  stage_e_lattice_generated_exposure_jarvis_orderless_partial_v2/adapter.pt

/home/workspace/lrh/DATA/T2C-Flow/evaluations/
  stage_e_countsfix_shape_scale025_official_smoke32_v1/
  stage_e_orderless_partial_v2_shape_scale025_smoke32_v1/
  stage_e_orderless_partial_v2_shape_scale0_smoke32_v1/
  generated_state_replay_cache_smoke_v2/
  generated_state_replay_tiny_real_smoke_v3/
    training_contract_audit.json
    training_contract_audit_with_forbidden_panel.json
    forbidden_source_ids_stage_d_stage_e_v1.json
    forbidden_source_ids_stage_d_stage_e_v1.manifest.json
  generated_state_replay_correctness_train_smoke_v2/
    training_summary.json
    checkpoint_metadata.json
    training_metrics.jsonl
  /home/workspace/lrh/DATA/T2C-Flow/runs/
  generated_state_replay_correctness_34m_2k_v1/
    training_summary.json
    checkpoint_metadata.json
    training_metrics.jsonl
    checkpoint_step_00002000.pt
```

Code/provenance boundary:

```text
src/gaugeflow/production/generated_state_replay.py
scripts/smoke_generated_state_replay_manifest.py
scripts/build_tiny_generated_state_replay_cache.py
scripts/audit_generated_state_replay_training_contract.py
scripts/train_generated_state_replay_correctness.py
tests/test_generated_state_replay.py
```

Latest A-v2 provenance commits:

```text
8bb37cab docs: record generated-state replay cache status
48186d52 fix: load Stage-C backbone for tiny replay cache
55cdb62e docs: record tiny real generated-state replay smoke
11ac6a9 feat: audit generated-state replay training contract
2e44df8 docs: record replay training contract audit
23cde00 feat: train generated-state replay correctness smoke
```

Latest tiny training smoke:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_train_smoke_v2/
status: passed
steps: 20
entry_count: 8
role_weight: 0.25
all_final_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_max: 2.5471673704374154
first_step_parameter_update_norm: 1.0069770103808358
final_parameter_update_norm: 5.613741470653719
forbidden_source_id_check: executed, count=773
```

Latest 34M correctness run:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_2k_v1/
status: passed
steps: 2000
entry_count: 8
role_weight: 0.25
training_metrics_rows: 2000
checkpoint_step_00002000_sha256:
  2935365787b934cfdd58bc8a47a2cf104654cd736b946eb5f493b0223de9e560
all_final_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_max: 5.617618305543426
first_step_parameter_update_norm: 1.0069770103808358
final_parameter_update_norm: 37.47103131901986
forbidden_source_id_check: executed, count=773
```
