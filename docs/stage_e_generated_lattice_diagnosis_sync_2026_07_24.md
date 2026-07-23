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

## Current Conversation Sync

The latest working goal is to continue diagnosis until the Stage-E
generated-lattice/coordinate degradation has an evidence-backed root cause and
then apply only the minimum authorized fix.  If those local fixes cannot
restore retention, the fallback is the staged retraining plan in
`../../重训计划.md`, not an immediate blind restart of A/B/C/E.

The latest diagnosis should be read in this order:

1. `composition_counts` provenance was a real interface bug and explains the
   historical adapter's volume drift.
2. The counts-fixed adapter then exposed a more specific failure:
   volume correction and local geometry were decoupled, with full shape
   residual worsening `oracle_ca` NN/condition tails.
3. Lattice-coordinate counterfactuals showed that clean coordinates on the
   C-new lattice did not immediately fail, so the endpoint lattice is not by
   itself proven unphysical.
4. The remaining failure is a reverse-trajectory coupling problem: generated
   shape trajectories change periodic distances and dynamic neighbor graphs,
   placing the coordinate score outside its qualified carrier distribution.
5. Shape-residual dose testing showed `shape_scale=0.25` is a useful
   diagnostic candidate, but not a Stage-E pass.
6. Partial/MASK orderless exposure is a real missing carrier regime, yet a
   single shared E-v1 lattice adapter cannot serve both clean/full and
   partial/MASK carriers without damaging retention.
7. The eight-entry generated-state replay run closed the optimizer/interface
   path, but overfit its tiny cache and damaged free-generation retention.

Therefore the active route is broader provenance-checked generated-state
coverage at 34M before any capacity expansion.  Available GPUs may be used for
parallel cache building, smoke evaluation, and later predeclared capacity
competition, but not to bypass the 34M replay-role plus free-retention gate.

## Do Not Do

- Do not start Stage-F.
- Do not claim Stage-E pass.
- Do not rerun A/B/C v1 to overwrite history.
- Do not make the partial/MASK adapter a production default.
- Do not keep scalar-tuning E-v1 adapters as the main line.
- Do not update the paper to say tensor-conditioned generation works.
- Do not delete historical worktrees, checkpoints, runs or evaluations without
  a separate read-only inventory and explicit approval.
- Do not make 58M/98M, larger batch, multi-GPU, or larger model capacity the
  next explanation for the current failure until the 34M generated-state path
  has non-degraded rollout evidence.
- Do not treat coordinate exposure as authorized solely from the NN regression;
  the clean-coordinate counterfactual first points to generated lattice/graph
  trajectory coupling.

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
10. Smoke32 replay-role/free-generation evaluation for the 2k checkpoint,
    with both EMA and raw weights.
11. Replay-cache builder now supports fail-closed `--forbidden-source-ids`
    selection and deterministic permuted source windows via `--selection-seed`.
12. A 32-source real replay cache was built from the frozen Stage-C 40523 base
    with the same four carrier roles, forbidden-source panel and sampler
    protocol.
13. The 32-source cache passed the same training-contract audit and a 20-step
    34M optimizer smoke without saving a production checkpoint.
14. A 32-source 2k correctness run passed training-interface checks but still
    degraded smoke32 free-generation NN, so broader coverage alone is not yet
    sufficient.
15. Shorter-update diagnostics showed that the usable region is early and
    EMA-dependent: 100/200-step EMA checkpoints preserve smoke32
    free-generation much better while still reducing all replay-role losses,
    while 100/200-step raw weights and 300+ step EMA already drift.

Current immediate task:

1. Treat the 8-entry 2k correctness checkpoint as an overfit diagnostic, not a
   production candidate.
2. Build a broader provenance-checked replay cache before any longer A-v2
   correctness run.
3. Do not expand to 58M/98M or multi-GPU capacity competition until 34M has
   both replay-role improvement and non-degraded free-generation retention.
4. Extend the replay-cache builder before scaling it: accept a forbidden source
   ID file, fail closed on overlap, and prefer deterministic random/permuted
   source selection over contiguous source slices.
5. Run the same training-contract audit and same smoke32 evaluator on the
   broader 32/64-source cache before increasing steps, batch size, or model
   width.
6. The 32-source cache has passed cache/audit/20-step training smoke, and the
   bounded 2k run has now been evaluated.  The next allowed experiment is not
   capacity scaling; it is a predeclared update-dose/checkpoint-selection
   diagnostic around the 32-source 200-step EMA candidate.

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
  generated_state_replay_correctness_eval_smoke32_v1.json
  generated_state_replay_correctness_eval_smoke32_noema_v1.json
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
25cbde3b fix: guard generated-state replay source selection
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

Latest 34M 2k evaluation:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_eval_smoke32_v1.json
checkpoint weights: EMA
replay role losses: lower than base for all four roles
free-generation base/candidate NN-W1: 0.5384073850 / 2.0589264243
free-generation base/candidate volume-W1: 0.3337970015 / 0.4414314562
free-generation base/candidate distance-valid: 1.0 / 0.9375

/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_eval_smoke32_noema_v1.json
checkpoint weights: raw model
replay role losses: lower than base for all four roles
free-generation base/candidate NN-W1: 0.5384073850 / 1.9576429188
free-generation base/candidate volume-W1: 0.3337970015 / 0.4852205126
free-generation base/candidate distance-valid: 1.0 / 1.0
```

Interpretation:

```text
The 8-entry replay correctness run proved optimizer/interface closure, but it
overfits the tiny replay cache and harms short free-generation retention.  The
failure is not explained by EMA lag, because raw weights show the same NN-W1
regression.  This blocks capacity scaling until the replay cache is broadened
and the same evaluator shows non-degraded free-generation retention.
```

Latest 32-source provenance cache:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32_real_v1/
status: passed
entries: 128 = 32 real source structures x 4 roles
selection mode: permuted
selection_seed: 6101
reverse_steps: 4
refresh_id: 2
manifest SHA-256:
  f59f58545bc1dab62664fad39b14806c0ef42e85f3d786c3cbaee78f131e4909
forbidden_source_id_check: executed, count=773
sampler_commit:
  25cbde3b6be0109d5e6cf68748f051632366a2f0
```

Latest 32-source training-contract audit:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32_real_v1/training_contract_audit.json
status: passed
entry_count: 128
all_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_to_max_generated: 0.3621880364977982
forbidden_source_id_check: executed, count=773
```

Latest 32-source 20-step correctness smoke:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32_train_smoke_v1/
status: passed
steps: 20
entry_count: 128
role_weight: 0.25
all_final_role_terminal_gradient_groups_nonzero: true
parameters_updated: true
first_step_parameter_update_norm: 1.0244015304898244
final_parameter_update_norm: 5.814717134166754
clean_retention_loss_ratio_max: 0.7123302917464417
forbidden_source_id_check: executed, count=773
```

32-source 34M correctness/evaluation summary:

| cache | steps | weights | replay role losses | free NN-W1 | NN delta | volume-W1 | volume delta | distance-valid |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8-source | 2000 | EMA | all lower | 2.058926 | +1.520519 | 0.441431 | +0.107634 | 0.9375 |
| 8-source | 2000 | raw | all lower | 1.957643 | +1.419236 | 0.485221 | +0.151424 | 1.0000 |
| 32-source | 100 | EMA | all lower | 0.545142 | +0.006735 | 0.326940 | -0.006857 | 1.0000 |
| 32-source | 100 | raw | all lower | 0.969413 | +0.431006 | 0.296441 | -0.037356 | 0.9688 |
| 32-source | 200 | EMA | all lower | 0.571096 | +0.032688 | 0.325896 | -0.007901 | 1.0000 |
| 32-source | 200 | raw | all lower | 1.124272 | +0.585865 | 0.267458 | -0.066339 | 1.0000 |
| 32-source | 300 | EMA | all lower | 0.669791 | +0.131383 | 0.328313 | -0.005484 | 1.0000 |
| 32-source | 500 | EMA | all lower | 0.666141 | +0.127733 | 0.327441 | -0.006356 | 1.0000 |
| 32-source | 2000 | EMA | all lower | 1.085160 | +0.546752 | 0.288742 | -0.045055 | 1.0000 |
| 32-source | 2000 | raw | all lower | 1.492386 | +0.953979 | 0.315211 | -0.018586 | 0.9375 |

Current interpretation:

```text
Broader provenance coverage fixes the catastrophic 8-entry overfit mode only
when the update dose is kept small and EMA weights are used.  Replay-role loss
improvement by itself is not a sufficient selection metric: raw weights at
100/200 steps are already too aggressive, and EMA checkpoints at 300/500/2000
steps continue to improve cached role losses while degrading free rollout
geometry.  The next root-cause hypothesis is replay-over-optimization/update
dose under a still-narrow generated-state support, not model capacity.
```

Current candidate:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_100_v1/
checkpoint_step_00000100.pt
checkpoint SHA-256:
  8b9bbd2cd30216b7801282f58af85e52c9742fac9a6f3b353eb5ac8e9ffa5a16

/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_200_v1/
checkpoint_step_00000200.pt
checkpoint SHA-256:
  164dc4277c6fd80274990ff4452731f1cb43b4c7a2ef61e7d27c45a68a03f995
status: diagnostic candidates only; 100-step EMA has the smallest smoke32 NN
drift, 200-step EMA has slightly stronger replay-role improvement.
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

## Current Handoff State

This handoff lineage includes:

```text
ef22924dc84fe1688d51dee7b70144e7dc59d90b
  docs: record replay early-EMA dose window
250da94b4a14c9da46165c25098bc2974f8e965b
  feat: select generated-state replay checkpoint
4b072bb5d0cf16d83a29117c7670a33a7586b296
  docs: record replay 64-sample validation
```

The active scientific boundary is:

- Stage-E v1 remains blocked.
- Counts absence explains the old lattice-adapter volume catastrophe.
- Full shape residual explains the counts-fixed `oracle_ca` NN/shape tail.
- Partial/MASK exposure is a real carrier gap, but the single shared E-v1
  adapter is not a production candidate.
- The current A-v2 route is generated-state replay coverage with strict
  provenance, not immediate model-capacity scaling.

The current diagnostic candidates are:

```text
100-step EMA:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_100_v1/checkpoint_step_00000100.pt
SHA-256:
8b9bbd2cd30216b7801282f58af85e52c9742fac9a6f3b353eb5ac8e9ffa5a16

200-step EMA:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_200_v1/checkpoint_step_00000200.pt
SHA-256:
164dc4277c6fd80274990ff4452731f1cb43b4c7a2ef61e7d27c45a68a03f995
```

They are diagnostic candidates only.  The predeclared checkpoint-selection
audit has now been implemented as:

```text
scripts/select_generated_state_replay_checkpoint.py
```

It was run on all existing 8-source and 32-source dose evaluation JSONs.  The
selection report is:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32src_checkpoint_selection_v1.json
```

Minimum selector requirements:

- every replay role total loss is lower than the base;
- free NN-W1 is non-degraded within the declared tolerance;
- volume-W1 is non-inferior or improved;
- distance-valid fraction and sampling failures do not regress;
- exact composition, finite-positive lattice and terminal masks remain valid;
- tie-breaks are declared before looking at any new validation panel.

Selector result:

```text
selected_label: 32src_100_ema
selected_checkpoint:
  /home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_100_v1/checkpoint_step_00000100.pt
selected_checkpoint_sha256:
  8b9bbd2cd30216b7801282f58af85e52c9742fac9a6f3b353eb5ac8e9ffa5a16
nn_w1_delta: 0.006734872866905883
volume_w1_delta: -0.006856855537134221
distance_valid_delta: 0.0
replay_total_loss_improvement: 0.5635005235671997
```

This was still diagnostic checkpoint selection, not a production promotion.
The selected 100-step EMA checkpoint was then evaluated on a bounded 64-sample
panel with the same evaluator and frozen random-stream policy:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_32src_100_eval_val64_v1.json
```

64-sample result:

```text
base/candidate NN-W1:
  0.7234887633353652 / 0.7274343809186921
NN-W1 delta:
  +0.003945617583326899
base/candidate volume-W1:
  0.11830551960912764 / 0.11909540452679011
volume-W1 delta:
  +0.0007898849176624784
distance-valid delta:
  0.0
sampling failures delta:
  0.0
terminal masks delta:
  0.0
exact composition delta:
  0.0
finite-positive lattice delta:
  0.0
```

The same selector was applied to the 64-sample output:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32src_checkpoint_selection_val64_v1.json
status:
  no_eligible_checkpoint
reason:
  volume_w1_non_inferior=false under max_volume_w1_delta=0.0
```

Interpretation:

```text
The 32-source 100-step EMA checkpoint preserves NN and hard validity on the
64-sample panel, but it does not pass the predeclared strict volume
non-inferiority rule.  It remains a diagnostic candidate only.
```

Full 58M/98M or multi-GPU capacity training remains deferred until 34M shows
both replay-role improvement and non-degraded rollout retention beyond smoke32.
The next admissible work is either broader generated-state replay support or a
new predeclared non-inferiority rule before running another bounded 34M
diagnostic; it is not immediate capacity scaling.

## 64-Source Replay Support Follow-Up

A broader 64-source real replay cache was built as the next admissible
diagnostic, without changing model capacity, loss, optimizer or sampler:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64_real_v1/
entries:
  256 = 64 sources x 4 roles
selection_seed:
  6101
source_start_index:
  32
refresh_id:
  3
reverse_steps:
  4
manifest SHA-256:
  bd10fa96d0175fa799da075906a18cb96fcffb609d9e3df63c5bea9dfcdfe11f
forbidden source ID check:
  executed, count=773
```

The 64-source training-contract audit passed:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64_real_v1/training_contract_audit.json
all_role_terminal_gradient_groups_nonzero:
  true
clean_retention_loss_ratio_to_max_generated:
  0.31113227758714657
```

64-source early-dose smoke32:

| cache | steps | weights | replay role losses | NN delta | volume delta | distance-valid delta | selector |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| 64-source | 50 | EMA | all lower | +0.019689 | -0.008498 | 0.0 | selected |
| 64-source | 100 | EMA | all lower | +0.053088 | -0.016800 | 0.0 | rejected: NN |
| 64-source | 200 | EMA | all lower | +0.088420 | -0.029519 | 0.0 | rejected: NN |

The selected 64-source 50-step EMA checkpoint is:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_50_v1/checkpoint_step_00000050.pt
SHA-256:
acd2cd7b298961f9b0b80fc4004b7fd1bdf78531592c1e7c3e8202577545ab5a
```

Selector reports:

```text
smoke32:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_smoke32_v2.json
status:
  diagnostic_checkpoint_selected

val64:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val64_v1.json
status:
  diagnostic_checkpoint_selected
NN-W1 delta:
  -0.003287582641064324
volume-W1 delta:
  -0.0015413750587506825

val128:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val128_v1.json
status:
  no_eligible_checkpoint
NN-W1 delta:
  +0.021104979598597695
volume-W1 delta:
  +0.004166020493344594
reason:
  volume_w1_non_inferior=false under max_volume_w1_delta=0.0
```

Interpretation:

```text
The failure mode is now more specific than "need more capacity".  Broader
generated-state replay support plus a smaller EMA update dose can pass smoke32
and val64, while 100/200 steps already over-update NN.  However strict
volume-W1 non-inferiority is still unstable by val128.  The active root-cause
hypothesis is replay support/on-policy coverage plus update-dose selection,
not 58M/98M model size.
```

Do not promote the 64-source 50-step checkpoint to production.  It is the best
current diagnostic candidate only.  The next bounded experiment should either
increase on-policy replay support under the same 34M model with an even more
explicit selection grid, or predeclare a statistically meaningful paired
non-inferiority margin before judging volume-W1.  It should not start Stage-F
or capacity scaling.

### 2026-07-24 update: 64-source 25-step dose

To test whether the 50-step candidate was already too large an update, a
25-step EMA checkpoint was trained on the same 64-source replay cache, with the
same base model, replay entries, target-exclusion contract, optimizer family
and evaluator.  This was a dose diagnostic only, not a new architecture or a
production run.

```text
checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_25_v1/checkpoint_step_00000025.pt
SHA-256:
be6502201e45f9c148eca62e267b99e94688d084ef7ee87d1208acd0eaa07e7e
training status:
  passed
clean_retention_loss_ratio_max:
  0.6652758184518932
final_parameter_update_norm:
  6.36599659641117
all_final_role_terminal_gradient_groups_nonzero:
  true
forbidden source ID check:
  count=773
```

Evaluation reports:

```text
smoke32:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_64src_25_eval_smoke32_v1.json
NN-W1 delta:
  +0.011847870611836897
volume-W1 delta:
  -0.003934128688430516

val64:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_64src_25_eval_val64_v1.json
NN-W1 delta:
  +0.0018497076198800144
volume-W1 delta:
  -0.001064961873843237

val128:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_64src_25_eval_val128_v1.json
NN-W1 delta:
  +0.014133542065700389
volume-W1 delta:
  +0.001539324195554817
```

Hard validity did not regress in these panels, and all replay-role total losses
were lower than base.  The selector reports were:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_smoke32_v3.json
status:
  diagnostic_checkpoint_selected
selected:
  64src_25_ema

/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val64_v2.json
status:
  diagnostic_checkpoint_selected
selected:
  64src_50_ema_val64
reason:
  both 25-step and 50-step were eligible, but the selector tie-break chose the
  lower NN-W1 delta: 50-step -0.0032876 vs 25-step +0.0018497

/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val128_v2.json
status:
  no_eligible_checkpoint
reason:
  both 25-step and 50-step violated strict volume_w1_non_inferior=false under
  max_volume_w1_delta=0.0
```

The 25-step run reduces the val128 volume drift relative to 50-step
(`+0.001539` vs `+0.004166`) while keeping NN drift small and hard validity
stable.  It does not remove the strict zero-margin val128 failure.  The current
root-cause statement is therefore:

```text
Missing composition_counts caused the historical adapter volume drift.
Full shape residual caused the counts-fixed oracle_ca NN/shape tail.
A-v2 generated-state replay is the right diagnostic direction and update dose
matters, but 34M replay retention is still sitting on a narrow rollout-level
volume non-inferiority boundary.  Stage-E v1 remains blocked.
```

Do not use the 25-step checkpoint as production.

### 2026-07-24 update: 10/15/20-step dose window

The smaller-dose follow-up has now been run on the same 64-source replay cache,
with the same base checkpoint, optimizer family, seed, evaluator and
target-exclusion contract.  These runs only changed the number of update steps.

```text
10-step checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_10_v1/checkpoint_step_00000010.pt
SHA-256:
e49d01c1a67d7b3fb64e090703ea1e55bf4ce16c10b18a6f1459ec9fd55e8ee3
clean_retention_loss_ratio_max:
  0.6535009824720732
final_parameter_update_norm:
  4.178169844414849

15-step checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_15_v1/checkpoint_step_00000015.pt
SHA-256:
d2fc6d1bb1c7b9088d329e84901f9b3c08c1a42fe7f3e9654dafac3f046da6ca
clean_retention_loss_ratio_max:
  0.6654225055069591
final_parameter_update_norm:
  5.077290425434707

20-step checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_20_v1/checkpoint_step_00000020.pt
SHA-256:
2d0921d7da0c5ed3364fc9641dc553b0c03c3c217ea6b7abbaae7c96f31a6c26
clean_retention_loss_ratio_max:
  0.6657369364399568
final_parameter_update_norm:
  5.789911179140174
```

All three training audits passed: final role terminal element/lattice/coordinate
gradient groups were nonzero, parameters updated, and the forbidden-source ID
check still covered 773 IDs.

The 10-step checkpoint was evaluated on smoke32, val64 and val128:

| panel | NN-W1 delta | volume-W1 delta | hard validity | selector status |
| --- | ---: | ---: | --- | --- |
| smoke32 | +0.001246 | -0.000618 | unchanged | rejected: clean_clean replay loss not lower |
| val64 | -0.001313 | -0.000123 | unchanged | rejected: clean_clean replay loss not lower |
| val128 | +0.002923 | +0.000934 | unchanged | rejected: clean_clean replay loss and volume |

The val128-only middle-dose panel is:

| steps | all replay role losses lower | clean_clean loss delta | NN-W1 delta | volume-W1 delta | hard validity |
| ---: | --- | ---: | ---: | ---: | --- |
| 10 | no | +0.000272 | +0.002923 | +0.000934 | unchanged |
| 15 | yes | -0.000434 | +0.010864 | +0.000786 | unchanged |
| 20 | yes | -0.000819 | +0.011899 | +0.001154 | unchanged |
| 25 | yes | -0.001068 | +0.014134 | +0.001539 | unchanged |
| 50 | yes | -0.004706 | +0.021105 | +0.004166 | unchanged |

The updated selector reports are:

```text
smoke32:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_smoke32_v4.json
selected:
  64src_25_ema

val64:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val64_v3.json
selected:
  64src_50_ema_val64

val128:
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val128_v4.json
status:
  no_eligible_checkpoint
```

Interpretation:

```text
There is no measured 64-source 34M EMA dose that satisfies both parts of the
strict selector on val128.  The smallest dose preserves rollout metrics best
but does not lower clean_clean replay loss.  Once clean/replay losses all move
in the right direction, val128 volume-W1 remains strictly positive.  This is
not evidence for immediate 58M/98M capacity scaling; it is evidence that the
current replay support/update-dose setup lacks a robust retention window under
the zero-margin volume rule.
```

The next admissible action is either to broaden/on-policy refresh replay support
under the same 34M model, or to predeclare a statistically meaningful paired
volume non-inferiority margin before judging another bounded diagnostic.  Do
not retrospectively loosen the selector, do not start Stage-F, and do not launch
capacity training before the 34M evidence is stable beyond val128.

### 2026-07-24 update: 128-source support probe

The next support probe stayed on the same 34M base and same tensor-free sampler.
Before running it, the generated-state replay audit/training runners were made
role-microbatch aware so a 128-source cache can be audited and trained on 24 GB
GPUs.  This is a runner implementation fix only: it adds
`--max-graphs-per-role-batch`, defaults to the old full-role behavior, slices by
graph, and scales chunk losses by graph fraction so the role-average loss scale
is unchanged.

Code commits:

```text
d69e80b8 fix: microbatch generated-state replay runners
d484fdae fix: match replay runner import ordering
```

Server verification:

```text
PYTHONPATH=src:. /home/workspace/lrh/miniconda3/envs/gaugeflow/bin/python -m pytest -q tests/test_generated_state_replay.py
  21 passed

/home/workspace/lrh/miniconda3/envs/gaugeflow/bin/ruff check scripts/audit_generated_state_replay_training_contract.py scripts/train_generated_state_replay_correctness.py tests/test_generated_state_replay.py
  All checks passed

PYTHONPATH=src:. /home/workspace/lrh/miniconda3/envs/gaugeflow/bin/mypy scripts/audit_generated_state_replay_training_contract.py scripts/train_generated_state_replay_correctness.py tests/test_generated_state_replay.py
  Success
```

The new replay cache:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_128_real_v1/
entries:
  512 = 128 sources x 4 roles
selection_seed:
  6101
source_start_index:
  96
reverse_steps:
  4
manifest SHA-256:
  6e7cfd853b6a3ee1464b31ddfd623df89ed00bc0c1a2c7bef3805708ceee2283
forbidden source ID check:
  count=773
```

Training-contract audit passed with role chunks:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_128_real_v1/training_contract_audit.json
max_graphs_per_role_batch:
  64
clean_retention_loss_ratio_to_max_generated:
  0.3016592524713397
all_role_terminal_gradient_groups_nonzero:
  true
```

Two small-dose 34M diagnostics were trained on this cache:

```text
15-step:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_128src_15_v1/checkpoint_step_00000015.pt
SHA-256:
  c903523e446f4f4df973e7fdc3e0c6b6fd80e695573f9bdcc2f7896aab76c859
clean_retention_loss_ratio_max:
  0.5973009881913536
final_parameter_update_norm:
  5.064493047689135

25-step:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_128src_25_v1/checkpoint_step_00000025.pt
SHA-256:
  e0054559728b00097c589e4134b520445ed743b3163124c1b29ba9c1992ad42d
clean_retention_loss_ratio_max:
  0.639684125160313
final_parameter_update_norm:
  6.526418000036793
```

Val128 results:

| cache | steps | all replay role losses lower | NN-W1 delta | volume-W1 delta | hard validity |
| --- | ---: | --- | ---: | ---: | --- |
| 128-source | 15 | yes | +0.004080 | +0.001527 | unchanged |
| 128-source | 25 | yes | +0.010673 | +0.002142 | unchanged |

Selector:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_128src_checkpoint_selection_val128_v1.json
status:
  no_eligible_checkpoint
```

Interpretation:

```text
Broadening to the next 128-source permutation window does not open a strict
zero-margin volume-retention window.  It keeps replay losses and hard validity
moving correctly, and NN drift is smaller than the old 64-source 25/50-step
tail, but volume-W1 remains positive on val128.  The failure is therefore not
explained by the 64-source cache being too small in a simple way.
```

The next minimum evidence-backed step is not 58M/98M capacity scaling.  It is
either an on-policy replay refresh that uses carriers from the current
candidate rather than only frozen Stage-C base carriers, or a predeclared paired
statistical volume non-inferiority rule.  Stage-E remains blocked.
