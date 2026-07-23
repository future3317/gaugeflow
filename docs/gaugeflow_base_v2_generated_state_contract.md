# GaugeFlow-base v2 Generated-State Contract

Status: draft implementation contract with tiny-cache provenance and bounded
34M 2k correctness training passed, but smoke32 retention failed.  This is not
a generated-quality or capacity competition result.

Implementation status as of `23cde00`:

- `GeneratedStateReplayEntry` validates role/source compatibility, exact counts,
  partial reveal semantics, lattice positivity, shape subspace membership,
  finite coordinates and forbidden source-ID overlap.
- `GeneratedStateReplayManifest` records deterministic per-tensor payload
  hashes and canonical JSON SHA-256.
- `write_generated_state_replay_cache` writes a manifest plus tensor payload;
  `load_generated_state_replay_cache` reloads with `weights_only=True` and
  rejects stale checkpoint/sampler identities, duplicate cache keys, forbidden
  source IDs and tampered payloads.
- `scripts/smoke_generated_state_replay_manifest.py` covers the four carrier
  roles on a synthetic cache.  The latest server smoke is:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_cache_smoke_v2/
manifest SHA-256:
  a4ab27e0793c30c5a76fb077867828c41b4eb6b3d2096dabd3d23f5f7cfafac1
```

This is still only the synthetic provenance/cache layer.  It did not authorize
34M training by itself; the next required step was a tiny real replay writer.

Real tiny-cache status as of `55cdb62e`:

- `scripts/build_tiny_generated_state_replay_cache.py` now loads the frozen
  Stage-C 30k/global 40523 backbone from its physical continued-pretraining
  checkpoint, plus the frozen `p(C|N)` composition law.
- It selects real Alex P1 source rows, records real source IDs and clean side
  states, then writes four roles per source:
  `clean_clean`, `generated_assignment`, `generated_lattice` and
  `generated_joint`.
- `generated_assignment` and `generated_joint` use sampled model composition
  counts, not clean target counts; `generated_lattice` is generated while
  conditioning on clean assignment tokens.
- A two-source, four-step server smoke passed:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_tiny_real_smoke_v3/
base checkpoint SHA-256:
  8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9
sampler protocol SHA-256:
  587bf38c705bade6034f73a819cc254c188abdb018b97654c3ec545e232388e1
manifest SHA-256:
  c2878dcc8404d5c47bc32f95fe85506a624c1f867fc6b837b0a83afe896e7e6a
entries:
  8 = 2 real source structures x 4 roles
source IDs:
  mp-1007760, mp-1091415
```

This closes the first real cache-provenance smoke.  It is still not a training
result and not a generated-quality claim.  The next completed step was a
replay-cache per-role loss/gradient correctness audit proving that the current
training objective can consume each carrier role without target leakage or
silent dead gradients.

Completed audit requirements:

- load the tiny real cache at
  `/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_tiny_real_smoke_v3/`;
- reconstruct the frozen Stage-C 30k/global 40523 backbone and current
  `TensorFreeHybridDiffusion` training path;
- pack each role into a graph batch without mixing structures;
- run the current product-space denoising loss without changing the model or
  loss definition;
- report finite total, element, coordinate, volume and shape losses per role;
- backprop a weighted role loss and confirm nonzero gradients in the active
  element, lattice and coordinate parameter groups;
- confirm `clean_clean` retention does not immediately explode relative to the
  generated roles;
- save a small JSON report under the server evaluations directory;
- run `pytest`, `ruff` and `mypy` on the touched files before committing.

Training-contract audit status:

- `scripts/audit_generated_state_replay_training_contract.py` consumes the tiny
  real replay cache through the current `TensorFreeHybridDiffusion` objective,
  without changing the model or loss.
- It packs entries by role, verifies role endpoint assignment tokens realize
  `composition_counts`, checks the diffusion's observed composition counts
  match the replay entry, and backpropagates each role loss.
- The server audit passed on the tiny real cache:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_tiny_real_smoke_v3/
  training_contract_audit.json
  training_contract_audit_with_forbidden_panel.json
  forbidden_source_ids_stage_d_stage_e_v1.json
  forbidden_source_ids_stage_d_stage_e_v1.manifest.json

status: passed
entry_count: 8
roles: clean_clean, generated_assignment, generated_lattice, generated_joint
all_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_to_max_generated: 0.61315789912139
forbidden_source_id_check: executed, count=773
base_checkpoint_sha256:
  8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9
manifest_sha256:
  c2878dcc8404d5c47bc32f95fe85506a624c1f867fc6b837b0a83afe896e7e6a
training_contract_audit_with_forbidden_panel_sha256:
  0851830f0feccbc3156f3a826ddd42b0696a84695dcbf9899936fd6913e2c64c
forbidden_source_ids_sha256:
  043de2544a52de76a81b656a27e49365e1d0cb3908b8dcb3d1dbae3affcd9650
```

The forbidden-source-ID panel contains all Stage-D validation/test material IDs
and the frozen 256-sample Stage-E factorial target panel.  The Stage-E target
panel is a subset of Stage-D validation under
`configs/gates/stage_e_e1a_factorial_rollout_v2_data_clean.json`
(`stage_e_factorial_unique_target_count=256`), and the combined forbidden set
contains 773 unique IDs.  The overlap-enabled audit proves the tiny replay
cache sources `mp-1007760` and `mp-1091415` do not intersect that panel.

Correctness-training smoke status:

- `scripts/train_generated_state_replay_correctness.py` implements the bounded
  34M replay correctness runner without changing model, loss or diffusion
  semantics.
- It uses equal role weights over `clean_clean`, `generated_assignment`,
  `generated_lattice` and `generated_joint`, accumulates all roles into one
  `ProductionTrainer` optimizer step, and records per-role losses, per-role
  terminal gradient contributions, clean retention and parameter update norms.
- The 20-step server smoke passed:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_train_smoke_v2/
status: passed
steps: 20
entry_count: 8
manifest_sha256:
  c2878dcc8404d5c47bc32f95fe85506a624c1f867fc6b837b0a83afe896e7e6a
all_final_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_max: 2.5471673704374154
first_step_parameter_update_norm: 1.0069770103808358
final_parameter_update_norm: 5.613741470653719
forbidden_source_id_check: executed, count=773
```

- The bounded 34M 2k correctness run also passed on the same replay contract:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_2k_v1/
status: passed
steps: 2000
entry_count: 8
training_metrics_rows: 2000
checkpoint_sha256:
  2935365787b934cfdd58bc8a47a2cf104654cd736b946eb5f493b0223de9e560
all_final_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_max: 5.617618305543426
first_step_parameter_update_norm: 1.0069770103808358
final_parameter_update_norm: 37.47103131901986
forbidden_source_id_check: executed, count=773
```

- The follow-up smoke32 evaluation showed that this 8-entry training run is not
  a production candidate:

```text
EMA checkpoint:
  replay role losses: lower than base for all four roles
  free-generation NN-W1: 0.5384073850 -> 2.0589264243
  free-generation volume-W1: 0.3337970015 -> 0.4414314562
  distance-valid: 1.0 -> 0.9375

Raw checkpoint:
  replay role losses: lower than base for all four roles
  free-generation NN-W1: 0.5384073850 -> 1.9576429188
  free-generation volume-W1: 0.3337970015 -> 0.4852205126
  distance-valid: 1.0 -> 1.0
```

The interface and optimizer path are therefore closed, but the tiny replay
cache is too narrow: it reduces the cached role losses while damaging
short free-generation retention.  This is not an EMA artifact.  Larger model
capacity is still deferred; the next correctness step is broader replay-state
coverage under the same provenance contract.

Broader-cache status as of `25cbde3b`:

- `scripts/build_tiny_generated_state_replay_cache.py` now accepts
  `--forbidden-source-ids` and fails closed when selected source IDs overlap a
  held-out panel.
- It also accepts `--selection-seed`; when set, `start-index/sample-count`
  select a deterministic window from a full-split random permutation rather
  than a contiguous source slice.
- The new source-selection helpers are covered by unit tests for JSON and
  newline forbidden files, contiguous selection, seeded deterministic
  selection, out-of-range rejection and forbidden-overlap rejection.
- A real 32-source cache has passed the same provenance and training-interface
  checks:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32_real_v1/
entries:
  128 = 32 real source structures x 4 roles
selection mode:
  permuted, selection_seed=6101
reverse_steps:
  4
manifest SHA-256:
  f59f58545bc1dab62664fad39b14806c0ef42e85f3d786c3cbaee78f131e4909
forbidden_source_id_check:
  executed, count=773
```

Training-contract audit:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32_real_v1/training_contract_audit.json
status: passed
all_role_terminal_gradient_groups_nonzero: true
clean_retention_loss_ratio_to_max_generated: 0.3621880364977982
```

20-step optimizer smoke:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32_train_smoke_v1/
status: passed
steps: 20
all_final_role_terminal_gradient_groups_nonzero: true
parameters_updated: true
clean_retention_loss_ratio_max: 0.7123302917464417
final_parameter_update_norm: 5.814717134166754
```

The bounded 34M correctness run on this cache has now been completed at several
update doses.  The 2k run passed replay-role training checks but still failed
short free-generation retention.  Shorter EMA checkpoints define the first
diagnostic window that keeps smoke32 free-generation close to base while
lowering all replay-role losses:

```text
32-source 100-step EMA:
  checkpoint:
    /home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_100_v1/checkpoint_step_00000100.pt
  checkpoint SHA-256:
    8b9bbd2cd30216b7801282f58af85e52c9742fac9a6f3b353eb5ac8e9ffa5a16
  free-generation NN-W1:
    0.5384073850 -> 0.5451422579
  free-generation volume-W1:
    0.3337970015 -> 0.3269401460
  distance-valid:
    1.0 -> 1.0

32-source 100-step raw:
  free-generation NN-W1:
    0.5384073850 -> 0.9694130108
  distance-valid:
    1.0 -> 0.96875

32-source 200-step EMA:
  checkpoint:
    /home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_200_v1/checkpoint_step_00000200.pt
  checkpoint SHA-256:
    164dc4277c6fd80274990ff4452731f1cb43b4c7a2ef61e7d27c45a68a03f995
  free-generation NN-W1:
    0.5384073850 -> 0.5710955690
  free-generation volume-W1:
    0.3337970015 -> 0.3258963194
  distance-valid:
    1.0 -> 1.0
  replay role losses:
    lower than base for all four roles

32-source 300-step EMA:
  free-generation NN-W1:
    0.5384073850 -> 0.6697907347

32-source 500-step EMA:
  free-generation NN-W1:
    0.5384073850 -> 0.6661407599

32-source 2000-step EMA:
  free-generation NN-W1:
    0.5384073850 -> 1.0851598509

32-source 2000-step raw:
  free-generation NN-W1:
    0.5384073850 -> 1.4923858893
  distance-valid:
    1.0 -> 0.9375
```

This does not authorize capacity scaling.  The evidence now says replay-role
loss minimization can over-optimize the cached generated states while harming
rollout geometry.  Raw weights are already too aggressive at 100/200 steps, so
the viable diagnostic path is early EMA checkpoint selection with explicit
free-generation retention.  The next A-v2 correctness work is not a larger
model.

## Purpose

GaugeFlow v1 is blocked because the base and Stage-E adapters do not jointly
cover the generated assignment-to-lattice-to-coordinate path.  A-v2 must train
the base product model on explicitly provenance-tracked generated side states
before tensor conditioning is retried.  This contract defines the smallest
allowed A-v2 data interface and validation sequence.

## Probability Object

A-v2 still models the same product law:

\[
p(B,N,C,A,L,F)
=p(B,N)\,p(C\mid N)\,p(A\mid C,B)\,p(L\mid A,C,N,B)\,p(F\mid A,L,N,B).
\]

The v2 change is not a new variable or a new final distribution.  It is a
training-state coverage change: the denoiser must see clean and generated
side-state carriers with explicit provenance.

## Carrier Roles

Every training example must declare one of these roles:

| role | assignment carrier | lattice carrier | coordinate carrier | clean target access |
| --- | --- | --- | --- | --- |
| `clean_clean` | clean `A_c` | clean `L_c` | noisy/clean `F` per task | yes, only as supervised endpoint |
| `generated_assignment` | detached generated `A_g` | clean `L_c` | noisy/clean `F` per task | yes, only endpoint and counts |
| `generated_lattice` | clean `A_c` | detached generated `L_g` | noisy/clean `F` per task | yes, only endpoint |
| `generated_joint` | detached generated `A_g` | detached generated `L_g` | noisy/generated `F` per task | yes, only endpoint |

`A_g` must be count-exact for its declared `C`.  `L_g` must be generated from
the same carrier lineage as `A_g` when the role is `generated_joint`.

## Provenance Fields

Each generated-state row or replay entry must carry:

```text
source_structure_id
source_split
node_count
parent_or_flexible_carrier_id
composition_counts
composition_source
assignment_tokens
assignment_source
assignment_reveal_rank
assignment_reveal_count
lattice_matrix
lattice_source
lattice_log_volume
lattice_log_shape
fractional_coordinates
coordinate_source
coordinate_time
element_time
lattice_time
base_checkpoint_sha256
sampler_commit
sampler_protocol_sha256
random_seed
random_stream_id
generation_step_or_refresh_id
```

Allowed `*_source` values:

```text
clean
sampled_composition
generated_assignment
generated_lattice
generated_joint
replay_cache
```

Any generated-side row with `assignment_source=clean` or
`lattice_source=clean` must fail closed unless its role explicitly allows that
clean side state.

## Replay Cache

The first implementation should use a truncated or periodically refreshed
replay cache, not full reverse sampling inside every optimizer step.

Required cache key:

```text
(
  source_structure_id,
  role,
  base_checkpoint_sha256,
  sampler_commit,
  sampler_protocol_sha256,
  refresh_id,
  seed,
  coordinate_time,
  element_time,
  lattice_time
)
```

Required cache invariants:

- `composition_counts.sum == node_count`;
- generated assignment observed counts equal `composition_counts`;
- `assignment_reveal_rank` is a permutation within each graph;
- `assignment_reveal_count` is compatible with `element_time`;
- lattice is finite and has positive determinant;
- projected log-shape remains in the P1 shape subspace;
- fractional coordinates are finite before terminal wrapping;
- clean endpoint fields are stored separately from generated carrier fields;
- no cache entry may be reused with a different checkpoint or sampler commit.

## Loss Sequencing

Do not add every objective at once.  The first three A-v2 experiments are:

1. `A-v1 objective` under the current frozen implementation.
2. `A-v1 + four-role generated-state exposure`.
3. `A-v1 + generated-state exposure + legal covariance pairing`.

Only after these pass small validation may a one-way path-distillation
regularizer be considered.  If used, it must be labelled as path distillation,
not a strict tower identity, unless the less/more states are generated from a
formally nested corruption process.

## Legal Covariance Pairing

Allowed augmentations must be exact physical equivalences:

- proper rotations \(R\in SO(3)\);
- bounded unimodular lattice basis changes \(B\in GL(3,\mathbb Z)\);
- site permutations \(P\).

The transformation must act consistently on lattice, fractional and Cartesian
coordinates, tensor representation, typed noise, assignment tokens, parent
action metadata and all modality clocks.  Property tests must verify periodic
distances, composition, assignment quotient, tensor representation and inverse
transform consistency.

## Small 34M Validation Gate

Before any 58M/98M capacity run, the 34M A-v2 prototype must pass:

- finite forward/backward on every carrier role;
- nonzero gradients for element, lattice and coordinate paths in active roles;
- exact generated composition counts;
- no generated-side target leakage;
- replay-cache checkpoint hash enforcement;
- finite positive lattice in short rollout;
- distance-valid fraction does not immediately regress;
- clean-side retention on `clean_clean`;
- per-role metrics reported separately;
- zero overlap with Stage-D validation/test and Stage-E factorial targets.

The first 2--5k update run is a correctness run, not a final capacity result.

## Retraining Plan Boundary

The retraining plan in `../../重训计划.md` is the fallback after the Stage-E v1
diagnosis is frozen, not permission to restart the full pipeline immediately.
The authorized order is:

```text
freeze E-v1 evidence
  -> broaden A-v2 generated-state replay coverage
  -> 34M correctness and retention validation
  -> predeclared 34M/58M/98M capacity competition
  -> formal A-v2 training
  -> B/C-v2 only after A-v2 improves free-joint generation
  -> E-v2 tensor adapter only after the base path is qualified
```

Batch size, model width and GPU count may be increased only after the data
contract and rollout-retention checks are stable.  Larger batch/multi-GPU runs
must keep the same generated-state provenance fields, forbidden-source checks,
checkpoint-hash enforcement, sampler protocol identity, evaluation targets and
predeclared checkpoint-selection rule.  Higher throughput is an execution
optimization, not a scientific variable.

The immediate A-v2 implementation remains the broader 34M replay cache:

- treat the 32-source 100/200-step EMA checkpoints as diagnostic candidates,
  not production checkpoints;
- require a predeclared checkpoint-selection rule that includes free-generation
  retention, not replay-role loss alone;
- run the next bounded 34M correctness diagnostic with explicit intermediate
  checkpoints or a shorter update budget around the 100--200 step region;
- build a 64-source cache only if the 32-source run shows retention is no
  longer immediately broken or if the 32-source evidence is too noisy to decide.

## Capacity Competition

Only after the 34M contract passes may 34M/58M/98M be compared.  They must use:

- identical data contract;
- identical effective token or graph budget;
- identical random-flow policy where feasible;
- identical validation panels;
- predeclared Pareto-minimax or non-inferiority checkpoint selection.

The historical 4221-vs-final A1 result is evidence for checkpoint selection by
validation rule, not for using the last checkpoint by default.

## Non-Goals

This contract does not authorize:

- Stage-F;
- tensor adapter E-v2 training;
- B/C-v2 physical transfer;
- relaxation, DFT or DFPT;
- claiming Stage-E pass;
- reusing target/clean metadata as generated carrier state;
- changing thresholds after looking at the validation panel.

## Next Implementation Step

Add or use a checkpoint-selection protocol for the 34M generated-state replay
correctness run that evaluates intermediate checkpoints against both replay
roles and smoke32 free-generation retention.  The first target region is the
32-source 100--200 step EMA window.  Multi-GPU 58M/98M capacity training remains
deferred until 34M has replay-role improvement and non-degraded free-generation
retention under that predeclared selection rule.
