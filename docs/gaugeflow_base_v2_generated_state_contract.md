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

The checkpoint-selection protocol for the 34M generated-state replay
correctness run is implemented in
`scripts/select_generated_state_replay_checkpoint.py`.  The first report:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32src_checkpoint_selection_v1.json
```

selects the 32-source 100-step EMA checkpoint:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_32src_100_v1/checkpoint_step_00000100.pt
SHA-256:
8b9bbd2cd30216b7801282f58af85e52c9742fac9a6f3b353eb5ac8e9ffa5a16
```

This is a diagnostic candidate only.  The bounded 64-sample validation was run
with the same evaluator, frozen target panel and random-stream policy:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_32src_100_eval_val64_v1.json
```

It retained NN and hard-validity behavior but failed the strict predeclared
volume non-inferiority rule:

```text
NN-W1 delta: +0.003945617583326899
volume-W1 delta: +0.0007898849176624784
distance-valid delta: 0.0
sampling failures delta: 0.0
terminal masks delta: 0.0
exact composition delta: 0.0
finite-positive lattice delta: 0.0
```

The selector report is:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_32src_checkpoint_selection_val64_v1.json
status: no_eligible_checkpoint
```

Therefore the 100-step EMA checkpoint remains diagnostic and is not promoted.
The next A-v2 step broadened generated-state replay support to 64 sources while
keeping the 34M model, loss, optimizer, seed family and evaluator fixed:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64_real_v1/
entries: 256 = 64 sources x 4 roles
manifest SHA-256:
bd10fa96d0175fa799da075906a18cb96fcffb609d9e3df63c5bea9dfcdfe11f
training_contract_audit:
  passed
```

The 64-source dose result is:

```text
50-step EMA smoke32:
  NN-W1 delta +0.01968915038580843
  volume-W1 delta -0.008498153915681483
  selector status diagnostic_checkpoint_selected
100-step EMA smoke32:
  NN-W1 delta +0.05308763983682652
  selector rejected by NN non-inferiority
200-step EMA smoke32:
  NN-W1 delta +0.08841968069719286
  selector rejected by NN non-inferiority
```

The selected 50-step EMA checkpoint:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_50_v1/checkpoint_step_00000050.pt
SHA-256:
acd2cd7b298961f9b0b80fc4004b7fd1bdf78531592c1e7c3e8202577545ab5a
```

It passed the strict 64-sample selector but not the 128-sample selector:

```text
val64:
  NN-W1 delta -0.003287582641064324
  volume-W1 delta -0.0015413750587506825
  status diagnostic_checkpoint_selected
val128:
  NN-W1 delta +0.021104979598597695
  volume-W1 delta +0.004166020493344594
  status no_eligible_checkpoint
```

The follow-up 25-step EMA dose on the same 64-source cache was:

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
```

Its paired evaluator results were:

```text
smoke32:
  NN-W1 delta +0.011847870611836897
  volume-W1 delta -0.003934128688430516
  selected by /home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_smoke32_v3.json

val64:
  NN-W1 delta +0.0018497076198800144
  volume-W1 delta -0.001064961873843237
  eligible, but selector kept 50-step because its NN-W1 delta was lower

val128:
  NN-W1 delta +0.014133542065700389
  volume-W1 delta +0.001539324195554817
  rejected by strict volume non-inferiority
```

Thus reducing the update dose from 50 to 25 improves the val128 volume drift
but does not make the candidate eligible under the predeclared zero-margin
volume rule.  The 25-step checkpoint is diagnostic only.

The 10/15/20-step dose window was then filled in without changing cache,
optimizer family, seed, base checkpoint or evaluator:

```text
10-step checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_10_v1/checkpoint_step_00000010.pt
SHA-256:
e49d01c1a67d7b3fb64e090703ea1e55bf4ce16c10b18a6f1459ec9fd55e8ee3

15-step checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_15_v1/checkpoint_step_00000015.pt
SHA-256:
d2fc6d1bb1c7b9088d329e84901f9b3c08c1a42fe7f3e9654dafac3f046da6ca

20-step checkpoint:
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_64src_20_v1/checkpoint_step_00000020.pt
SHA-256:
2d0921d7da0c5ed3364fc9641dc553b0c03c3c217ea6b7abbaae7c96f31a6c26
```

All three training audits passed with nonzero final role terminal gradient
groups and the same 773-ID forbidden-source check.  The updated val128 selector
report:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_64src_checkpoint_selection_val128_v4.json
status:
  no_eligible_checkpoint
```

Val128 dose table:

| steps | all replay role losses lower | clean_clean loss delta | NN-W1 delta | volume-W1 delta |
| ---: | --- | ---: | ---: | ---: |
| 10 | no | +0.000272 | +0.002923 | +0.000934 |
| 15 | yes | -0.000434 | +0.010864 | +0.000786 |
| 20 | yes | -0.000819 | +0.011899 | +0.001154 |
| 25 | yes | -0.001068 | +0.014134 | +0.001539 |
| 50 | yes | -0.004706 | +0.021105 | +0.004166 |

Hard validity, exact composition, finite-positive lattice, sampling failures
and terminal masks did not regress in these val128 runs.  The active contract
conclusion is that no measured update dose simultaneously satisfies all-role
replay improvement and strict zero-margin val128 volume retention.

The next support probe used a new 128-source replay cache over the following
permutation window, still under the same 34M model and frozen sampler:

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
```

The runner had to be made microbatch-capable for this cache size:

```text
d69e80b8 fix: microbatch generated-state replay runners
d484fdae fix: match replay runner import ordering
```

This does not change model or loss semantics; `--max-graphs-per-role-batch`
defaults to full-role behavior and only chunks graph batches.  Server
verification passed for pytest, ruff and mypy on the replay runner files.

The 128-source contract audit passed:

```text
max_graphs_per_role_batch:
  64
clean_retention_loss_ratio_to_max_generated:
  0.3016592524713397
all_role_terminal_gradient_groups_nonzero:
  true
```

Val128 diagnostics:

| cache | steps | all replay role losses lower | NN-W1 delta | volume-W1 delta |
| --- | ---: | --- | ---: | ---: |
| 128-source | 15 | yes | +0.004080 | +0.001527 |
| 128-source | 25 | yes | +0.010673 | +0.002142 |

Selector report:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_128src_checkpoint_selection_val128_v1.json
status:
  no_eligible_checkpoint
```

Thus the simple broader-support probe did not fix the strict val128 volume
retention failure.  This strengthens the current contract interpretation:
replay direction and hard validity are correct, but the zero-margin
rollout-level volume rule has no robust overlap yet with replay-loss
improvement.

An opt-in retention-record diagnostic was added next:

```text
b0ea2a3 feat: record paired replay retention diagnostics
```

The default evaluator path is unchanged.  With
`--record-free-generation-samples`, the evaluator stores paired base/candidate
sample records and reports a paired generated-sample bootstrap against the same
fixed validation reference distribution.  Server pytest, ruff and mypy passed
for the modified evaluator and replay tests.

Record-mode val128 reruns exactly matched the existing aggregate-only JSONs:

| cache | steps | all replay role losses lower | agg NN delta | agg volume delta | NN 95% CI | volume 95% CI |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 64-source | 10 | no | +0.002923 | +0.000934 | [-0.002316, +0.007145] | [+0.000132, +0.001114] |
| 64-source | 15 | yes | +0.010864 | +0.000786 | [+0.000165, +0.024719] | [-0.000531, +0.001657] |
| 128-source | 15 | yes | +0.004080 | +0.001527 | [-0.001886, +0.009150] | [+0.000125, +0.001797] |
| 128-source | 25 | yes | +0.010673 | +0.002142 | [-0.002022, +0.024236] | [+0.000205, +0.003024] |

This makes the current failure sharper.  The volume drift is not just an
aggregate formatting artifact: three of the four paired bootstrap intervals are
strictly positive for volume-W1, while the one interval crossing zero
(`64-source`, 15 steps) has a positive NN-W1 interval.  There is still no
measured 34M checkpoint that combines all-role replay loss improvement with a
stable val128 retention window.

The simple on-policy support hypothesis was then tested without changing model
capacity, loss, optimizer family or sampler semantics.  The replay cache
builder gained an optional `--carrier-checkpoint` argument:

```text
78b4258 feat: refresh replay carriers from candidate checkpoints
```

When this argument is supplied, `--base-checkpoint` remains the training base
identity and the carrier checkpoint is used only for detached replay-carrier
generation.  A 1-source smoke and a full 128-source refresh from the 128src/15
candidate both passed provenance and training-contract audits.

On-policy refresh artifact:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_128_onpolicy_from_128src15_v1/
manifest SHA-256:
  8248155385c7db5373638465207e5d6cf77a80935ae67eb423c972a5d7a572e8
carrier checkpoint SHA-256:
  c903523e446f4f4df973e7fdc3e0c6b6fd80e695573f9bdcc2f7896aab76c859
```

The resulting 15-step checkpoint:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_128src_onpolicy_from_128src15_15_v1/checkpoint_step_00000015.pt
SHA-256:
  5c65a51808bff7d0143cbdd326a9389bbb6c41a222d842d168f5cfbe3737c3da
```

Val128 comparison:

| candidate | all replay role losses lower | agg NN delta | agg volume delta | NN 95% CI | volume 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen carriers, 128src/15 | yes | +0.004080 | +0.001527 | [-0.001886, +0.009150] | [+0.000125, +0.001797] |
| on-policy carriers from 128src/15, 15-step | yes | +0.004054 | +0.001537 | [-0.001985, +0.009145] | [+0.000119, +0.001806] |

This does not support the simple explanation that frozen Stage-C carriers are
the main bottleneck.  The current 34M evidence points instead to an
update-dose / retention-boundary mismatch under the replay objective.

The update-dose hypothesis was then tested directly.  Lowering learning rate on
the same 128-source cache reduced parameter displacement and reduced rollout
drift, but did not satisfy zero-margin volume retention:

| lr | steps | final update norm | all replay role losses lower | clean loss delta | agg NN delta | agg volume delta | volume 95% CI |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2e-4 | 15 | 5.064493 | yes | -0.000402 | +0.004080 | +0.001527 | [+0.000125, +0.001797] |
| 1e-4 | 15 | 3.024477 | yes | -0.000147 | +0.003632 | +0.001351 | [+0.000146, +0.001546] |
| 5e-5 | 15 | 1.830502 | no | +0.000072 | +0.002677 | +0.000827 | [+0.000125, +0.000953] |

Finally, zero-training interpolations along the 128src/15 update direction
separated optimizer path from update direction:

| alpha | update norm | all replay total losses lower | clean loss delta | agg NN delta | agg volume delta | volume 95% CI |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.0625 | 0.316531 | yes | -0.000149 | +0.000269 | +0.000096 | [+0.000007, +0.000113] |
| 0.125 | 0.633062 | no | +0.000067 | +0.000523 | +0.000191 | [+0.000014, +0.000226] |
| 0.25 | 1.266123 | yes | -0.000001 | +0.000873 | +0.000382 | [+0.000028, +0.000451] |
| 0.5 | 2.532247 | yes | -0.000128 | +0.001905 | +0.000764 | [+0.000060, +0.000900] |
| 1.0 | 5.064493 | yes | -0.000402 | +0.004080 | +0.001527 | [+0.000125, +0.001797] |

The active contract conclusion is now sharper: the current replay-improvement
direction conflicts with free-generation volume retention even at very small
nonzero displacement.  This is not solved by smaller LR, fewer effective steps,
or on-policy carrier refresh.  A future fix must introduce an explicit
retention/trust constraint or keep the replay update out of production.

Multi-GPU 58M/98M capacity training remains deferred.  The next A-v2 step must
address replay support/on-policy coverage or predeclare a statistically
meaningful paired non-inferiority margin before another bounded 34M diagnostic
is run.

A zero-training parameter-block projection then tested whether the measured
128src/15 update conflict is concentrated in broad update groups rather than
requiring another trained adapter.  The committed utility is:

```text
715eb074 feat: project replay checkpoint update blocks
scripts/project_generated_state_replay_checkpoint.py
```

It generated a diagnostic checkpoint with:

```text
element_alpha: 0.0
coordinate_alpha: 1.0
lattice_head_alpha: 0.25
```

Official artifact:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/generated_state_replay_correctness_34m_128src_15_trust_coord1_lattice025_v1/checkpoint_step_00000000.pt
SHA-256:
f003a264bf19f1b7a47683e873acab67d73bc591a5b3eae7966d5d747f12ef1d
```

The official artifact was evaluated under the same val128 record-mode evaluator:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/generated_state_replay_correctness_128src_15_trust_coord1_lattice025_eval_val128_records_v1.json
```

It exactly reproduces the temporary block-combo measurement.  All replay total
losses are lower; clean loss delta is `-0.00017905235290527344`, NN-W1 delta is
`+0.0024332300859237765`, volume-W1 delta is
`-0.000040659420989408446`, and volume bootstrap CI is
`[-0.0007411272712751895,+0.00030002156927976496]`.  Hard validity, exact
composition, finite-positive lattice, sampling failures and terminal masks do
not regress.

This is the first measured trust-projection window that is compatible with
val128 volume non-inferiority, but it is still a diagnostic candidate only.  It
must be checked on an independent support/window before any production use or
capacity scaling.

This contract therefore does not authorize more E-v1 scalar tuning.  If the
official projection fails to reproduce or fails outside the original support,
the replay update remains non-production and Stage-E stays blocked.  Capacity
scaling is allowed only after the generated-state/trust rule is stable under
the same provenance, forbidden-source and paired-retention checks.

The independent support/window check has now been run for the official
projection:

| replay cache | manifest SHA-256 | all replay role losses lower | free NN-W1 delta | free volume-W1 delta | volume 95% CI |
| --- | --- | --- | ---: | ---: | ---: |
| 32-source | `f59f58545bc1dab62664fad39b14806c0ef42e85f3d786c3cbaee78f131e4909` | yes | +0.0024332300859237765 | -0.000040659420989408446 | [-0.0007411272712751895, +0.00030002156927976496] |
| 64-source | `bd10fa96d0175fa799da075906a18cb96fcffb609d9e3df63c5bea9dfcdfe11f` | yes | +0.0024332300859237765 | -0.000040659420989408446 | [-0.0007411272712751895, +0.00030002156927976496] |
| 128-source | `6e7cfd853b6a3ee1464b31ddfd623df89ed00bc0c1a2c7bef3805708ceee2283` | yes | +0.0024332300859237765 | -0.000040659420989408446 | [-0.0007411272712751895, +0.00030002156927976496] |

The free-generation deltas are identical because these evaluations use the
same checkpoint, validation reference, random streams and free-generation
surface; the independent evidence is the replay-cache loss behavior across
three distinct provenance windows.  In all three windows, all four role losses
decrease and hard validity does not regress.

This is enough to stop E-v1 scalar tuning and enter the retraining plan, but it
does not by itself launch a larger model.  The current
`train_generated_state_replay_correctness.py` runner is a 34M Stage-C replay
update runner: it reconstructs the frozen Stage-C backbone from a checkpoint
and therefore cannot train 58M/98M base models or a from-scratch A-v2 product
model.  The next implementation step is an A-v2 retraining runner/protocol that:

1. initializes preregistered 34M/58M/98M model specs without requiring a
   same-shaped Stage-C checkpoint;
2. trains on the clean A-v1 product objective plus provenance-checked
   generated-state replay roles;
3. keeps forbidden-source checks, checkpoint/protocol hashes, exact counts,
   per-role losses and paired free-retention selectors;
4. supports larger effective batch and multi-GPU execution as an execution
   optimization, not as a changed scientific variable;
5. first runs a bounded 34M smoke before any 58M/98M capacity competition.

That first runner now exists:

```text
9be8950 feat: add GaugeFlow-base v2 generated-state smoke runner
8bc5ba9 fix: match A-v2 smoke runner import order
68b2233 test: cover A-v2 generated-state smoke helpers
```

Files:

```text
configs/gates/gaugeflow_base_v2_generated_state_smoke_v1.json
scripts/train_gaugeflow_base_v2_generated_state_smoke.py
tests/test_generated_state_replay.py
```

The smoke initializes each preregistered capacity from its model spec rather
than from a Stage-C checkpoint, accumulates one clean A-v1 product batch plus
four generated-state replay roles into the same optimizer step, checks
forbidden-source IDs and replay manifest hashes, and reports clean/replay
terminal gradient groups separately.

Server verification at `68b2233`:

```text
PYTHONPATH=src:. /home/workspace/lrh/miniconda3/envs/gaugeflow/bin/python -m pytest -q tests/test_generated_state_replay.py
  27 passed

/home/workspace/lrh/miniconda3/envs/gaugeflow/bin/ruff check scripts/train_gaugeflow_base_v2_generated_state_smoke.py tests/test_generated_state_replay.py
  All checks passed

PYTHONPATH=src:. /home/workspace/lrh/miniconda3/envs/gaugeflow/bin/mypy scripts/train_gaugeflow_base_v2_generated_state_smoke.py tests/test_generated_state_replay.py
  Success
```

One-step A-v2 clean+generated-state smoke results:

| candidate | status | parameters | peak CUDA MiB | final update norm | clean terminal groups | replay role terminal groups |
| --- | --- | ---: | ---: | ---: | --- | --- |
| small_34m | passed | 34,284,207 | 7,832.16 | 0.964336 | nonzero | nonzero for all roles |
| base_58m | passed | 57,682,095 | 11,959.72 | 1.259605 | nonzero | nonzero for all roles |
| large_98m | passed | 97,580,719 | 17,183.43 | 1.633221 | nonzero | nonzero for all roles |

Artifacts:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_generated_state_smoke_34m_v1/
/home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_generated_state_smoke_58m_v1/
/home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_generated_state_smoke_98m_v1/
```

This qualifies the A-v2 clean+generated-state training interface and shows all
three capacities fit the current RTX 4090 memory envelope for the bounded
smoke.  It is still not a capacity selection result and not a long training
run.  The next retraining step is to add exact-resume/checkpointing and a
predeclared short-run selector before launching any multi-GPU 2--5k or
full-pass jobs.

That resume/checkpointing precondition is now implemented for the A-v2 smoke
runner:

```text
1d5538c feat: add A-v2 smoke exact resume checkpoints
7fb36ab fix: allow interrupted A-v2 smoke runs
b65fb20 fix: make A-v2 smoke resume deterministic
a7e57c2 fix: preserve optimizer keys in A-v2 checkpoints
```

The runner now writes hash-sidecar checkpoints containing model, EMA,
optimizer, CPU/CUDA RNG state, device generator state, DataLoader generator
state and epoch cursor.  It supports `--resume`, `--checkpoint-every-steps` and
`--stop-step` for interrupted-resume qualification.  The smoke protocol also
contains the preregistered short-run selector contract; this selector is only
for post-training paired evaluation of bounded A-v2 short runs and does not
qualify Stage-E or authorize Stage-F.

Server verification at `a7e57c2`:

```text
/home/workspace/lrh/miniconda3/envs/gaugeflow/bin/ruff check scripts/train_gaugeflow_base_v2_generated_state_smoke.py tests/test_generated_state_replay.py
  All checks passed

PYTHONPATH=src:. /home/workspace/lrh/miniconda3/envs/gaugeflow/bin/mypy scripts/train_gaugeflow_base_v2_generated_state_smoke.py tests/test_generated_state_replay.py
  Success

PYTHONPATH=src:. /home/workspace/lrh/miniconda3/envs/gaugeflow/bin/python -m pytest -q tests/test_generated_state_replay.py
  28 passed
```

Two-step exact-resume CUDA smoke:

```text
temporary protocol:
  /tmp/gaugeflow_base_v2_generated_state_smoke_steps2_a7e57c2.json

interrupted run:
  /home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_generated_state_resume_smoke_34m_steps2_a7e57c2_v1

uninterrupted run:
  /home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_generated_state_full2_smoke_34m_steps2_a7e57c2_v1

resume status: passed
full status: passed
metrics rows: 2 in both runs
final checkpoint SHA-256:
  5a9676312d9120d73cd95c2ebb6671979ee45615fc8adf048910e9ce7468ae11
exact_final_checkpoint_match: true
```

The first attempted exact-resume comparison exposed a real checkpoint bug:
integer optimizer-state keys were converted to strings during CPU-portable
checkpoint serialization.  That made `AdamW.load_state_dict` silently restore a
non-equivalent optimizer state.  Commit `a7e57c2` preserves optimizer keys and
adds a non-empty AdamW-state round-trip test.

With this fix, the next authorized retraining step is a bounded 34M A-v2
2--5k-update short run with periodic checkpoints and the preregistered paired
selector.  The 58M/98M capacity competition remains blocked until the 34M
short run passes clean retention, replay-role and free-generation retention
checks.

The first bounded 34M short-run protocol is now frozen:

```text
15607bb feat: add A-v2 short-run protocol
configs/gates/gaugeflow_base_v2_generated_state_short_run_34m_v1.json
```

It keeps the same seed, optimizer, loss weights, replay cache, forbidden-source
contract and model definition as the smoke, but increases the clean batch to 64
and runs `2000` updates with checkpoints every `250` updates.  The protocol
allows only `small_34m`; 58M/98M remain blocked until 34M passes the selector.

The short-run protocol itself passed the same two-step exact-resume CUDA smoke:

```text
interrupted run:
  /home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_short_run_protocol_resume_smoke_34m_steps2_15607bb_v1

uninterrupted run:
  /home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_short_run_protocol_full2_smoke_34m_steps2_15607bb_v1

final checkpoint SHA-256:
  ec8186831b24e5778a4b4dd242d85eea577764b4bdf7a6ce3175bf4d23c06d9b
exact_final_checkpoint_match: true
peak CUDA MiB: about 13,400
```

The full 34M/2000-step short run completed:

```text
run:
  /home/workspace/lrh/DATA/T2C-Flow/runs/gaugeflow_base_v2_short_run_34m_2000_15607bb_v1

metrics rows: 2000
checkpoints:
  0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000
final checkpoint SHA-256:
  0765f84e05733b4c2a72d806eeb9413a6887280e1870405d780596f6a319a706
peak CUDA MiB:
  17048.096
status:
  passed training-interface checks
```

This training pass is not an A-v2 success.  It only authorizes checkpoint
selection/evaluation.  To evaluate it, the generated-state replay evaluator was
extended to read A-v2 hash-sidecar checkpoints:

```text
037b1e8 feat: evaluate A-v2 generated-state checkpoints
```

Server verification at `037b1e8`:

```text
ruff: passed
mypy: passed
pytest tests/test_generated_state_replay.py: 29 passed
```

The first selector surface used the existing 32-source replay cache and a
32-sample paired free-generation panel.  It evaluated all nonzero checkpoints:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_250_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_500_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_750_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_1000_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_1250_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_1500_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_step_1750_eval32_037b1e8_v1.json
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_eval32_037b1e8_v1.json
```

Summary of candidate-minus-Stage-C deltas:

| step | clean_clean replay loss delta | all generated replay roles lower | free NN-W1 delta | free volume-W1 delta | hard validity deltas |
| ---: | ---: | --- | ---: | ---: | --- |
| 250 | +1.644628 | no | +0.718542 | +0.326028 | unchanged |
| 500 | +1.321868 | nearly, but clean fails | +0.557899 | +0.228987 | unchanged |
| 750 | +1.088746 | no | +0.466493 | +0.146872 | unchanged |
| 1000 | +0.884881 | no | +0.368408 | +0.075038 | unchanged |
| 1250 | +0.724149 | no | +0.254024 | +0.029822 | unchanged |
| 1500 | +0.594618 | no | +0.225156 | +0.000628 | unchanged |
| 1750 | +0.489394 | no | +0.178541 | -0.019747 | unchanged |
| 2000 | +0.392433 | no | +0.115212 | -0.031482 | unchanged |

Selector report:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/gaugeflow_base_v2_short_run_34m_2000_eval32_selection_037b1e8_v1.json
status:
  no_eligible_checkpoint
```

Interpretation: this A-v2 objective/curriculum has the same failure signature
as the late E-v1 diagnostics at a broader base-training level.  It steadily
improves volume and generated-state replay losses, but it does so while
degrading clean-side retention and free-generation local geometry.  Therefore
capacity scaling to 58M/98M is still not authorized.  The next minimal
hypothesis is not "make the model larger"; it is that the generated-state
replay term is over-weighted or insufficiently clean-retentive even in A-v2.
The next experiment should keep the 34M model and exact same replay support,
then reduce replay pressure or add an explicit clean-retention selector before
any larger-capacity competition.
