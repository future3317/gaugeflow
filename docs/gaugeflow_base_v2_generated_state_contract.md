# GaugeFlow-base v2 Generated-State Contract

Status: draft implementation contract with tiny-cache provenance audit passed;
not a generated-quality or capacity-training result.

Implementation status as of `2e44df8`:

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

Implement the 34M generated-state replay correctness runner, then run a tiny
8-entry smoke before the 2--5k correctness experiment.  It must consume the same
replay contract, report losses and gradients by carrier role, record clean
retention and generated-state stratification, verify parameter updates, and
remain bounded as a correctness run.  Multi-GPU 58M/98M capacity training
remains deferred until this 34M generated-state contract is proven.
