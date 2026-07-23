# GaugeFlow-base v2 Generated-State Contract

Status: draft implementation contract, not a training result.

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

Implement only the replay/provenance data layer and its property tests first.
Do not start multi-GPU training until the cache contract can fail closed on
synthetic leakage and stale-checkpoint cases.
