# GaugeFlow contributor instructions

These instructions apply to the entire repository.

## One active baseline

The only runtime is the revised hybrid-diffusion implementation under
`gaugeflow.production`. Do not restore retired continuous-logit flows,
harmonic/Hopf conditioners, FlowMM dependencies, exploratory Gate runners, or
checkpoint compatibility fallbacks. Historical reproduction belongs to Git
tags and `docs/research_iteration_history.md`, not to active dispatch code.

GaugeFlow and PiezoJet remain separate projects. GaugeFlow may consume
versioned data/oracle artifacts but must not import PiezoJet modules.

## Current boundary

- Mathematical/state-space and Cartesian-atlas runtime qualification passed.
- The tensor-free trainer, EMA/checkpoint recovery, and joint reverse sampler
  passed a bounded CUDA software closure; this is not real-data evidence.
- H0 data/catalogue qualification passed. The current representation uses a
  species-free parent carrier plus exact occupational ordering, low-index
  supercells, OPD displacement modes, Kelvin strain, and a bounded residual.
- The complete 675,204-row H1a P1 cache is built and independently qualified.
- Real-data H1a is frozen as failed: joint training learned coarse chemistry
  and lattice statistics but not the train-reference local-geometry
  distribution; one-pass coordinate-only pretraining also missed its frozen
  validation and low-noise endpoint thresholds.
- The joint H1a run did use all 540,164 training structures for 20,000 updates
  (1,280,000 graph presentations, about 2.37 passes). Later 1/4/16/64-state
  panels are mechanism audits, not substitutes for that full-data run.
- Corrected tangent and exact Helmert-quotient audits show full physical rank
  `30/30` but severe anisotropy: condition numbers are `2.3e7--3.5e7`, effective
  rank is about `2.2`, and a one-state exact readout needs an update of
  `2079.20` from an initial norm of `0.80036`. The current attribution is
  optimization geometry plus state-dependent feature learning, not a missing
  coordinate direction, corrupt cache, or probability-path closure failure.
- No external pretrained weights or failed H1a checkpoint initialize the active
  model. "Coordinate-only pretraining" denotes a from-scratch auxiliary
  objective on the qualified train split, not transfer learning.
- The corrected Cartesian-tangent baseline completed one exact pass over all
  540,164 train structures at seed 5705 but failed final/initial validation
  (`0.70396 > 0.5`) and the low-noise endpoint (`0.04207 A > 0.04 A`).
- A zero-step readout-span audit rules out the final global linear head as the
  main limitation. At step 8441 a train-fitted minimum-norm head changes train
  loss by only `-5.23%` and validation loss by `+3.46%`; a validation-label
  oracle head explains only `49.61% < 75%`. All designs are rank `80/80`, but
  validation effective rank is `21.32`. The classification is
  `backbone_span_limited`. Production now retains one compact factorized
  Cartesian angular-moment mechanism: 64-dimensional persistent scalar edge
  state, eight first/second-moment channels, one concatenated linear-time
  segment reduction, and no explicit triplet index. It adds 466,944 parameters
  and is qualified at `489.10 graphs/s / 182.86 MiB` on the RTX 4060 Ti.
- Its exact-one-pass run improves validation ratio to `0.63864`. A causal audit
  then qualifies the equivalent chart `v_tilde=V^(-1/3)v_r`, which reduces the
  ratio to `0.58940` while preserving the fractional path and sampler. The
  low-noise endpoint is `0.040084 A`; the run still fails.
- A degree-three STF extension improves the ratio only to `0.57240` and the
  low-noise endpoint to `0.03938 A`, at lower training throughput. It is not in
  active runtime. The only active operator is the vectorized degree-one/two
  implementation; do not restore cubic, harmonic, or compatibility branches.
- Refreshing persistent edges from current layer state improves the ratio to
  `0.54417` but still fails. Explicit shell-complete TopK triplets (`0.56794`),
  unbalanced induced R=8 (`0.54583`), and fixed-six-iteration balanced induced
  R=8 (`0.53314`) all fail. The balanced branch is causally used but remains
  non-specialized: maximum slot mass `0.19579`, minimum representation rank
  `1.351`, and maximum inter-slot cosine `0.99974`. TopK, induced/R16, matched
  initialization, their active dispatch, and their experiment runners are
  removed. Do not restore them or search slot rank/balance iterations.
- H1b and H2--H6, tensor conditioning, oracle work, relaxation, DFT, and DFPT
  have not started.
- The frozen-checkpoint middle-noise reciprocal attribution is complete and all
  three preregistered checks fail: endpoint retrieval is `0.40315 < 0.75`, the
  low/high normalized reciprocal-residual ratio is `1.05348 < 1.15` with
  `0/5` supporting times, and the frozen low-k probe improves held-out MSE by
  only `0.002257` with low-minus-high `0.002939 < 0.03`. The decision is
  `do_not_implement_reciprocal_carrier`. Do not revive either reciprocal
  residual or add a new reciprocal production carrier.
- The independent Bridge audit reaches the same NO-GO on the earlier
  volume-normalized checkpoint: middle-noise held-out low-frequency explained
  fraction `-0.001368`, low-minus-random `0.000695`, low-minus-graph-token
  `-0.001368`, and low-shell excess over the permutation null `0.007755`.
  Its source hash and relationship to the main audit are recorded in
  `bridge_no_go_synthesis.md`. Do not rerun either low-k protocol.
- The complete all-pair clean-topology v2 audit covers `1.0` of the clean
  coordination mass. Middle-noise clean/noisy soft Jaccard is `0.50413`, hard
  switch fraction is `0.26469`, and the clean oracle improves residual energy
  by `0.10716` while the noisy control gives `-0.00354`. The frozen probe is
  predictive (`AUC=0.87923`, explained fraction `0.61362`) but its direct
  plug-in carrier worsens the residual by `0.04391`. The decision is
  `probe_predictive_but_topology_correction_not_residual_causal`; do not add
  the frozen probe or linear topology carrier to production.
- The unchanged dynamic coordinate model completed one separately frozen
  from-scratch two-pass exposure curve at seed 5705. Validation ratios at
  `0/0.25/0.5/1/2` passes are
  `1.00000/0.73837/0.63348/0.54371/0.49103`. One-to-two-pass improvement is
  `0.096876`, between the preregistered plateau (`<=0.05`) and undertraining
  (`>=0.10`) boundaries, so the decision is `ambiguous`. The one-pass point
  reproduces the archived `0.54417`; H1a remains failed. Do not add another
  pass or seed, and do not use the post-hoc crossing of ratio `0.5` to claim a
  Gate pass.
- The zero-training coordinate sampler comparison is complete on the same
  two-pass checkpoint and a fixed 256-graph validation panel. Probability flow
  at 25/50 NFE meets the latency checks but fails nearest-neighbour Wasserstein
  non-inferiority (`1.02007/0.79848` versus reverse-SDE-100 `0.56626`); 25 NFE
  also degrades the >=0.5 A minimum-distance fraction. All paths are finite
  with zero failures. Retain reverse SDE; do not promote deterministic
  probability flow. Reverse-SDE-50 (`0.56892`, about half the latency) is a
  post-hoc follow-up hypothesis, not a qualified production replacement.
- The independent 512-structure nested-Brownian qualification rejects that
  reverse-SDE-50 hypothesis: its W1-difference UCB95 is `0.05767 A > 0.03 A`
  and its 1%/5% minimum-distance lower tails degrade, despite a `0.4957`
  latency ratio, finite states, zero failures, and non-inferior endpoint RMS.
  Retain reverse-SDE-100 for coordinate-only audits and stop sampler search.
- The exposure-conditioned exact all-pair topology audit is complete with a
  frozen `mixed` decision. The two-pass middle clean-oracle gain is `0.09293`,
  retention versus 0.25 pass is `0.6640`, and all three middle-time bootstrap
  lower bounds remain positive. The effect is time-localized: gain falls to
  `0.04099` at `t=0.4` but remains `0.14203` at `t=0.6`. Probe explained
  fraction rises to `0.65384`, while the fixed learned carrier remains harmful
  (`-0.04325`). This neither authorizes a full ACF branch nor supports more
  exposure as a sufficient repair.
- The subsequent zero-training quotient-Tweedie audit also closes the simple
  staged self-conditioning hypothesis. At `t=0.6` its topology MSE improves
  `31.27%` over the noisy field, but AUC is `0.77003 < 0.8`; more importantly,
  inserting it through the shared clean-oracle Cartesian carrier worsens the
  held-out coordinate residual by `4.95%`, with the full structure-bootstrap
  95% interval below zero. The stronger frozen linear probe is likewise
  non-causal. Do not add ACF, a one-step Tweedie topology branch, or the old
  linear carrier to production.
- Variant-specific optimal ridge carriers remain non-causal, and a matched
  nonlinear pair-to-vector MLP yields only `+0.00537` held-out topology
  increment with a structure-bootstrap interval crossing zero while both
  readouts overfit. Do not add another deterministic topology conversion.
- A separate production audit found that historical coordinate-only training
  corrupted elements and lattice although conditional rollouts fixed both to
  their true values. The repaired clean-side-information contract passes its
  frozen 2,111-step screen: validation ratio `0.49382` versus historical
  `0.73837`, `t=.6` explained fraction `0.39070` versus `0.13024`, rollout RMS
  `0.07684/0.12153 A` from `t=.1/.2`, and zero failures. Future coordinate-only
  training/evaluation must set and verify
  `coordinate_clean_side_information=true`; joint generation must not use it.
  This does not change historical H1a or authorize ACF, H1b-H6, tensor/oracle
  work, relaxation, DFT or DFPT.
- The unchanged repaired task has now completed one exact pass over all 540,164
  training structures and passes every frozen check: validation ratio
  `0.33219`, t=.6 explained fraction `0.63509`, t=.005/.1 endpoint RMS
  `0.03756/0.04919 A`, rollout RMS `0.05123/0.07039 A` from t=.1/.2, and zero
  failures. This qualifies only `p(F|A,L,N)`, where `A` is the clean per-node
  element-token list, not composition counts alone. It is the conditional-coordinate
  substrate. Historical free joint H1a remains failed. Do not rerun retired
  local/topology branches on the old mismatched task.
- J0 confirms that this qualified field materially uses both side modalities:
  at coordinate time 0.5, controlled element and lattice corruption increase
  score MSE by `5.335x` and `5.163x`, and both give `9.939x`. J1 therefore ran
  one seed-5705, 2,111-step independent-clock attribution in the same backbone.
  Its clean/noisy-element/noisy-lattice/diagonal/interior validation ratios are
  `0.47273/0.51407/0.56107/0.57304/0.64015`; clean retention and diagonal
  improvement both pass their frozen bounds. All three clock embeddings have
  finite nonzero gradients. This supports a unified multimodal hybrid
  diffusion but does not isolate clock identity from the changed task mixture
  and 3.9% capacity increase, and does not qualify free joint H1a. The next
  parameter-matched C0/C1/C2 comparison is now complete and fails its specific
  clock-attribution criterion: C2-minus-C0 diagonal/interior intervals cross
  zero, although C2 significantly improves clean and element-only corners.
  J1 remains a successful composite task-mixture intervention, not proof that
  separate clocks caused its noisy/noisy gain.
- The zero-step gradient-geometry audit finds no persistent regime conflict:
  median clip scale is `0.2661 > 0.2`, every pairwise median cosine is positive,
  and no pair reaches the frozen 75% negative fraction. Retain global clipping;
  do not add blockwise clipping, AGC or target-RMS normalization.
- The final substrate is one heterogeneous product-space reverse field over
  `(A,F,L)`, not three modules assembled after E1/L1/M1. The five J1 regimes are
  a finite task-path measure over `(t_A,t_F,t_L)`; their regime index is audit
  metadata, never model input. Joint, conditional, staged and alternating
  generation are sampler paths of this field. The tensor orbit is a shared
  quotient-valued condition, not a fourth diffused state. The exact family has
  a nested-corruption tower identity, but no finite-model consistency loss or
  information-coordinate ablation is authorized before E1/L1/M1 qualification.
- J2 is authorized only after these attribution/gradient results are recorded,
  separately frozen E1 element and L1 lattice reverse heads qualify, and joint
  M1 training qualifies.
  Generated side states must be on-policy at the same reverse-clock time. The
  J1 coordinate-only checkpoint leaves those heads untrained and must not be
  used to fabricate them. Do not choose a hard chain, start free joint
  training, or enter later Gates before these prerequisites are satisfied.

Work remains inside H1a coordinate-generator diagnosis. H1b and H2--H6 are
prohibited. Do not add seeds or steps to rescue completed protocols, revive
either failed reciprocal-score residual, or initialize joint training from a
failed coordinate checkpoint. A new mechanism requires a separately frozen
causal or operator qualification first.

The reciprocal, clean-topology, exposure-conditioned, quotient-Tweedie,
variant-specific carrier and nonlinear pair-conversion audits are complete.
They reject another deterministic local/global feature branch. The matched
clean-side-information screen and exact-one-pass qualification identify the
coordinate task contract as the material repair. The conditional coordinate
substrate is qualified; the free joint generator is not. No result yet
authorizes H1b-H6, tensor condition, oracle, relaxation, DFT or DFPT.

The first clean production integration exposed a Cartesian index-type defect.
The reverse sampler adds a tangent drift to fractional coordinates, so the
only active chart is `v_r=v_f L` and `v_f=v_r L^-1`; the retired `L^T`
covector pullback must never return as a fallback.  After deterministic linear
reductions and a fixed FP32 geometry path, the no-training CUDA qualification
passed at `516.03 graphs/s`, with exact repeat determinism, output cosine
`0.999806`, and loss-gradient cosine `0.997593`.  The authorized one-pass
learning experiment nevertheless failed as recorded above, so work remains
inside H1a diagnosis and no joint initialization is allowed.

## Required environment

Use WSL 2 Ubuntu-22.04 for reported tests, benchmarks, training, and sampling:

```text
/home/future04/micromamba/envs/flowmm-t2c/bin/python
torch 2.5.1+cu124
CUDA 12.4
NVIDIA GeForce RTX 4060 Ti 16 GB
```

```bash
cd /mnt/e/CODE/T2C-Flow/gaugeflow
export PYTHONPATH="$PWD/src"
PY=/home/future04/micromamba/envs/flowmm-t2c/bin/python
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
```

Do not use the Windows CPU-only torch environment for reported results.

## Active data contract

- Large data stay under `E:/DATA/T2C-Flow`; do not copy them into Git.
- The active Alex split contains all 675,204 rows with exact counts
  540,164/67,520/67,520 and no cross-split formula, prototype,
  matcher-envelope, or connected-component overlap.
- H1a cache construction may apply only a certified Niggli `GL(3,Z)` basis
  change. It must fail closed; no unreduced fallback is allowed.
- IDs, source rows, split labels, formulas, prototypes, space groups, structure
  hashes, and Niggli transforms are audit metadata, never denoiser inputs.
- Future tensor validation/test uses source-verified TensorOrbit-JARVIS-v2 only
  after its Gate is authorized.

## Data cleaning

Confirmed corrupt or task-domain-incompatible records are removed at a
versioned dataset boundary, with ID, evidence, and hashes recorded. Never delete
raw source rows, rewrite an archived result, or add a model fallback for bad
data. Do not remove a scientifically valid hard example merely because the
model learns it poorly.

The current parent-occurrence exclusion list is
`configs/data_quality/parent_occurrence_quarantine_v2.json`. Its exclusions are
task-scoped: they do not automatically filter the P1 structure corpus.

## Physics and leakage rules

- The tensor condition and current noisy/generated state may be model inputs.
  Never pass a paired target CIF/lattice/graph, material ID, target space group,
  target stabilizer, endpoint token, or target species mapping to the denoiser.
- Distinguish a physical zero tensor from a missing/null condition.
- A polar rank-three tensor orbit is an `SO(3)` object. Improper `O(3)`
  operations enter crystal compatibility only with explicit parity.
- Descriptor-frame ambiguity groups are not automatically physical stabilizers.
- The Cartesian atlas is a state-dependent finite prior, not a Haar quadrature
  claim. Preserve candidate multiplicity correction, duplicate/order
  invariance, smooth stratum transitions, and the proven 4,032-candidate generic
  path.

## Hierarchical method rules

- The exact space group is a parent prior, not a mandatory terminal symmetry.
- The parent geometry is species-free. Terminal integer elements are a separate
  occupational field with exact stabilizer
  `H_occ={g: a[pi_g(i)] = a[i] for all i}`.
- The child group intersects supercell-preserving parent operations,
  occupational stabilizer, and active OPD stabilizers.
- The current scientific domain is ordered, stoichiometric, commensurate
  crystals with `det(B)<=4`, at most two active OPD branches, and a registered
  bounded residual. Do not represent defects, disorder, partial occupancy,
  large supercells, or thermal ensembles through the residual.
- Use the Cartesian atlas only after a concrete geometry exists. Pre-geometry
  categorical choices use orbit invariants and reachable-child compatibility.

## Implementation rules

- Prefer physically/mathematically equivalent compact representations and
  vectorization: node permutation plus 3x3 rotation instead of dense 3N actions,
  Kelvin coordinates for symmetric strain, shared factorizations for periodic
  CVP, packed stabilizers, and batched reductions.
- Equivalent acceleration must preserve the acceptance set and be covered by a
  reference test. Do not replace exact crystallographic certification by a
  learned, metric-only, or tolerance-widened shortcut.
- Do not add a method until a small diagnostic identifies the mechanism it is
  intended to repair.
- One-off builders/auditors may be archived after their current artifact is
  qualified; production readers and model code remain active.

## Required validation

At minimum run:

```bash
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
$PY scripts/audit_code_redundancy.py
```

Atlas/runtime changes additionally require a no-write CUDA smoke for candidate
counts, finite outputs, reference equivalence, latency, and peak memory.
