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

Work remains inside H1a coordinate-generator diagnosis. H1b and H2--H6 are
prohibited. Do not add seeds or steps to rescue completed protocols, revive
either failed reciprocal-score residual, or initialize joint training from a
failed coordinate checkpoint. A new mechanism requires a separately frozen
causal or operator qualification first.

The middle-noise oracle curve, score-residual reciprocal-shell spectrum, and
frozen low-k linear probe have now rejected a reciprocal global carrier. A
future H1a protocol must separately preregister an attribution involving
conditional target variance, data exposure, probability-path information, or
staged/self-conditioned coordinate generation; do not silently turn this
diagnosis into architecture search.

The clean-topology audit and fixed-model 0.25/0.5/1/2-pass curve are complete.
The only next bounded attribution currently supported is a zero-training
exposure-conditioned topology residual persistence audit on those frozen
checkpoints, using the exact v2 pair panels and carrier definitions. It should
ask whether clean-oracle gain disappears with exposure. It must not add an
optimizer step, seed, data pass, production topology branch, H1b-H6, tensor
condition, oracle, relaxation, DFT or DFPT.

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
