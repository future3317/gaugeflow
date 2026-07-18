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
- H1b and H2--H6, tensor conditioning, oracle work, relaxation, DFT, and DFPT
  have not started.

Work remains inside H1a coordinate-generator diagnosis. H1b and H2--H6 are
prohibited. Do not add seeds or steps to rescue completed protocols, revive
either failed reciprocal-score residual, or initialize joint training from a
failed coordinate checkpoint. A new mechanism requires a separately frozen
causal or operator qualification first.

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
