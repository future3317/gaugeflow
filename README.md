# GaugeFlow

GaugeFlow is a research implementation of tensor-orbit-conditioned crystal
generation. The active code follows the revised hybrid-diffusion design and
uses a **Stratified Cartesian Gauge Atlas with residual descriptor-frame
marginalization**. It does not use the retired continuous-logit flow as a
fallback.

The project now has a tensor-free production trainer and joint reverse sampler.
Their bounded S1a-I0 software closure passed on CUDA. The first H0 data
activation audit did not pass, so real-data H1a/H1b training has not started.
Consequently, the repository does
not claim successful tensor-conditioned generation or target-separated sample
distributions.

## Current status

| Component | Status |
|---|---|
| Categorical absorbing-mask process | Implemented and unit tested |
| Wrapped periodic coordinate quotient | Implemented and unit tested |
| Volume/shape lattice chart | Implemented and unit tested |
| Cartesian STF geometry queries | Implemented and unit tested |
| Stratified Cartesian Gauge Atlas | Implemented and numerically qualified |
| Equivariant hybrid denoiser | Implemented as a model primitive |
| Symmetry compatibility router | Implemented; S1a uses leakage-free P1 blueprints, not a full 230-group/Wyckoff sampler |
| Parent--distortion--child hierarchy | Mathematical/code contracts implemented; no mode catalogue, parent-pair corpus, or hierarchical training is qualified |
| TensorOrbit-JARVIS-v2 data protocol | Built and audited for future external-oracle qualification |
| Production trainer, EMA and checkpoints | Implemented; S1a-I0 closure passed |
| Joint reverse sampler | Implemented; S1a-I0 closure passed |
| H0 data activation | Frozen as `H0_not_passed_stop_before_H1` |
| Tensor-free real-data H1a and full-blueprint H1b | Not authorized |
| Real tensor fine-tuning/oracle/DFT/DFPT | Not authorized |

The condensed no-training evidence is:

- S0.1/S0.2: mathematical, symmetry-chart and software-interface checks passed.
- S0.3-v1: the 24-frame-only atlas failed and is frozen. It must not be restored.
- S0.4-v1: the weighted `24 x 7 x 24 = 4,032` Cartesian prior passed scientific
  checks but failed its frozen CUDA latency limit (`41.89 ms > 20 ms`).
- S0.4.1: the same 4,032-candidate prior qualified at `14.62 ms/forward` and
  `15.19 MB` on an RTX 4060 Ti. This does not reclassify S0.4-v1 or start S1a.
- S1a-I0 v1--v1.2: frozen failed trainer/sampler closure attempts that exposed
  the raw lattice-score instability.
- S1a-I0 v1.3: clean-lattice production closure passed; scientific real-data
  S1a remains unrun.

## Active model definition

The intended generated state is

```text
(masked element tokens, wrapped fractional coordinates, lattice volume/shape)
```

The revised model is assembled from:

1. `AbsorbingMaskDiffusion` for discrete atom types;
2. `AdaptiveWrappedQuotient` or `ScalableWrappedQuotient` for periodic
   fractional coordinates;
3. `PointGroupMetricChart` and `LatticeVolumeShape` for the lattice;
4. `CartesianSTFGeometryQueryEncoder` for condition-free angular geometry;
5. `StratifiedCartesianGaugeAtlas` for rank-three tensor-orbit conditioning;
6. `HybridCrystalDenoiser` for the shared equivariant backbone;
7. `TerminalGroupCompatibilityRouter` for terminal-group diagnostics and
   `ReachableChildCompatibilityRouter` for parent-to-child path marginalization;
8. `ParentBlueprint`, `DistortionBlueprint`, `ModeCatalog` and
   `ChildReconstructor` for the versioned low-index commensurate
   parent--distortion--child extension.

The tensor-free objective uses clean-token prediction for the categorical
state, a wrapped quotient score for coordinates, and clean-state prediction
for lattice log volume/log shape. Clean lattice prediction avoids the
high-noise `1/alpha(t)` inversion that failed the frozen S1a-I0 v1--v1.2
closures; it is not a clipping or Cholesky-jitter fallback.

The atlas defines a state-dependent finite discrete measure rather than a Haar
quadrature approximation. Generic states use 4,032 weighted candidates. Axial
and descriptor-isotropic strata use multiplicity-corrected residual rules and a
smooth partition of unity. Physical zero tensors bypass directional alignment;
a nonzero tensor is never discarded merely because one quadratic descriptor is
isotropic.

The complete direct-CG baseline remains in `gaugeflow.direct_irrep`. It is a
future matched baseline, not the production conditioner.

## Symmetry breaking without discarding the exact parent generator

The exact space-group blueprint is now interpreted as a **parent** prior, not a
claim that the final child must retain that space group. The versioned
hierarchical design factors generation into an ordered parent followed by a
sampled low-index commensurate distortion:

```text
ParentBlueprint + parent hybrid diffusion
  -> DistortionBlueprint(B, k, irrep, OPD, active)
  -> ModeDiffusionState(amplitudes, invariant strain, bounded residual)
  -> ChildReconstructor
```

The v1 code enforces `det(B) <= 4`, at most two active modes, OPD selection
before continuous amplitude diffusion, child-group intersection, mass-weighted
mode reconstruction and a fail-closed 0.10 Angstrom residual RMS budget. The
exact branch is `d = empty`, so there is no duplicate legacy generator.

Tensor compatibility is evaluated on deduplicated physical reachable-child
path classes with an explicit base-measure mass. Catalogue tuple multiplicity
and ordering cannot change the prior. It is not a hard parent-space-group filter: a centrosymmetric
parent remains available when an inversion-odd distortion reaches a compatible
polar child. The full Cartesian atlas is reserved for mode/strain/residual
denoising after a parent geometry exists; discrete parent/path decisions use
orbit invariants and child-compatibility residuals.

See [`docs/hierarchical_symmetry_breaking_v1.md`](docs/hierarchical_symmetry_breaking_v1.md).
The original Chinese design/data note is retained as
[`docs/method_update_and_dataset_usage_zh.md`](docs/method_update_and_dataset_usage_zh.md).
These interfaces do not authorize hierarchical training. The first formal H0
activation audit is frozen as `H0_not_passed_stop_before_H1`; H1a/H1b and all
later gates remain unauthorized.

## Repository layout

```text
src/gaugeflow/production/   revised hybrid and hierarchical model primitives
src/gaugeflow/tensor.py     rank-three tensor conversions and response probes
src/gaugeflow/parity.py     SO(3)/O(3) parity rules
src/gaugeflow/stabilizer.py proper/full point-group utilities
src/gaugeflow/data.py       TensorOrbit crystal dataset loader
src/gaugeflow/direct_irrep.py complete direct-CG baseline
scripts/                    production train/sample, current data and audit entry points
configs/                    current generation and TensorOrbit-v2 protocols
reports/tensororbit_*/      current data activation evidence
reports/h0_data_activation_v1/ current hierarchical data-activation evidence
docs/                       current design and condensed iteration history
tests/                      active production, physics and data regressions
```

Historical Gate A--A11, P5-D0/C0, substrate-v2 and vNext Q0/Q1 files were
retired after their lessons were consolidated in
[`docs/research_iteration_history.md`](docs/research_iteration_history.md).
Their exact source and reports remain available at Git tag
`archive/pre-production-cleanup-20260716`.

## Required environment

Use WSL 2, Ubuntu-22.04, and the existing `flowmm-t2c` micromamba environment:

```bash
cd /mnt/e/CODE/T2C-Flow/gaugeflow
export PYTHONPATH="$PWD/src"
PY=/home/future04/micromamba/envs/flowmm-t2c/bin/python

$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The qualified machine reports torch `2.5.1+cu124`, CUDA 12.4, and an NVIDIA
GeForce RTX 4060 Ti. The Windows Anaconda environment is CPU-only and must not
be used for reported experiments.

Install/update the editable package if needed:

```bash
$PY -m pip install -e '.[dev]'
```

## Validation

Run the active suite:

```bash
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
$PY scripts/audit_code_redundancy.py
```

The redundancy audit checks the production modules and current train, sample,
and data entry points
for duplicate normalized bodies, unreachable branches, unused private
definitions, stored-but-unread attributes, constant branches and unused CLI
arguments.

Superseded S0 runners, harmonic/Hopf reference code, intermediate configs and
per-run reports are intentionally absent. Their exact state is recoverable from
Git tag `archive/pre-runtime-cleanup-20260717`; the manuscript and
`docs/research_iteration_history.md` retain the scientific conclusions.

## TensorOrbit-JARVIS-v2

The current data path is source-verified TensorOrbit-JARVIS-v2. Relevant
artifacts include:

```text
data/tensororbit_jarvis_v2/
data/tensororbit_jarvis_v2_full_o3_v2/
artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/splits.json
artifacts/tensororbit_jarvis_v2_raw_build_v1/attestation.json
artifacts/tensororbit_jarvis_v2_full_o3_v2/attestation.json
```

Future validation/test and external tensor-oracle qualification must use a
versioned v2 protocol. The retired v1 preprocessed cache is intentionally absent
from the active tree.

Data build/audit entry points are:

```bash
$PY scripts/build_tensororbit_v2_raw.py --help
$PY scripts/audit_tensororbit_v2_build.py --help
$PY scripts/prepare_v2_oracle_qualification.py --help
$PY scripts/audit_alex_mp20_source.py --help
$PY scripts/audit_h0_activation.py --help
```

The trainer and sampler below are implementation entry points, not current
authorization to run H1. After H0 passes, run them only under a versioned H1a
protocol:

```bash
$PY scripts/train_production.py --csv /path/to/train.csv \
  --split-manifest /path/to/splits.json --split train \
  --output outputs/s1a_tensor_free

$PY scripts/sample_production.py \
  --checkpoint outputs/s1a_tensor_free/checkpoint_step_00100000.pt \
  --output outputs/s1a_samples --num-samples 100
```

These entry points never enable a tensor condition or read a target space
group. They use a training-split node-count prior and a P1 blueprint. They do
not authorize oracle promotion, relaxation, DFT or DFPT.

## Development rules

- Do not reintroduce the old continuous-logit `flow.py`/`model.py` implementation.
- Do not restore archived harmonic code or audit runners as runtime fallbacks.
- Keep a physical zero tensor distinct from a missing condition.
- Use SO(3) for the polar rank-three tensor orbit and O(3) only for crystal
  compatibility diagnostics where parity is explicit.
- H0 must pass before H1a starts. H1 requires both the P1 real-data H1a
  generator and the full 230-space-group/Wyckoff H1b generator to pass.
- Any atlas simplification must be a new versioned method. The failed 24-frame
  approximation cannot be reused.
- Matched conditioner comparisons must share the same backbone, data, budget,
  seeds and sample noise.

## Next implementation milestone

The next milestone is completion of H0, not training. Alex-MP-20 contains
675,204 structurally valid rows, but its upstream split has 15,621 train--val,
15,524 train--test and 4,278 val--test reduced-formula overlaps. GaugeFlow must
freeze a formula/prototype-disjoint child split first. H0 also requires the
remaining PhononDB derivation attestations, a frozen MatPES-PBE teacher, a
deduplicated OPD physical path measure and the bounded parent-decomposition
pilot. Only then may H1a and H1b start. Tensor conditioning remains H6.
