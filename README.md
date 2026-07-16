# GaugeFlow

GaugeFlow is a research implementation of tensor-orbit-conditioned crystal
generation. The active code follows the revised hybrid-diffusion design and
uses a **Stratified Cartesian Gauge Atlas with residual descriptor-frame
marginalization**. It does not use the retired continuous-logit flow as a
fallback.

The project is currently at mathematical/interface qualification. The
Cartesian conditioner and hybrid-state primitives are implemented and tested;
an end-to-end production trainer and reverse sampler are not yet qualified.
Consequently, the repository does not claim successful tensor-conditioned
generation or target-separated sample distributions.

## Current status

| Component | Status |
|---|---|
| Categorical absorbing-mask process | Implemented and unit tested |
| Wrapped periodic coordinate quotient | Implemented and unit tested |
| Volume/shape lattice chart | Implemented and unit tested |
| Cartesian STF geometry queries | Implemented and unit tested |
| Stratified Cartesian Gauge Atlas | Implemented and numerically qualified |
| Equivariant hybrid denoiser | Implemented as a model primitive |
| Symmetry compatibility router | Implemented as a primitive; a complete blueprint sampler is still absent |
| TensorOrbit-JARVIS-v2 data protocol | Built and audited for future external-oracle qualification |
| Production trainer, EMA and checkpoints | Not implemented |
| Qualified reverse sampler | Not implemented |
| Tensor-free S1a training | Not started |
| Real tensor fine-tuning/oracle/DFT/DFPT | Not authorized |

The formal no-training evidence is versioned as follows:

- S0.1/S0.2: mathematical, symmetry-chart and software-interface checks passed.
- S0.3-v1: the 24-frame-only atlas failed and is frozen. It must not be restored.
- S0.4-v1: the weighted `24 x 7 x 24 = 4,032` Cartesian prior passed scientific
  checks but failed its frozen CUDA latency limit (`41.89 ms > 20 ms`).
- S0.4.1: the same 4,032-candidate prior qualified at `14.62 ms/forward` and
  `15.19 MB` on an RTX 4060 Ti. This does not reclassify S0.4-v1 or start S1a.

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
7. `SpaceGroupCompatibilityRouter` and symmetry expansion primitives for the
   future blueprint-to-structure interface.

The atlas defines a state-dependent finite discrete measure rather than a Haar
quadrature approximation. Generic states use 4,032 weighted candidates. Axial
and descriptor-isotropic strata use multiplicity-corrected residual rules and a
smooth partition of unity. Physical zero tensors bypass directional alignment;
a nonzero tensor is never discarded merely because one quadratic descriptor is
isotropic.

The complete direct-CG baseline remains in `gaugeflow.direct_irrep`. It is a
future matched baseline, not the production conditioner.

## Repository layout

```text
src/gaugeflow/production/   revised hybrid model primitives
src/gaugeflow/tensor.py     rank-three tensor conversions and response probes
src/gaugeflow/parity.py     SO(3)/O(3) parity rules
src/gaugeflow/stabilizer.py proper/full point-group utilities
src/gaugeflow/data.py       TensorOrbit crystal dataset loader
src/gaugeflow/direct_irrep.py complete direct-CG baseline
scripts/                    current data and S0 audit entry points only
configs/                    current S0 and TensorOrbit-v2 protocols
reports/paper_s0_*/         formal paper-facing S0 evidence
reports/tensororbit_*/      current data activation evidence
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
cd /mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit
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

The redundancy audit checks the production modules and current S0 entry points
for duplicate normalized bodies, unreachable branches, unused private
definitions, stored-but-unread attributes, constant branches and unused CLI
arguments.

Official S0 reports are immutable. Their runners refuse to overwrite an
existing result directory. Inspect the committed evidence instead of silently
rerunning a protocol with changed code:

```text
reports/paper_s0_2_scalability_symmetry_chart_v1/
reports/paper_s0_3_cartesian_atlas_v1/
reports/paper_s0_4_cartesian_atlas_prior_v1/
reports/paper_s0_4_1_cartesian_atlas_runtime_v1/
```

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
```

These commands prepare or audit data. They do not authorize GaugeFlow training,
oracle promotion, relaxation, DFT or DFPT.

## Development rules

- Do not reintroduce the old continuous-logit `flow.py`/`model.py` implementation.
- Do not make archived harmonic code a runtime fallback.
- Keep a physical zero tensor distinct from a missing condition.
- Use SO(3) for the polar rank-three tensor orbit and O(3) only for crystal
  compatibility diagnostics where parity is explicit.
- A new trainer must first qualify tensor-free reverse generation before
  enabling the Cartesian conditioner.
- Any atlas simplification must be a new versioned method. The failed 24-frame
  approximation cannot be reused.
- Matched conditioner comparisons must share the same backbone, data, budget,
  seeds and sample noise.

## Next implementation milestone

The next code milestone is a production-only tensor-free trainer and reverse
sampler for categorical, wrapped-coordinate and lattice states. It must expose
checkpoint/EMA/optimizer recovery and decoded crystal metrics. Only after that
S1a substrate passes should the project test whether the Cartesian atlas adds
causal target separation relative to invariant-only, direct-CG and fixed-node
Cartesian baselines.
