# GaugeFlow

GaugeFlow is a standalone implementation of tensor-orbit-conditioned crystal
generation. It does not import FlowMM at
runtime. The former local FlowMM working tree has been removed; the upstream
baseline and the preserved local T2C patch set are documented under
`../legacy_backups/flowmm_local_2026-07-14/`.

## Current experimental status (2026-07-15)

Gate A v1 is a **frozen negative result**, not a partially completed run. The
protocol in `configs/gate_a_v1.json` used eight real training crystals (2--6
atoms, including a physical zero-response example), 400 optimizer steps, one
seed, and matched capacity for all four methods. All four methods generated
without sampling failures, but none reached the pre-registered generated-target
between/within separation threshold of 1.2.

| Method | representative velocity error | generated between/within |
| --- | ---: | ---: |
| `raw_tensor` | 0.35221 | 1.00444 |
| `direct_irrep` | 0.21211 | 1.00937 |
| `stabilizer_pooling` | 0.11765 | 1.00339 |
| `orbit_alignment` (GaugeFlow) | 0.04523 | 1.00664 |

Thus representative-consistent, condition-sensitive velocities did not form
separated decoded generation trajectories. Follow-up A1/A2/A3 causal and
early-branching screens also failed. A4 analytically closes the current
interpolant and sampler, but the endpoint-ID control fails atom-type
generation; the present generator substrate is therefore unqualified. No full
4,000/499/499 training result, external tensor-oracle score, relaxation, DFT,
or DFPT result is claimed.

Two implementation defects were found before restarting the frozen run:

- The integer-action catalogue incorrectly contained infinite-order shear and
  hyperbolic matrices. It now retains only finite crystallographic orders
  1, 2, 3, 4, and 6 (3,480 proposals reduced to 792).
- Exact physical zero tensors exposed a zero-norm backward singularity. Model
  norms now use a finite-gradient `safe_norm`, with regression tests covering
  the zero-response condition.

The finite-order correction makes the construction physically meaningful, but
`stabilizer_pooling` is still the clear throughput bottleneck on this tiny
panel. Runtime and memory cost therefore remain part of the Gate A/C method
decision, not merely an optimization task.

The core condition is a rank-three piezoelectric tensor orbit. The active
``orbit_alignment`` encoder selects and averages over a finite SO(3) orbit of
the tensor. Its latent proper-automorphism weights are estimated from the current
generated periodic state (lattice, coordinates, and evolving atom-type state),
never from the paired target CIF. Thus training and tensor-only sampling have
exactly the same conditioning information. The complete, lossless Cartesian
vector response field ``F_e(n) = e:(n outer n)`` is queried on nearest periodic
bonds, not on selected lattice columns.

The active quotient approximation uses a finite catalogue of integer lattice
*proposals*, projects every proposed action to a proper Cartesian SO(3)
rotation, then scores lattice residual and type-aware periodic self-match at the
current flow state. This yields a differentiable posterior over latent proper
automorphisms, not a claim to recover an exact space group from a noisy
intermediate structure. Cell-basis changes never rotate the tensor directly.
Right tensor-stabilizer actions are implicit: ``rho(R h)e = rho(R)e`` for every
``h`` preserving the requested tensor. Six fixed response probes span the
symmetric strain space and complement local bond queries. Earlier
``double_coset`` checkpoints/configurations are
accepted as a legacy alias for ``orbit_alignment``; they never receive
target-CIF stabilizer metadata.

Before batching, the data path performs a tracked Niggli reduction: lattice
rows change by an integer unimodular basis transform and fractional coordinates
by its inverse, while the tensor stays in its Cartesian physical frame.
Stabilizer utilities retain only proper (determinant +1) rotations; improper
parity operations remain distinct.

## Layout

- src/gaugeflow/tensor.py: tensor conventions, orbit samples, lossless response queries, and directional metrics.
- src/gaugeflow/manifold.py: standalone product crystal flow coordinates.
- src/gaugeflow/model.py: finite-orbit response encoder, Cartesian direct-irrep control, and graph vector field.
- src/gaugeflow/data.py: direct CSV/CIF reader using PyG Data/Batch, independent of FlowMM.
- src/gaugeflow/unit_cell.py: strict Niggli reduction with tracked basis changes.
- src/gaugeflow/stabilizer.py: crystallographic/tensor-stabilizer utilities plus the active state-derived soft-stabilizer estimator.
- src/gaugeflow/flow.py: conditional flow-matching objective and sampler.
- scripts/: training and tensor-orbit sampling entry points.

## Environment and basic use

The currently verified environment is WSL Ubuntu 22.04 with the
`flowmm-t2c` micromamba environment. Run commands from the repository root:

```bash
micromamba activate flowmm-t2c
export PYTHONPATH="$PWD/src"
python -m pytest -q
```

Train the active method with a Cartesian tensor target cache:

## TensorOrbit-JARVIS-v1 data artifact

GaugeFlow retains the historical local artifact under
`data/tensororbit_jarvis_v1/`: 4,998 rows, a 4,000/499/499 split, and
Reynolds-projected Cartesian tensor targets. This is not a qualified
formula-disjoint split: the audit found 165 cross-split formula groups covering
672 rows and 56 structural near duplicates. It is retained unchanged for
reproducibility only. TensorOrbit-JARVIS-v2 is the versioned, formula-disjoint
activation candidate for future validation/test work and is not active yet.
Neither artifact is a runtime dependency on PiezoJet or any predictor
checkpoint. GaugeFlow keeps zero-response crystals as physical negatives and
never emits target-CIF stabilizers as model inputs.

```bash
PYTHONPATH=src python scripts/train.py \
  --train-csv /mnt/e/CODE/T2C-Flow/gaugeflow/data/piezo \
  --split-manifest /mnt/e/CODE/T2C-Flow/gaugeflow/data/tensororbit_jarvis_v1/splits.json \
  --split train \
  --target-cache-dir /mnt/e/CODE/T2C-Flow/gaugeflow/data/tensororbit_jarvis_v1/reynolds_projected_targets \
  --checkpoint checkpoints/gaugeflow.pt \
  --conditioning-mode orbit_alignment
```

Sample from a trained checkpoint using a tensor target file; the sampler does
not accept a target lattice or target graph:

```bash
PYTHONPATH=src python scripts/sample.py \
  --checkpoint checkpoints/gaugeflow.pt \
  --target examples/zero_tensor_orbit.json \
  --num-samples 16 \
  --num-atoms 4 \
  --steps 200 \
  --output outputs/samples.pt
```

The historical v1 artifact is response-norm stratified (4,000 / 499 / 499),
but its split leakage means it must not support validation/test claims.
Training uses square-root inverse-frequency sampling across its five response
strata by default; set `--no-condition-balanced-sampling` for a uniform
ablation. Future v2 validation and test must retain their natural distribution
and report both aggregate and per-stratum metrics.

## Status contract

The new package is the active GaugeFlow path. QR canonicalization, raw
component conditioning and FlowMM are
baselines, not fallbacks. The prepared
JARVIS/GMTNet CSV may be read as an input dataset, but no other project's
Python module or model checkpoint is imported.

Use ``--conditioning-mode orbit_alignment`` for the active finite-orbit model
and ``--conditioning-mode direct_irrep`` for the Cartesian direct-interaction
baseline. The latter uses exact Cartesian tensor contractions (``einsum``),
not spherical-harmonic evaluation or Clebsch--Gordan layers. Classifier-free
guidance is trained with ``--condition-dropout`` (default ``0.1``); a zero
physical tensor remains distinct from the learned null condition.

## Gated experimental execution

Do not start a full 4,000/499/499 experiment from a finite loss alone. Gate A
uses the exact IDs and budget in `configs/gate_a_v1.json` and trains the four
matched modes `raw_tensor`, `direct_irrep`, `stabilizer_pooling`, and
`orbit_alignment`. `scripts/evaluate_gate_a.py` computes the pre-registered
oracle-free supporting checks: condition-shuffle sensitivity, representative
velocity consistency, and generated target separability. The full decision
additionally requires frozen-oracle orbit error and the DFPT micro-audit listed
above.

After all four matching checkpoints exist, run the supporting evaluator with:

```bash
PYTHONPATH=src python scripts/evaluate_gate_a.py \
  --protocol configs/gate_a_v1.json \
  --checkpoint-dir outputs/gate_a_v1/checkpoints \
  --output outputs/gate_a_v1/report.json \
  --device cuda
```

Gate B tests random tensor representatives at fixed tensor orbits using
velocity equivariance and distributional comparisons (C2ST/MMD), Gate C is the
three-seed method screen, and Gate D is the only stage allowed to consume
relaxation/DFPT budget. A finite objective or a smoke sample is never a pass.
