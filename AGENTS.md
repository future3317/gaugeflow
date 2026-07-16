# GaugeFlow contributor instructions

These instructions apply to the entire repository.

## Scientific scope

GaugeFlow is a standalone tensor-orbit-conditioned crystal generator. Prefer a
physically and geometrically correct implementation over compatibility with an
old checkpoint or exploratory runner.

- The active design is the revised hybrid-diffusion architecture under
  `gaugeflow.production`.
- Historical Gate A--A11, P5-D0/C0, substrate-v2 and vNext Q0/Q1 code was
  retired at tag `archive/pre-production-cleanup-20260716`. Never copy it back
  as a runtime fallback.
- Archived harmonic code may be used only for the paper's diagnostic reference.
- Keep GaugeFlow and PiezoJet separate. GaugeFlow may consume versioned data and
  oracle artifacts, but must not import PiezoJet modules.
- Do not restore FlowMM as a runtime dependency.

## Required environment

All reported tests, benchmarks, training and sampling use WSL 2 Ubuntu-22.04:

```text
/home/future04/micromamba/envs/flowmm-t2c/bin/python
torch 2.5.1+cu124
CUDA 12.4
NVIDIA GeForce RTX 4060 Ti
```

From WSL:

```bash
cd /mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit
export PYTHONPATH="$PWD/src"
PY=/home/future04/micromamba/envs/flowmm-t2c/bin/python
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
```

Do not use the Windows CPU-only torch environment for reported results.

## Active repository boundary

Keep:

- revised production modules and their tests;
- rank-three tensor, parity, geometry, symmetry and data primitives;
- the complete direct-CG baseline;
- current TensorOrbit-JARVIS-v2 protocols and attestations;
- formal S0.1--S0.4.1 evidence referenced by the manuscript.

Do not accumulate exploratory runners, checkpoint sweeps, profiler traces or
per-Gate reports in the active tree. Summarize a superseded research cycle in
`docs/research_iteration_history.md`, tag the last complete version, then remove
its code/data/report surface.

## Current gate state

- S0.1/S0.2 passed within their mathematical/interface scope.
- S0.3-v1 (24 frames) failed and is frozen.
- S0.4-v1 (weighted 4,032-candidate prior) failed only its frozen latency gate.
- S0.4.1 preserved the prior and passed runtime qualification.
- No production trainer or qualified reverse sampler exists yet.
- S1a training, tensor fine-tuning, oracle promotion, relaxation, DFT and DFPT
  have not started.

S0.4.1 does not authorize tensor work. The next milestone is tensor-free
production reverse-generation qualification.

## Physics and leakage rules

- The tensor condition and current noisy/generated state are model inputs.
  Never pass the paired target CIF, target lattice, target graph, material ID,
  target space group, target stabilizer or target species mapping into the
  denoiser.
- Distinguish a physical zero tensor from a missing/null condition.
- The polar rank-three tensor orbit is an SO(3) object. Improper O(3)
  operations may enter crystal compatibility diagnostics only with explicit
  parity handling.
- Descriptor-frame ambiguity groups are not automatically physical
  stabilizers.
- Do not infer “isotropic tensor” from a single degenerate quadratic covariant.
- The Cartesian atlas is a state-dependent finite prior, not a claimed Haar
  quadrature approximation.

## Implementation rules

- Do not add compatibility fallbacks to retired `flow.py`/`model.py` APIs.
- Production entry points must fail clearly when a required component is not
  implemented; do not dispatch to historical code.
- Keep candidate multiplicity correction, enumeration-order invariance,
  duplicate-expansion invariance and smooth stratum transitions tested.
- Preserve the proven-unique 4,032-candidate generic fast path. Mixed/axial
  paths retain multiplicity-corrected deduplication.
- Avoid adding a new method until a small versioned diagnostic identifies the
  failure mechanism it addresses.
- Do not modify a completed protocol or report in place. Create a successor.

## Required validation for changes

At minimum run:

```bash
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
$PY scripts/audit_code_redundancy.py
```

Atlas/runtime changes also require a no-write CUDA smoke confirming candidate
counts, finite outputs, reference equivalence, latency and peak memory. Never
overwrite S0.3/S0.4/S0.4.1 evidence.

## Future conditioner comparison

After tensor-free S1a qualifies, compare invariant-only, complete direct-CG,
fixed Cartesian nodes and the current stratified atlas with the same backbone,
training budget, data, seeds and common random numbers. Report stratum usage,
effective frame count, posterior-vs-prior KL, condition gradients, target
separation, validity, latency and memory. Complexity earns promotion only when
it produces measurable causal generation benefit.
