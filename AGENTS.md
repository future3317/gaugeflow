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
- Harmonic/Hopf diagnostic code and frozen audit runners live only in Git tag
  `archive/pre-runtime-cleanup-20260717`; do not restore them to the package.
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
cd /mnt/e/CODE/T2C-Flow/gaugeflow
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
- current TensorOrbit-JARVIS-v2 reports and protocols referenced by future
  data/oracle work.

Do not accumulate exploratory runners, checkpoint sweeps, profiler traces or
per-Gate reports in the active tree. Summarize a superseded research cycle in
`docs/research_iteration_history.md`, tag the last complete version, then remove
its code/data/report surface.

## Current gate state

- S0.1/S0.2 passed within their mathematical/interface scope.
- S0.3-v1 (24 frames) failed and is frozen.
- S0.4-v1 (weighted 4,032-candidate prior) failed only its frozen latency gate.
- S0.4.1 preserved the prior and passed runtime qualification.
- The tensor-free production trainer, EMA/checkpoint recovery and joint reverse
  sampler passed the bounded S1a-I0 implementation closure in v1.3.
- Real-data S1a training, the full space-group/Wyckoff blueprint sampler,
  tensor fine-tuning, oracle promotion, relaxation, DFT and DFPT have not
  started.
- The parent--distortion--child mathematical/code contracts are implemented.
  `h0_data_activation_v1` remains frozen as a failed historical audit.
  `h0_data_activation_v4` has qualified H0-A, H0-B and H0-C: the Alex child-first split is
  formula-, exact-prototype-, matcher-envelope- and component-disjoint, and
  PhononDB v2 has a full-universe Hessian-algebra certificate plus a frozen
  1,024-material long-tail/stratified mode audit; frozen TensorNet and QET
  MatPES-PBE-2025.2 teachers pass the 512/32 held-out and covariance audit.
  H0-D-v1 is frozen failed. H0-D-v2 now qualifies the complete abstract
  finite-affine catalogue for all 230 groups and all 6,188 canonical
  `det(B)<=4` HNF orbits: 53,441 physical-real irreps and 75,416 OPD classes,
  with an independent exhaustive structural audit and deterministic algebra
  rebuild panel. H0-E-v1 evaluated concrete occurrence in the direct sum of
  compact displacement and six-dimensional Kelvin strain representations. All
  125 discovered candidates qualified, but candidate coverage was
  `125/1024 = 0.12207 < 0.15`; the independently audited gate is frozen failed.
  The overall decision remains `H0-E-v1_failed_stop_before_H1`.
  Its active-tree runners were retired after commit
  `f6f0262bfe9bbd983213467b20e66bce5fcb8485`; use that commit to reproduce
  the frozen v1 artifact. The H0-E-v2 E1a maximal-t parent-occurrence protocol
  is frozen on 64 v1 no-candidate rows but has not yet run. It may not relabel
  the old spglib tolerance ladder. E1a failure stops E1b and H1a; E1a success
  permits only a separately frozen maximal-k E1b protocol.
  H0-E-v2 E0 has qualified a source-hashed, independently certified
  maximal t/k-subgroup and Wyckoff-splitting embedding catalogue: 3,744 raw
  source rows reduce to 2,843 rational affine embeddings without multiplicity
  bias. E0 authorizes only the bounded parent-occurrence E1 pilot; H0-E and H0
  remain unqualified and H1a remains unauthorized.
  H1a/H1b and H2--H6 have not started. The P1 tensor-free substrate is named
  `ParentBlueprintBatch`.

S0.4.1 and S1a-I0 do not authorize tensor work. H0-E-v1 failed and is frozen;
no later H gate may run until a separately proposed, versioned H0-E successor
addresses the observed parent-candidate coverage mechanism. H1a remains
unauthorized.

## Hierarchical symmetry-breaking rules

- Treat the sampled exact space group as a parent group, not automatically the
  final child group.
- Never hard-reject a parent using piezoelectric compatibility. Marginalize
  compatibility over versioned reachable child paths and apply the Reynolds
  residual to the terminal child group.
- The v1 distortion domain is ordered, stoichiometric and commensurate with
  `det(B) <= 4`, at most two active OPD branches and a registered bounded
  residual. Do not smuggle defects, disorder, partial occupancy or large
  supercells through the residual head.
- Sample an OPD/isotropy branch before diffusing its reduced continuous
  amplitude. Do not allow noisy irrep coordinates to change the child subgroup.
- Use the Cartesian atlas only after a concrete parent geometry exists, for
  mode/strain/residual conditioning. Parent/path categorical decisions use
  orbit invariants and reachable-child compatibility.
- H0 data qualification, real-data H1a, full-blueprint H1b, H2 mode supervision, H3
  reconstruction, H4 PES supervision, H5 tensor-free hierarchy and H6 tensor
  conditioning are strictly ordered. H1 passes only after H1a and H1b pass. A
  later stage may not run after an earlier failure.

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
- Do not infer an isotropic tensor from a single degenerate quadratic covariant.
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
- Offline OPD construction uses the exact fixed-space-lattice algorithm,
  compact permutation-plus-3x3 displacement actions, generator-complete
  homomorphism checks and packed stabilizers. Do not replace it with dense
  `3N x 3N` actions or full subgroup enumeration in the production builder.
- Parent-occurrence acceleration must preserve the independently certified
  HNF, site mapping, periodic quotient and OPD contracts. Metric-only parent
  projection and blind symmetry-tolerance expansion are rejected candidate
  sources: both failed the frozen real-material diagnostic and must not return
  as runtime fallbacks.
- Avoid adding a new method until a small versioned diagnostic identifies the
  failure mechanism it addresses.
- Completed exploratory protocols belong in Git history and the condensed
  iteration document, not as executable compatibility paths in the active tree.

## Required validation for changes

At minimum run:

```bash
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
$PY scripts/audit_code_redundancy.py
```

Atlas/runtime changes also require a no-write CUDA smoke confirming candidate
counts, finite outputs, reference equivalence, latency and peak memory. The
frozen S0.3/S0.4/S0.4.1 artifacts must be read from the archive tag, not copied
back into the active tree.

## Future conditioner comparison

After tensor-free S1a qualifies, compare invariant-only, complete direct-CG,
fixed Cartesian nodes and the current stratified atlas with the same backbone,
training budget, data, seeds and common random numbers. Report stratum usage,
effective frame count, posterior-vs-prior KL, condition gradients, target
separation, validity, latency and memory. Complexity earns promotion only when
it produces measurable causal generation benefit.
