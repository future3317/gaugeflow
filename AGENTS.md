# GaugeFlow contributor instructions

These instructions apply to the entire `gaugeflow` repository. GaugeFlow is a
standalone tensor-orbit-conditioned crystal generator. Prefer a physically and
geometrically correct implementation over a cheaper surrogate merely because
the paper text or an old checkpoint describes the surrogate. When code, paper,
and physics disagree, document the mismatch, implement the justified method,
test it at small scale, and only then update scientific claims.

## Project boundary

- Keep GaugeFlow and PiezoJet as separate codebases. GaugeFlow may consume the
  frozen TensorOrbit-JARVIS data artifact and shared conversion/audit formats,
  but must not import PiezoJet modules or depend on its evolving weights.
- PiezoJet is not the primary GaugeFlow oracle. A future PiezoJet checkpoint
  may be one member of an independently qualified frozen ensemble only after it
  passes the same pre-registered qualification criteria as other candidates.
- Do not restore FlowMM as a runtime dependency. Historical FlowMM patches live
  under `../legacy_backups/flowmm_local_2026-07-14/` and are baselines only.
- The model input is a tensor condition plus the current generated state. Never
  use the paired target CIF, target graph, target lattice, target stabilizer, or
  target space group to construct a training-only conditioning path.

## Experimental gates

Execute the research program strictly in order:

1. **Gate A -- conditioning works on a tiny real panel.** Use the frozen IDs,
   methods, seed, and budgets in `configs/gate_a_v1.json`. Compare
   `raw_tensor`, `direct_irrep`, `stabilizer_pooling`, and `orbit_alignment`.
2. **Gate B -- coherent representative invariance.** At fixed structure and
   tensor orbit, rotate the input representative and measure velocity
   equivariance, composition/prototype stability, C2ST, MMD, and orbit-error
   distributions.
3. **Gate C -- small method screen.** Use one frozen structural checkpoint,
   one sample budget, and at least three seeds. Advance GaugeFlow only if orbit
   fidelity and representative/cell consistency improve without losing
   validity or degrading the high-symmetry subset.
4. **Gate D -- physical validation.** Freeze the model and ranking rule before
   top-K relaxation, symmetry re-identification, oracle recomputation, and the
   pre-registered DFPT audit.

Do not start the full 4,000/499/499 run while Gate A is unresolved. A finite
training loss, a smoke sample, or a completed checkpoint is not evidence that a
gate passed. Record negative results and failures; do not silently tune a
pre-registered threshold after seeing outcomes.

As of 2026-07-15, Gate A v1, A1, A2 S1, and A3 are archived negative results.
Gate A's four methods all fail generated-target separation; GaugeFlow's
representative velocity consistency alone is not sufficient evidence of
conditional generation. A4 analytically validates the present probability
path and sampler, but its endpoint-ID substrate control fails, so neither
tensor-conditioned work nor any full 4,000/499/499 run may resume. A5--A10
are audit-only substrate diagnostics; A11.0 ruled out its geometry-only gate
as unidentifiable, while A11-Q0 is only a no-training exact-assignment
mathematical check. See `README.md` and the separate `gaugeflow_perf_audit`
repository for frozen protocols and evidence.

## Physical and conditioning invariants

- Treat the rank-three piezoelectric tensor as a Cartesian physical tensor.
  An integer unit-cell basis change updates lattice rows and fractional
  coordinates; it does not directly rotate the tensor.
- Convert lattice-action proposals to proper Cartesian SO(3) actions before
  acting on tensors. Do not pool improper rotations or parity operations into
  a proper-rotation orbit.
- The active integer proposal catalogue must contain finite crystallographic
  orders 1, 2, 3, 4, or 6 only. Do not reintroduce infinite-order shear or
  hyperbolic SL(3,Z) matrices as point-group candidates.
- Infer latent alignment/stabilizer weights from the current noisy generated
  lattice, coordinates, and atom-type state so training and tensor-only
  sampling receive the same information.
- Preserve the distinction between an exact physical zero tensor and the CFG
  null condition. Condition dropout is graphwise and must carry an explicit
  condition-present mask.
- Keep `direct_irrep` a genuine Cartesian equivariant direct-interaction
  baseline. Its exact tensor contractions do not require spherical harmonics or
  Clebsch--Gordan layers; do not weaken it into raw component concatenation.
- Use numerically safe norms on quantities that can be exactly zero and test
  both finite forward values and finite backward gradients.

## Evidence and reporting

- Do not call the historical TensorOrbit-JARVIS-v1 4,000/499/499 split
  formula-disjoint: its audit found cross-split formula groups and structural
  near duplicates. Preserve v1 unchanged for reproducibility. Only the future
  versioned v2 activation protocol may support validation/test claims, after
  its formula-group, StructureMatcher, response-stratum, zero-tensor, ID-join,
  and hash audits pass.
- Gate A oracle-free diagnostics support debugging but cannot replace
  training-set orbit tensor error from a qualified frozen oracle ensemble.
- Report representative/cell consistency and distributional behavior, not only
  single-sample tensor error. Include validity and the high-symmetry subset as
  guardrails.
- Include throughput, memory, sample budget, seeds, failed structures, oracle
  abstentions, relaxation failures, and DFPT failures in experimental records.
- Never claim unrun relaxation, DFT/DFPT, oracle, discovery, or benchmark
  results. Label planned, running, supporting, and gate-passing evidence
  separately.

## Editing and validation

- Preserve the user's dirty worktree. Do not reset, clean, overwrite unrelated
  changes, or commit generated checkpoints unless explicitly requested.
- Put source in `src/gaugeflow`, entry points in `scripts`, protocol files in
  `configs`, and regression tests in `tests`. Keep large outputs under
  `outputs` and data artifacts under `data`.
- When behavior changes, update the corresponding tests and user-facing method
  description. Run the narrow relevant tests first, then the full suite when
  feasible in the WSL `flowmm-t2c` environment.
- Keep experimental configurations explicit and reproducible: material IDs,
  seeds, capacities, steps, sampling budgets, thresholds, and checkpoint paths
  must be recoverable from a versioned config or report.
- Do not optimize away a failed scientific control. Diagnose whether the
  failure is mathematical, physical, numerical, data-related, or purely a
  throughput issue before changing the method or protocol.
