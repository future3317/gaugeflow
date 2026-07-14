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
   Gate A v1 is an archived negative result at
   `artifacts/gate_a_v1_frozen_archive/manifest.json`: never edit its
   threshold (1.2), reports, split, or checkpoints.
2. **Gate A2 S1 -- shared conditional-control mechanism.** The versioned
   successor protocol `configs/gate_a2_conditional_control_v1.json` is
   direct-irrep-only and runs exactly four fixed 400/800-step combinations on
   the v1 eight-ID training panel. It repairs conditional control with a
   condition-free base plus a time-gated, per-message-block FiLM/residual
   field; it does not alter the GaugeFlow alignment posterior. Its completed
   S1 result is `s1_direct_irrep_not_passed` in
   `reports/gate_a2_conditional_control_v1/manifest.json`. Do not rerun,
   retune, extend beyond 800 steps, select CFG=1 as the main result, or start
   S2. S2 remains locked unless a separately versioned protocol is explicitly
   authorized after all S1 criteria pass.
3. **Gate A3 two-target early branching.**
   `configs/gate_a3_early_branching_v1.json` is a completed, frozen two-target
   direct-irrep test of early all-negative tangent identification, not another
   residual/FiLM/CFG/counterfactual search. The deterministic pair is InN
   (`JVASP-1180`) and BN (`JVASP-22673`); its result is
   `two_target_not_passed` in
   `reports/gate_a3_early_branching_v1/manifest.json`. Do not extend to 4 or 8
   targets, increase its 400-step budget, or add more conditional modules.
   Inspect the probability path, atom-type manifold, decoder, and flow-target
   definition before proposing any new mechanism. Continuous state separation
   alone is insufficient: use the decoded-state audit and endpoint retrieval.
4. **Gate B -- coherent representative invariance.** At fixed structure and
   tensor orbit, rotate the input representative and measure velocity
   equivariance, composition/prototype stability, C2ST, MMD, and orbit-error
   distributions.
5. **Gate C -- small method screen.** Use one frozen structural checkpoint,
   one sample budget, and at least three seeds. Advance GaugeFlow only if orbit
   fidelity and representative/cell consistency improve without losing
   validity or degrading the high-symmetry subset.
6. **Gate D -- physical validation.** Freeze the model and ranking rule before
   top-K relaxation, symmetry re-identification, oracle recomputation, and the
   pre-registered DFPT audit.

Do not start the full 4,000/499/499 run while Gate A is unresolved. A finite
training loss, a smoke sample, or a completed checkpoint is not evidence that a
gate passed. Record negative results and failures; do not silently tune a
pre-registered threshold after seeing outcomes.

As of 2026-07-14, all four corrected-code 400-step checkpoints and the common
oracle-free evaluator are complete. The supporting status is **failed** because
the GaugeFlow generated-target between/within distance ratio is 1.0066 against
the frozen 1.2 threshold, despite passing condition-shuffle and representative
consistency checks. Do not tune the threshold after seeing this result. The
full gate also lacks a qualified external oracle ensemble and training-panel
orbit-tensor-error distributions. See `README.md` and
`reports/performance_data_scientific_audit.md`.

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

- Do not describe TensorOrbit-JARVIS v1 as formula-disjoint. The audit found
  165 reduced-formula groups crossing splits and 56 cross-split structural
  near-duplicate pairs. Keep v1 unchanged only for historical Gate A panels.
  The formula-disjoint v2 split under `artifacts/` is an inactive candidate;
  its activation protocol is
  `artifacts/tensororbit_jarvis_v2_activation_audit/activation_protocol.json`.
  Every future validation/test or full benchmark must use v2 after a new
  versioned protocol and new checkpoints, never a silently substituted v1
  split. Training may use condition-stratum balancing; validation and test
  retain natural distributions.
- v2 is presently prepared only for independent external tensor-oracle
  qualification under
  `configs/tensororbit_jarvis_v2_oracle_qualification_v1.json`. The GMTNet and
  e3nn SE(3)-Transformer manifests must receive pinned external source commits,
  environment locks, and a committed qualification protocol before training.
  Do not start GaugeFlow full training from this preparation, and do not use
  PiezoJet as the primary oracle.
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
- Exclude diagnostic checkpoints, binary preprocessing caches, and profiler
  Chrome traces from source control. Commit the cache manifest, small summaries,
  tests, and reproducible builder/benchmark scripts.
