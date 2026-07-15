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

## Required execution environment

- Run all tests, training, sampling, and reported benchmarks in **WSL 2**,
  distribution `Ubuntu-22.04`, using the `flowmm-t2c` micromamba environment.
  The exact interpreter is
  `/home/future04/micromamba/envs/flowmm-t2c/bin/python`.
- This environment is verified with `torch 2.5.1+cu124`, CUDA 12.4, and the
  NVIDIA GeForce RTX 4060 Ti. Before a GPU experiment, verify the interpreter
  and device explicitly:

  ```bash
  /home/future04/micromamba/envs/flowmm-t2c/bin/python -c \
    "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

- The unqualified WSL `python3` is not the experiment environment and may not
  have PyTorch installed. The Windows Anaconda interpreter currently has
  CPU-only `torch 2.11.0+cpu`; never use it for reported training, sampling,
  profiling, or regression results. Set `PYTHONPATH="$PWD/src"` after entering
  the repository in WSL.

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
4. **Gate A4 generator substrate.** `configs/gate_a4_generator_substrate_v1.json`
   is a completed, frozen InN/BN audit of the probability path, atom-type
   manifold, decoder, flow target, and joint generator. A4.0 exact-velocity
   closure passed for all four subspaces, so there is no evidence of an
   analytic time-direction, Euler, torus-wrap, or SPD-log/exp defect. The
   fixed endpoint-ID qualification did not pass: type-only decoded
   composition is 0.000 (minimum 0.95), geometry retrieval is 0.5625 (minimum
   0.90), and joint retrieval is 0.625 (minimum 0.90). The result is recorded
   in `reports/gate_a4_generator_substrate_v1/`. Do not resume tensor
   conditioning, extend targets, add conditional losses, or silently replace
   the vocabulary. The `{B,N,In}` type vocabulary is diagnostic-only. Any
   atom-type manifold/decoder or joint-substrate repair must be proposed in a
   new versioned protocol, with a new fixed budget.
5. **A5--A10 substrate repair/audit sequence.** These are completed, versioned
   endpoint-ID-only studies on the frozen InN/BN pair; none starts a tensor
   gate. A5 corrected simplex/quotient paths but did not qualify the substrate.
   A6 replaced repeated-argmax "categorical" decoding with an absorbing
   discrete-flow posterior and sampler; its analytic closure passed but type
   composition reached only 0.75. A7 added a generated (not externally
   conditioned) 119-element composition-count latent and exact count-constrained
   transitions: graph composition reached 1.0, while site assignment still
   failed. A8 repaired an implementation defect in which original-injection
   fields were time-blind for endpoint-ID/raw/direct conditioning; atom accuracy
   improved to 0.59375 but did not qualify. A9 used one fixed
   Beta(1/2,1) source-weighted discrete-path objective and also did not qualify.
   A10 is a read-only, species-aware StructureMatcher audit: A7/A8/A9 match
   rates are 0.4375/0.5625/0.1875, proving the residual issue is a real chemical
   sublattice mismatch, not merely CIF row order. Do not add further sampler or
   loss searches. Any successor must first pass the separate A11.0 unlabeled
   site-orbit audit; it must not use target atom order or target composition.
6. **A11.0 periodic-site identifiability audit.**
   `configs/gate_a11_0_periodic_site_orbits_v1.json` is a completed,
   read-only audit of the frozen InN/BN endpoint-ID geometry. It Niggli
   reduces an all-identical-species copy of each crystal, enumerates proper
   SO(3) and full O(3) periodic site automorphisms, then inspects target
   species only after the site-orbit partition is fixed. Both materials have
   species-mixed full-O(3) site orbits and a fixed-CIF deterministic
   O(3)-scalar ceiling of 0.5. Therefore do **not** start an A11-G
   geometry-only training run or reinterpret fixed-CIF accuracy as an
   attainable deterministic target. `configs/gate_a11_q_exact_assignment_v1.json`
   supersedes the earlier *prepared-only* A11-S/Q contract for future A11-Q
   work. Q0 is a completed read-only test of exact count-constrained chemical
   assignment enumeration, residual automorphism actions, and node-relabeling
   consistency; Q1 and Q2 are **not started**. For the four-site 2+2 InN/BN
   panel, production likelihood is the exact categorical distribution over
   six unique chemical assignments, not Sinkhorn or permutations of duplicate
   species slots. Sinkhorn/Hungarian remain future large-N
   approximation/diagnostic utilities only. Production quotients use only
   proper-SO(3) automorphisms; full-O(3) is diagnostic-only for scalar-decoder
   identifiability and improper operations must not be silently quotiented for
   a rank-three polar tensor. At a partial discrete state, marginalize only
   the state-dependent residual group `Gamma_t = {gamma: gamma y_t = y_t}`.
   Fixed-CIF site accuracy is diagnostic, never an A11-Q gate threshold. Do
   not start Q1 until Q0's manifest passes; if Q1 later passes, propose a
   separate Q2 material panel with distinct proper-SO(3) orbit structures
   before restoring tensor conditioning.
7. **Substrate-v2 decoration qualification (completed; not a historical rerun).**
   The source-pinned v2 raw build supplied the InN/BN structures and proper
   automorphisms. Versions v1/v2/v3 are all retained under
   `reports/substrate_v2_decoration_only_v*/`: v1 is implementation-invalid
   because it did not make a relabelled model forward; v2 is a valid failed
   numerical-equivalence qualification; v3 qualifies only the
   `rbf_vector_invariant_scorer` as a fixed-geometry, supplied-composition
   decoder. On every v3 candidate row it achieves exact proper-SO(3) quotient
   MAP and species-aware periodic matching of 1.0, exact count, zero masks and
   failures, and fresh-forward assignment-law relabel error at most `2.22e-16`.
   The unmodified v3 aggregate manifest remains failed because it incorrectly
   includes deliberately deficient negative controls; see its immutable CSV
   and `promotion_metric_audit.md`, not a rewritten manifest. This result uses
   dense element tokens `0..117`, a distinct mask `118`, float64 tiny-graph
   accumulation, and a fixed bounded categorical score. True counts were
   supplied only as exact-assignment support, so this is **not** a
   composition-generator, tensor-conditioned, or GaugeFlow result. A separate
   Q1 generated-composition protocol may now be proposed; do not silently
   substitute the decoder result for Q1.
8. **Gate B -- coherent representative invariance.** At fixed structure and
   tensor orbit, rotate the input representative and measure velocity
   equivariance, composition/prototype stability, C2ST, MMD, and orbit-error
   distributions.
9. **Gate C -- small method screen.** Use one frozen structural checkpoint,
   one sample budget, and at least three seeds. Advance GaugeFlow only if orbit
   fidelity and representative/cell consistency improve without losing
   validity or degrading the high-symmetry subset.
10. **Gate D -- physical validation.** Freeze the model and ranking rule before
   top-K relaxation, symmetry re-identification, oracle recomputation, and the
   pre-registered DFPT audit.

11. **P5 exact synthetic tensor-control gate (completed; not passed).**
    `configs/gate_p5_exact_synthetic_tensor_control_v1.json` is an
    oracle-free, coordinate-only control over an exact periodic SO(3)-
    equivariant rank-three teacher. Its analytic equivariance and target orbit
    separation passed, but harmonic-alignment coordinate-flow target retrieval
    was `0.0/0.5/0.5` across its three fixed seeds (minimum `0.9`). Do not
    rerun, extend, retune, or describe its large between/within ratios as
    successful control. `configs/gate_p4_matched_production_backbone_v1.json`
    is consequently blocked and must not start. P3's independent decoder
    protocol is separately blocked on the full-O(3) v2 raw build; it must not
    reuse the InN/BN endpoint-ID panel as a held-out qualification.
12. **P5-D0 coordinate-substrate diagnosis and repair (D0.3 completed; not
    sufficient for progression).** P5-D0.1's 64-source fixed-batch failure is not a
    generic inability to memorize: D0.2 memorized one state to `1.84e-11`
    velocity MSE. The diagnosis is structural. The historical
    `coordinate_gauge="absolute"` target contains a graphwise translation
    component, while the historical direction-only PBC backbone is invariant
    to a common fractional translation and discards edge length. Therefore it
    cannot infer that target component across independent sources. Historical
    D0/D0.1/D0.2 code, reports, thresholds, and conclusions remain immutable.
    The successor
    `configs/gate_p5_d0_3_translation_quotient_metric_v1.json`: it defines
    coordinate tangents modulo one graphwise translation
    (`coordinate_gauge="no_drift"`), evaluates translation-aligned periodic
    RMS, and uses the sole production closest-image distance/RBF coordinate
    feature. Its authorized CUDA run passed its *fixed-batch* qualification
    (`8.95e-5` velocity MSE, `0.00483` translation-aligned RMS), confirming
    that the corrected field can memorize the 64 quotient targets. It then
    failed source generalization (`0.1666` unseen aligned RMS) and free-running
    closure (`0.2107` aligned RMS); see
    `reports/gate_p5_d0_3_translation_quotient_metric_v1/`. Therefore D0.3
    does not authorize P5-D1 or amend the P5 failure. Do not rerun or tune it
    without another explicit versioned protocol.
    D0.4 then held exactly those 64 sources fixed but resampled an independent
    Uniform path time for every source and every one of 5,000 updates. It
    failed the complete-trajectory test: mean 33-time-grid velocity MSE was
    `0.00871` (maximum `0.001`), mean teacher-forced aligned RMS was `0.0423`
    (maximum `0.02`), and 100-step free-running aligned RMS was `0.1996`
    (maximum `0.05`), with zero sampling failures. Its attribution is
    time-conditioning/vector-field expression rather than source
    generalization, because no unseen source is used. The frozen failure is at
    `reports/gate_p5_d0_4_fixed_source_full_trajectory_v1/`; stop here and do
    not start P5-D1, harmonic, or real-tensor work.

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
- Do not conflate the preceding SO(3) *condition-orbit* rule with the
  O(3) crystal point-group compatibility rule. A raw physical polar rank-three
  target must be Reynolds-projected over the full crystallographic O(3) point
  group, including improper operations; inversion therefore projects it to
  zero. `configs/tensororbit_jarvis_v2_raw_build_v2_full_o3.json` is the only
  prospective v2 build contract for that corrected target. The materialized
  proper-only v2 cache and all historical v1 artifacts are legacy/read-only
  and may not be silently used for a future oracle or real-tensor benchmark.
- The active integer proposal catalogue must contain finite crystallographic
  orders 1, 2, 3, 4, or 6 only. Do not reintroduce infinite-order shear or
  hyperbolic SL(3,Z) matrices as point-group candidates.
- Infer latent alignment/stabilizer weights from the current noisy generated
  lattice, coordinates, and atom-type state so training and tensor-only
  sampling receive the same information.
- Preserve the distinction between an exact physical zero tensor and the CFG
  null condition. Condition dropout is graphwise and must carry an explicit
  condition-present mask.
- The frozen historical `direct_irrep` has only two Cartesian contractions and
  is not a complete direct baseline. Do not reinterpret its result. Every new
  fair direct comparison must use `CompleteDirectIrrepCoupling`, which retains
  all six `1o` CG paths of `(2x1o + 1x2o + 1x3o) x (0e + 2e)`, and must pass its
  SO(3)-equivariance regression test.
- `direct_irrep_complete_v1` and `harmonic_alignment_v1` are new, untrained
  versioned modes under `configs/gate_h1_harmonic_conditioning_v1.json`. The
  latter uses deterministic SO(3) quasi-Monte-Carlo nodes, not an exact
  quadrature or a substitute for a reported refinement study. Its early
  proper-SO(3)-invariant token, harmonic posterior, and the separate
  `0e/0o/1o/1e` parity prototype are operator-qualified only; no historical
  Gate result may be described as using them. Any causal training successor
  must pre-register grid size/refinement, data, seed, budget, and thresholds.
- The continuous harmonic score theorem is
  `s(R;g x,h e)=s(g^{-1}R h;x,e)`. It is covered by the deterministic
  `harmonic_covariance_audit` and by representative/high-symmetry/zero tests.
  That theorem does not make a finite QMC node set a group: left/right shift
  residuals must be reported rather than hidden or treated as exact posterior
  covariance.
- The production A11-Q0 tiny-panel likelihood remains exact enumeration. The
  new count-partition dynamic program is a tested scalable primitive for a
  future protocol; it must not silently replace Q0/Q1 evidence or evade the
  state-dependent proper-SO(3) residual-group requirement.
- The historical production vocabulary is raw atomic number in 119 logits and
  has an untrained index zero. Never reuse it in a new categorical protocol:
  use `vocabulary.py` dense elements `0..117` and internal mask `118`.
- Use numerically safe norms on quantities that can be exactly zero and test
  both finite forward values and finite backward gradients.
- The current coordinate flow is defined only modulo a graphwise translation.
  It always projects coordinate tangents to zero graphwise mean and every
  message layer receives closest-image distance/RBF features as well as the
  direction. The absolute-coordinate gauge and direction-only backbone have
  been removed from the current runtime; historical commits, not hidden
  fallbacks, preserve their evidence. Endpoint metrics must align one common
  fractional translation per graph.
- Feed path time directly to every message-passing node state. Passing time
  only through a condition encoder query is invalid because several legitimate
  conditioning encoders do not use that query.
- The production atom-type path is Euclidean 119-logit flow plus a final
  `argmax`; it is not a categorical or simplex manifold. A5--A9 add true
  simplex/discrete/type-set paths only for frozen substrate diagnostics. They
  are not tensor-conditioned methods and must not be selected as a replacement
  without a separately authorized protocol.

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
  The protocol/manifest commit is attested in
  `artifacts/tensororbit_jarvis_v2_oracle_qualification_v1/commit_attestation.json`;
  external model pins and qualification remain blocking.
  Do not start GaugeFlow full training from this preparation, and do not use
  PiezoJet as the primary oracle.
- Before external-oracle training, the raw source must be rebuilt through
  `configs/tensororbit_jarvis_v2_raw_build_v1.json` and
  `scripts/build_tensororbit_v2_raw.py`: release URL/version/hash, units,
  per-row Voigt convention, explicit exclusions, ID joins, zero counts and
  target-cache hashes are mandatory. A local inherited cache is not upstream
  provenance evidence.
- Gate A oracle-free diagnostics support debugging but cannot replace
  training-set orbit tensor error from a qualified frozen oracle ensemble.
- Use a species-aware periodic matcher when judging generated endpoint
  structures. CIF atom-row accuracy is diagnostic only: it can disagree with
  a physically equivalent atom permutation, while an unmatched decorated
  structure is a real sublattice failure.
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
