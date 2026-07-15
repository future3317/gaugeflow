# GaugeFlow

GaugeFlow is a standalone implementation of tensor-orbit-conditioned crystal
generation. It does not import FlowMM at
runtime. The former local FlowMM working tree has been removed; the upstream
baseline and the preserved local T2C patch set are documented under
`../legacy_backups/flowmm_local_2026-07-14/`.

## Current experimental status (2026-07-15)

The original four-method Gate A v1 is a frozen negative archive, not an
active experiment. No full 4,000/499/499 training result is claimed. The
original four-method 400-step result is frozen at
`artifacts/gate_a_v1_frozen_archive/manifest.json`; its 1.2 separation
threshold, checkpoints, and report are not editable evidence. The frozen v1
protocol is `configs/gate_a_v1.json`: eight real training crystals (2--6
atoms, including a physical zero-response example), 400 optimizer steps, one
seed, and matched capacity for all four methods.

| Method | Gate A status |
| --- | --- |
| `raw_tensor` | corrected-code 400-step checkpoint complete |
| `direct_irrep` | corrected-code 400-step checkpoint complete |
| `stabilizer_pooling` | corrected-code 400-step checkpoint complete |
| `orbit_alignment` (GaugeFlow) | corrected-code 400-step checkpoint complete |

The common oracle-free evaluation **failed** one pre-registered supporting
check. All methods respond to condition shuffling and GaugeFlow improves random
representative consistency (mean velocity error 0.0452 versus 0.3522 for raw
conditioning), but its generated target between/within distance ratio is only
1.0066 against the required 1.2. Gate A v1 therefore failed and remains
frozen. The full
decision additionally requires a pre-qualified frozen external tensor-oracle
ensemble, the training-set orbit-tensor-error distribution, and the registered
physical micro-audit. See `reports/performance_data_scientific_audit.md`.

### Next-generation mathematical implementation (H1; no training)

`configs/gate_h1_harmonic_conditioning_v1.json` records a new, versioned
operator qualification. It does not modify a historical checkpoint or claim a
Gate A result. The implementation adds:

- `direct_irrep_complete_v1`, which retains the six independent polar-vector
  Clebsch--Gordan pathways before message passing;
- `harmonic_alignment_v1`, a state-derived $l\leq3$ relative-alignment score
  on a deterministic SO(3) Hopf quasi-Monte-Carlo grid, paired with an early
  proper-SO(3)-invariant condition token, normalized orbit-shape plus
  log-magnitude/physical-zero decomposition, and an entropy/time alignment gate;
- a tested O(3)-typed prototype with separate $0e$, $0o$, $1o$, and $1e$
  channels; and
- a differentiable count-partition dynamic program, checked against exact
  small-support assignment enumeration.

The H1 CUDA operator audit deliberately treats the 240-node grid only as a
numerical reference: 24/60/120-node aligned-condition differences are
`2.0143/0.5956/0.1724`, respectively. Thus finite grid convergence is an open
pre-registered design choice, not an achieved theorem or a reason to start
training. See `reports/gate_h1_harmonic_conditioning_v1/`.

The continuous harmonic score has now been audited with
`s(R;g x,h e)=s(g^{-1}Rh;x,e)`: its maximum numerical discrepancy is
`3.37e-08`; high-symmetry and physical-zero probes also pass. Crucially, this
does **not** make the 24-node finite grid exactly covariant: its measured
nonidentity left/right nearest-node residuals are `1.807/1.776`. The grid is
therefore still an explicitly refinable approximation, not a finite group or
an exact quadrature. See
`reports/gate_h1_harmonic_conditioning_v1/harmonic_covariance_audit.md`.

### Physical point-group correction for future v2 data only

The previous prospective v2 raw builder used only proper crystal rotations in
its Reynolds step. That is incorrect for **crystal compatibility** of a polar
rank-three piezoelectric tensor: Neumann compatibility is with the full
crystal point group in $O(3)$, so mirrors are retained and inversion projects
the tensor to zero. This does not change the model's SO(3) condition orbit:
improper operations remain excluded from latent representative alignment.

The historical v1 cache, its gates, and its reports remain immutable. The
already materialized v2 proper-only cache is quarantined from future tensor
evaluation; `configs/tensororbit_jarvis_v2_raw_build_v2_full_o3.json` defines
the new schema-3 full-$O(3)$ rebuild under a new output directory. It is
prepared but not yet an activated oracle or training artifact.

### P5 exact synthetic tensor-control gate (completed, not passed)

The oracle-free P5 gate uses an analytic periodic rank-three teacher
`T(X)=sum_(i->j)(a_j-a_i) exp(-r_ij) n_ij^(x3)` on two four-site, triclinic
coordinate endpoints and four SO(3) representatives per orbit. Its teacher
equivariance error is `2.98e-07`, and its two target orbits are separated by
`1.0407`; the synthetic property itself is therefore valid and distinguishable.
Nevertheless the same harmonic production coordinate flow fails exact-property
retrieval at all three frozen seeds (`0.0`, `0.5`, `0.5`, required `>=0.9`).
The large between/within ratios arise from branches that do not retrieve the
requested tensor orbit and are not counted as control. P5 is thus a versioned
negative result, not evidence for real piezo generation. Its failure blocks the
P4 matched six-method comparison; see
`reports/gate_p5_exact_synthetic_tensor_control_v1/`.

`gate_p3_independent_decoder_qualification_v1` and the P4 comparison are
prepared but not run. P3 removes the InN/BN endpoint ID and requires held-out
formula/prototype groups; it is blocked on the corrected full-O(3) v2 build.
P4 is blocked by the failed P5 precondition. The next real-tensor step remains
the separately blocked two-oracle full-O(3) v2 qualification.

### P5-D0 coordinate-flow root-cause repair (D0.3--D0.8 completed; not passed)

The fixed-batch P5-D0.1 negative result has a concrete representation/path
mismatch, not evidence that the coordinate head is disconnected. The old path
uses an **absolute** fractional-coordinate velocity. Its target has a
source-dependent graphwise translation component (26.8% of target velocity
energy in the fixed audit), but the old backbone only observes pairwise
nearest-image **unit directions**. A common fractional translation leaves
those inputs unchanged; it also discards every bond length. D0.2 then showed
that one fixed state can be memorized (`1.84e-11` velocity MSE), which rules
out a broken coordinate-head gradient chain while leaving the multi-source
target unidentifiable.

The repair is implemented. The sole current
coordinate backbone reuses the shared closest-image PBC primitive and appends
a finite-cutoff Gaussian radial encoding of physical Cartesian edge distance
to each equivariant message. The coordinate flow always projects both target
and predicted tangents to zero graphwise mean and evaluates a
translation-aligned periodic RMS. This retains all relative geometry while
removing the arbitrary cell-origin degree of freedom. The erroneous absolute
coordinate gauge and directions-only backbone have been removed rather than
retained as runtime compatibility paths; the frozen Git commits preserve
historical reproducibility. None of the D0, D0.1, D0.2, or P5 results has been
altered or rerun. The authorized D0.3 run passed the narrow fixed-batch
qualification (`8.95e-5` velocity MSE and `0.00483` translation-aligned RMS),
so the corrected model can learn the 64 frozen quotient targets. It did **not**
generalize to 64 unseen sources (aligned RMS `0.1666`) or free-run from a
training source (aligned RMS `0.2107`); it is classified as a
source-coupling-generalization failure, with zero non-finite samples. Thus it
does not authorize P5-D1 or any tensor-conditioned follow-up. The frozen
contract and full result are
`configs/gate_p5_d0_3_translation_quotient_metric_v1.json` and
`reports/gate_p5_d0_3_translation_quotient_metric_v1/`.

D0.4 then tested whether the same 64 training sources can support an entire
trajectory rather than one sampled time per source. It resampled an independent
Uniform path time for every source at every one of 5,000 updates, evaluated 33
fixed times, and free-ran only those same sources. It failed: mean grid velocity
MSE was `0.00871` (required `<=0.001`), teacher-forced aligned RMS was `0.0423`
(required `<=0.02`), and free-running aligned RMS was `0.1996` (required
`<=0.05`), despite zero sampling failures. This isolates the remaining defect
to time conditioning/vector-field expression or trajectory stability, rather
than unseen-source generalization. The result is frozen at
`reports/gate_p5_d0_4_fixed_source_full_trajectory_v1/`; no P5-D1, harmonic,
or real-tensor work is authorized.

One separate, literature-grounded repair was then evaluated as D0.5. Rather
than supervise the source-ambiguous raw velocity at a collapsed endpoint, it
predicts the translation-quotient endpoint residual
\(P\operatorname{Log}_{x_t}(x_1)\), whose target is zero for every source at
the endpoint; its sampler uses the associated bounded bridge contraction. The
coordinate decoder also adds a direct closest-image metric edge-displacement
field to the existing equivariant vector messages. The derivation and sources
are in `reports/p5_d0_endpoint_bridge_rationale.md`. This is a real improvement
but not a pass: residual MSE `0.00113` (limit `0.001`), teacher-forced aligned
RMS `0.0314` (limit `0.02`), and free-running aligned RMS `0.1751` (limit
`0.05`), with zero sampling failures. The pre-registered D0.5 contract
therefore freezes an endpoint-residual-fit failure at
`reports/gate_p5_d0_5_endpoint_bridge_metric_v1/`; it forbids further tuning
or any next gate.

D0.6 then tested the requested **Quotient Rollout-Corrected Flow Map** without
altering D0.5. It predicts a finite quotient displacement from `(s,u)`, uses
only the prescribed two Fourier interval coordinates via FiLM in every message
block, and optimizes exactly the on-path map loss plus one detached-rollout
correction loss. The model now fits local analytic maps extremely well: the
mean adjacent 33-grid map MSE is `3.64e-5` (limit `0.001`). However, direct
long-horizon teacher-forced maps still give aligned RMS `0.0446` (limit `0.02`)
and composition of 100 learned maps gives `0.2327` (limit `0.05`), with zero
non-finite samples. This cleanly localizes the remaining issue to cross-horizon
map composition/rollout stability, not local map fitting. D0.6 is frozen at
`reports/gate_p5_d0_6_quotient_rollout_corrected_flow_map_v1/`; no P5-D1,
harmonic, or real-tensor work is authorized.

D0.7 is a separately frozen long-horizon stability study, recorded under
`reports/gate_p5_d0_7_multiscale_semigroup_flow_map_v1/`. Its read-only D0.6
reconstruction found direct-map MSE rising from `3.92e-5` at span `1/32` to
`1.87e-2` at span `1`, while the semigroup defect rises from `0.00507` to
`0.11102`. A quotient perturbation was amplified `3.56x` in the first learned
step and `1.17x` per step on average. The registered multiscale direct/endpoint,
differentiable rollout, and semigroup losses preserved low adjacent-map MSE
(`3.91e-5`) and zero sampling failures, but failed teacher-forced RMS (`0.1100`)
and 100-step RMS (`0.2248`). The result is a frozen negative result, not a
replacement for D0.6.

The one and only authorized D0.8 follow-up selected the finite-difference
translation-quotient 1-Lipschitz penalty because the D0.7 amplification was
above one; it did not add a global/local hierarchy or EMA distillation. The
penalty reduced the measured first-step amplification to `1.015x` and mean
per-step amplification to `1.033x`, but its fixed equal-weight training run
collapsed map fitting: map MSE `6.19e-5` remained below the local threshold,
while teacher-forced RMS worsened to `0.1256` and 100-step RMS to `0.2474`
(zero failures). Thus neither long-map error nor stability is qualified under
the fixed D0.7/D0.8 capacity, budget, and thresholds. D0.8 is terminal for
this authorization: do not start P5-D1, D0.9, harmonic, oracle, or real-tensor
experiments from these results.

### Gate A2 conditional-control successor (S1 completed, not passed)

`configs/gate_a2_conditional_control_v1.json` is a separate, immutable S1
protocol that tests whether the shared conditional flow backbone can be made
causal before changing any orbit-alignment machinery. It runs only the same
eight IDs with the `direct_irrep` baseline, identical capacity/seed/noise, and
the four pre-registered 800-step variants: legacy input injection, explicit
base-plus-residual conditional field, that field plus tangent-ranking loss, and
the same loss with graphwise condition dropout 0.1. The residual field uses
`g(t)=0.25+0.75*4t(1-t)`, has separate type/coordinate/lattice residual heads,
and applies FiLM plus a conditional residual gate in every message block. A
physical zero tensor remains a present condition and is distinct from the CFG
null token.

All four A2 S1 variants failed at both fixed learning-curve checkpoints; S2 is
locked and was not launched. At 800 steps, the best generated between/within
ratio was 1.00685 (requirement >= 1.2), and the best own-target win rate was
0.63889 (requirement >= 0.75). The counterfactual residual variant achieved a
positive mean own-target margin (0.41504), preserved common-noise terminal
state differences, and had zero sampling failures, but neither condition
response nor pre-registered CFG=1 supplementation produced generated-target
separation. See `reports/gate_a2_conditional_control_v1/gate_a2_s1_report.md`;
this result does not alter Gate A v1 or claim Gate A passage.

### Gate A3 early-branching successor (two-target screen completed, not passed)

`configs/gate_a3_early_branching_v1.json` pre-registers the two distinct,
four-atom, nonzero high-response targets `JVASP-1180` (InN) and `JVASP-22673`
(BN). Their 24-frame relative tensor-orbit distance is 0.98325 and their
scale-invariant lattice-shape distance is 0.26234. It compares FM-only against
one fixed early-time all-negative tangent-identification objective; it does
not tune residual gates, FiLM, CFG, counterfactual weights, or training steps.

The two-target gate failed at 400 steps. The identification variant reached
early/all-time own-target retrieval of 0.70/0.50 (requirements 0.90/0.80), a
generated between/within ratio of 1.01288 (requirement 1.2), and decoded
training-endpoint retrieval of 0.375 (requirement 0.75). It did retain a
positive tangent margin, common-noise continuous early branches, and zero
sampling failures. The matched-noise argmax compositions were not identical,
so it is not labelled “continuous control without discrete branch change”; the
decoded structures nevertheless fail to align reliably with either requested
endpoint. Consequently no 4-target/8-target extension or new conditional
module is permitted. The next scientific audit is the probability path,
atom-type manifold, decoder, and flow-target definition. See
`reports/gate_a3_early_branching_v1/gate_a3_two_target_report.md`.

### Gate A4 generator-substrate audit (completed, not qualified)

`configs/gate_a4_generator_substrate_v1.json` isolates the generator substrate
from tensor conditioning on the same frozen InN/BN pair. First, its exact
velocity closure test passed: the production Euler sampler recovered each
endpoint for type-only, coordinate-only, lattice-only, and joint paths with a
maximum continuous error of `3.11e-7` (tolerance `1e-5`) and decoded endpoint
accuracy `1.0`. Thus the current failure is not an analytic time-direction,
torus-wrap, SPD-log/exp, or exact-velocity integration error.

The two-class endpoint-ID qualification then failed after its fixed 400 steps:
type-only decoded composition accuracy was `0.000` (required `>=0.95`),
geometry-only endpoint retrieval was `0.5625` (required `>=0.90`), and joint
endpoint retrieval was `0.625` (required `>=0.90`); all three
between/within ratios were below `1.2`. There were zero non-finite samples.
The full 119-logit Euclidean type path ended at top-1 accuracies `0.00/0.00`
(InN/BN); the v2 active-element mask reached `0.25/0.25`; the diagnostic-only
`{B,N,In}` vocabulary reached `0.50/0.50`; a projected simplex path reached
`0.50/0.75`; and the fixed categorical diagnostic reached `0.50/0.25`. These
are mechanism diagnostics, not final model claims; in particular, the
`{B,N,In}` vocabulary is forbidden as a final vocabulary.

Therefore the generator substrate is **not qualified** even when the
condition is a trivial endpoint ID. Do not resume tensor-conditioned gates,
4/8-target A3, A2 S2, full training, relaxation, DFT, or DFPT. Any atom-type
manifold/decoder or joint-generation repair needs its own new versioned
protocol. See `reports/gate_a4_generator_substrate_v1/`, especially
`path_closure_report.md`, `endpoint_id_results.csv`,
`type_path_comparison.csv`, and `head_loss_gradient_audit.csv`.

### A5--A10 type-substrate repairs and site audit (all completed; none qualifies Gate A)

The CUDA A5 experiment corrected two mathematical mismatches without relaxing
any A4 rule: a true Dirichlet/simplex type path with endpoint NLL, and a
training-only periodic optimal-transport/no-drift coordinate coupling. Its
invariants passed, but endpoint-ID type composition reached only `0.3125` and
geometry retrieval `0.5625`; the substrate remained unqualified.

A6 replaced the earlier repeated-`argmax` categorical diagnostic with an
absorbing discrete-flow-matching posterior and its exact masked-to-element
jump sampler. Analytic closure was exact and common-noise target branching was
nonzero, yet type composition/atom accuracy were `0.75`/`0.46875`. A7 then
generated a graph-level 119-element count latent and enforced that count at
every discrete reveal. This made composition exact (`1.0`), but did not make
the species assignment to periodic sites reliable. A8 corrected a real
implementation bug: `original_injection` previously did not inject time into
the main node/message-passing path for endpoint-ID, raw-tensor, or direct-irrep
conditioning. The fixed 400-step A8 run improved atom accuracy to `0.59375`,
but still failed. A9's one fixed source-weighted Beta(1/2,1) DFM time measure
also failed (`0.578125` atom accuracy); it is archived, not tuned further.

The read-only A10 species-aware periodic StructureMatcher audit resolves a
potential ambiguity in the CIF row-order metric. Its match rates for A7/A8/A9
are `0.4375`/`0.5625`/`0.1875`, so the remaining failures are genuine chemical
sublattice mismatches, not merely arbitrary atom-index permutations. The
current blocker is therefore the scalar type-site representation: in
endpoint-ID mode it receives no tensor response edge field, and scalar messages
currently lack periodic edge-length/vector-state invariants. The next proposal
must first audit the unlabeled periodic-site symmetry before deciding whether
geometry alone, stochastic symmetry breaking, and/or quotient supervision are
required; it may not add more sampler/loss searches or use target atom order.
See `reports/gate_a5_quotient_substrate_v1/` through
`reports/gate_a10_site_representation_audit_v1/`.

### A11.0 periodic unlabeled-site identifiability audit (completed; no training)

`configs/gate_a11_0_periodic_site_orbits_v1.json` first tests whether the
fixed-CIF type labels in the InN/BN panel are mathematically identifiable from
the proposed A11-G geometry representation. It Niggli reduces a copy of each
endpoint with every species replaced by the same dummy atom, enumerates its
periodic automorphisms, and only then compares the resulting site orbits with
the true elements. It reports both proper SO(3) operations and full O(3)
operations; the full partition is decisive for the pre-registered first
geometry head because distances and dot products cannot distinguish a mirror.

The audit rules out A11-G on this panel. InN has one four-site mixed orbit;
BN has two two-site mixed orbits. Under both partitions, a deterministic
O(3)-scalar site decoder has a fixed-CIF accuracy ceiling of `0.5`. The
observed A7--A9 0.406--0.594 site accuracies are therefore not evidence that
more radial bases or a larger deterministic GNN would solve the panel. See
`reports/gate_a11_0_periodic_site_orbits_v1/`.

### A11-Q exact assignment quotient (Q0 passed; Q1/Q2 not started)

`configs/gate_a11_q_exact_assignment_v1.json` replaces the earlier prepared
A11-S/Q contract for the current four-site InN/BN panel. The production
assignment definition is exact: with predicted composition counts `n`, it
enumerates each unique chemical labeling in `A(n)` exactly once and uses
`p(Y) = softmax_Y sum_i C[i,Y_i]`. Thus a 2+2 composition has six assignments,
not 24 permutations of two indistinguishable slots. Q0 has verified this
finite law, exact-count Gumbel-max sampling, state-dependent residual groups,
and FP32 node-relabeling consistency without training a model. Its outputs are
in `reports/gate_a11_q_exact_assignment_v1/`.

The production quotient is `proper_so3` only. `full_o3_scalar` remains a
diagnostic comparison for the O(3)-scalar decoder considered in A11.0; improper
operations cannot be silently quotiented when a rank-three polar tensor enters
future GaugeFlow conditioning. At time `t`, Q1 must marginalize only
`Gamma_t = {gamma in Aut(X): gamma y_t = y_t}`, respecting both revealed
species and absorbing masks, and deduplicate equivalent labelings before
summing their likelihoods. Sinkhorn and Hungarian implementations remain
unit-tested utilities for a future large-N approximation, but are not used for
the Q0/Q1 scientific attribution. Fixed-CIF site accuracy is only a diagnostic:
high quotient accuracy with approximately 0.5 fixed-CIF accuracy can be the
correct removal of arbitrary CIF row labels.

Q0 and the subsequent supplied-composition decoder qualification permit a
separately versioned Q1 proposal, but Q1 has not started. It must remain
endpoint-ID, type-only, fixed-geometry, full-119-element, and
model-composition-only, with pre-registered composition/exact-assignment/
StructureMatcher/mask/failure/relabeling thresholds. If it passes, a separately
versioned Q2 must first test materials with different proper-SO(3) orbit
structures; tensor conditioning does not resume directly.

### TensorOrbit-JARVIS-v2 oracle preparation

The formula-disjoint v2 split remains inactive for GaugeFlow. Its external
oracle qualification protocol now fixes matched v2 manifests for GMTNet and an
architecture-distinct e3nn SE(3)-Transformer rank-three tensor predictor;
PiezoJet is not a primary oracle. The preparation is at
`configs/tensororbit_jarvis_v2_oracle_qualification_v1.json` and remains
inactive pending external source pins and matched training. Its protocol and
preparation manifest are committed and attested in
`artifacts/tensororbit_jarvis_v2_oracle_qualification_v1/commit_attestation.json`;
neither external training nor a GaugeFlow 4,000/499/499 run has started.

### Substrate-v2 decoder qualification (completed; supplied-composition only)

The negative substrate gates exposed implementation defects, not a reason to
declare conditional crystal generation impossible. A new isolated
`substrate-v2` implementation repairs the prerequisites without changing any
historical checkpoint or Gate conclusion:

- chemical elements use dense tokens `0..117` (physical atomic numbers
  `1..118`); the absorbing mask is the separate token `118`, so no untrained
  chemical class can be decoded;
- periodic edges retain closest-image displacement, integer image shift,
  distance, radial-basis features, and equivariant vector-state invariants;
- `GeometryAwareSiteScorer` predicts site--species scores from fixed periodic
  geometry and endpoint ID, with no CIF row index, target species map, tensor,
  target composition, or target stabilizer;
- the first qualification uses exact count-constrained, proper-SO(3)
  residual-automorphism quotient likelihood only as a decoder isolation. Its
  supplied composition is explicitly non-production and therefore distinct
  from unstarted A11-Q1;
- the former direct-irrep baseline is archived as a two-contraction historical
  control. New comparisons must use the complete six-path e3nn CG baseline in
  `src/gaugeflow/direct_irrep.py`.

The local source-pinned TensorOrbit-JARVIS-v2 raw build now supplies the InN/BN
panel, so this decoder-only qualification has been run on WSL CUDA with three
fixed seeds and 1,200 updates per seed. Its evidence is deliberately split by
version rather than overwritten:

| Version | Status | What it established |
| --- | --- | --- |
| v1 | implementation-invalid | The saved report is retained, but its relabeling check permuted post-hoc scores rather than executing a relabeled model forward. |
| v2 | valid failed numerical-equivalence check | The vector-invariant decoder achieved chemical assignment and structure matching, but unbounded FP32 score accumulation violated the unchanged assignment-law relabel threshold. |
| v3 | candidate decoder qualified | With fixed bounded categorical scores and float64 tiny-neighborhood accumulation, the RBF+vector-invariant candidate passed all six candidate rows: proper-SO(3) quotient MAP and species-aware periodic StructureMatcher rate `1.0`, exact count, zero masks/failures, and maximum fresh-forward assignment-law error `2.22e-16`. |

`reports/substrate_v2_decoration_only_v3/promotion_metric_audit.md` explains
why the v3 runner's unmodified aggregate manifest remains `not_passed`: it
incorrectly required the deliberately deficient legacy/RBF-only ablations to
pass as well. The candidate-only audit does **not** rewrite that manifest or
relax any threshold. It qualifies only a fixed-geometry decoder with the true
composition supplied as exact-assignment support; it does not qualify generated
composition, tensor-conditioned GaugeFlow, a full benchmark, or physical
validation. The next permitted step is a separately versioned Q1
model-generated-composition qualification.

`configs/synthetic_rank3_tensor_control_v1.json` provides a non-cancelling,
exactly SO(3)-equivariant synthetic rank-three teacher for tensor-plumbing
tests.

Two implementation defects were found before restarting the frozen run:

- The integer-action catalogue incorrectly contained infinite-order shear and
  hyperbolic matrices. It now retains only finite crystallographic orders
  1, 2, 3, 4, and 6 (3,480 proposals reduced to 792).
- Exact physical zero tensors exposed a zero-norm backward singularity. Model
  norms now use a finite-gradient `safe_norm`, with regression tests covering
  the zero-response condition.

Performance is no longer the Gate A blocker. The model always ran on CUDA, but
the old implementation starved the GPU with Python graph loops, repeated e3nn
basis construction, thousands of tiny kernels, and synchronizing copies.
After caching and vectorization, ordinary resident-batch measurements are
0.0118/0.0153/0.0160/0.0220 seconds per step for raw/direct/pooling/alignment.
Alignment is 1.44x slower than direct-irrep while retaining all 792 candidates.
Low `nvidia-smi` utilization is expected because each step is much shorter than
the utility's refresh interval and torch peak allocation is only 20--35 MiB.

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
- src/gaugeflow/vocabulary.py: versioned dense 118-element tokens and separate mask invariant.
- src/gaugeflow/geometry.py and substrate_v2.py: metric PBC edges and the geometry-aware discrete-decoration scorer.
- src/gaugeflow/direct_irrep.py: complete six-channel e3nn CG direct baseline for future comparisons.
- src/gaugeflow/provenance.py: explicit engineering-Voigt conversion and proper-SO(3) Reynolds projection.
- src/gaugeflow/manifold.py: standalone product crystal flow coordinates.
- src/gaugeflow/model.py: finite-orbit response encoder, Cartesian direct-irrep control, and graph vector field.
- src/gaugeflow/data.py: direct CSV/CIF reader using PyG Data/Batch, independent of FlowMM.
- src/gaugeflow/unit_cell.py: strict Niggli reduction with tracked basis changes.
- src/gaugeflow/stabilizer.py: crystallographic/tensor-stabilizer utilities plus the active state-derived soft-stabilizer estimator.
- src/gaugeflow/flow.py: conditional flow-matching objective and sampler.
- scripts/: training and tensor-orbit sampling entry points.

## End-to-end generation flow

1. Read a rank-three Cartesian piezoelectric target (or its 18-component irrep
   coordinate vector), preserve an exact physical zero as a present condition,
   and optionally apply graphwise CFG dropout using a separate Boolean mask.
2. Initialize 119-dimensional Euclidean atom-type logits, fractional coordinates on the three-torus, and
   an SPD lattice-log state from noise.
3. At each flow time, build periodic Cartesian bond geometry from the current
   state. `orbit_alignment` evaluates a fixed tensor orbit plus a dynamic
   posterior over all 792 proper lattice-action proposals inferred only from
   the current noisy lattice, coordinates, and type state.
4. Query the tensor response on fixed lossless probes and current bond
   directions, update invariant scalar and covariant vector messages, and
   predict type, coordinate, and lattice tangent velocities.
5. Euler-integrate the conditional vector field, optionally combining the
   conditional and learned-null predictions with classifier-free guidance, and
   decode the final type logits by `argmax`, wrapped fractional coordinates, and
   lattice. Sampling never receives a paired target CIF or target lattice.

## Environment and basic use

The currently verified environment is WSL Ubuntu 22.04 with the
`flowmm-t2c` micromamba environment. Run commands from the repository root:

```bash
micromamba activate flowmm-t2c
export PYTHONPATH="$PWD/src"
python -m pytest -q
```

This is the authoritative experimental environment: it has `torch 2.5.1+cu124`
and can see the 16 GB RTX 4060 Ti. The Windows Anaconda interpreter is
`torch 2.11.0+cpu`; do not use it for reported training or sampling results.

Train the active method with a Cartesian tensor target cache:

## TensorOrbit-JARVIS-v1 data artifact

GaugeFlow retains the historical local evaluation artifact under
`data/tensororbit_jarvis_v1/`: the 4,000/499/499 v1 split and the
Reynolds-projected Cartesian tensor targets. It is not formula-disjoint: its
audit found 165 cross-split formula groups covering 672 rows and 56 structural
near duplicates. It remains reproducibility evidence only; the separate v2
activation candidate is required before future validation/test claims. Neither
artifact is a runtime dependency on PiezoJet or any predictor checkpoint.
GaugeFlow keeps zero-response crystals as physical negatives and never emits
target-CIF stabilizers as model inputs.

```bash
PYTHONPATH=src python scripts/train.py \
  --train-csv /mnt/e/CODE/T2C-Flow/gaugeflow/data/piezo \
  --split-manifest /mnt/e/CODE/T2C-Flow/gaugeflow/data/tensororbit_jarvis_v1/splits.json \
  --split train \
  --target-cache-dir /mnt/e/CODE/T2C-Flow/gaugeflow/data/tensororbit_jarvis_v1/reynolds_projected_targets \
  --preprocessed-cache artifacts/tensororbit_jarvis_v1_preprocessed_v1.pt \
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

The artifact is response-norm stratified (4,000 / 499 / 499), but the audit
found that v1 is **not formula-disjoint**: 165 reduced-formula groups affecting
672 rows cross splits, with 56 cross-split near-duplicate structure pairs.
Consequently, v1 must not support formula-disjoint or clean-generalization
claims. It remains frozen for the current eight-record Gate A protocol. An
inactive formula-disjoint v2 candidate is under
`artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/`, with activation
requirements recorded in
`artifacts/tensororbit_jarvis_v2_activation_audit/activation_protocol.json`.
Activating it requires a new protocol version and new checkpoints. Every future
validation/test or full benchmark must use v2 after that activation; v1 may not
be silently substituted. Training uses square-root inverse-frequency sampling
across five response strata by default, while validation/test retain their
natural distributions.

The new raw reconstruction path is
`configs/tensororbit_jarvis_v2_raw_build_v1.json` plus
`scripts/build_tensororbit_v2_raw.py`. It requires a downloaded raw-record
file whose SHA-256 matches a release manifest, explicit per-record Voigt order,
engineering-shear declaration and units, an exhaustive exclusion map, and the
formula-grouped candidate split. It writes split CSVs, proper-SO(3)
Reynolds-projected Cartesian target cache files and build/exclusion manifests.
The initial v2 raw build is now materialized from a locally pinned GMTNet
release copy (5,000 records, source commit and file hashes recorded): it covers
all 4,998 candidate IDs and explicitly excludes two non-parent records. The
build and formula-disjoint audit pass, with 2,297 exact physical-zero targets.
Its attestation is
`artifacts/tensororbit_jarvis_v2_raw_build_v1/attestation.json`. The original
download timestamp for that local copy is unavailable, so it is not yet a
direct-release provenance qualification and it does not activate the two
external oracles or any generator training.

## Status contract

The new package is the active GaugeFlow path. QR canonicalization, raw
component conditioning and FlowMM are
baselines, not fallbacks. The prepared
JARVIS/GMTNet CSV may be read as an input dataset, but no other project's
Python module or model checkpoint is imported.

Use ``--conditioning-mode orbit_alignment`` for the active finite-orbit model
and ``--conditioning-mode direct_irrep`` for the Cartesian direct-interaction
baseline in the frozen historical runs. It has only two contractions and is not
a complete CG baseline. Future fair controls must instantiate
`CompleteDirectIrrepCoupling`, which preserves all six `1o` pathways from
`(2x1o + 1x2o + 1x3o) x (0e + 2e)`. Classifier-free guidance is trained with
``--condition-dropout`` (default ``0.1``); a zero physical tensor remains
distinct from the learned null condition.

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
  --preprocessed-cache artifacts/tensororbit_jarvis_v1_preprocessed_v1.pt \
  --device cuda
```

## What is currently blocked

- **Current Gate A failure:** the learned velocity is condition-sensitive, but
  400-step samples from different targets are not sufficiently separable.
- **Generator substrate failure:** Gate A4 passed analytic path closure but
  failed the trivial endpoint-ID type and geometry qualification. A5--A9
  isolated and repaired continuous type paths, composition conservation, and
  time injection; A10 shows the remaining blocker is geometry-blind scalar
  type-site decoding and symmetry breaking. Do not return to tensor
  conditioning, add more sampler/loss searches, or use target atom order as a
  shortcut.
- **Missing physical evidence:** no qualified frozen tensor-oracle ensemble or
  training-panel orbit-tensor-error distribution is available yet.
- **Future benchmark data:** v1 split leakage prevents a credible full
  4,000/499/499 generalization result. v2 has passed an activation audit but
  remains inactive until a separately versioned protocol creates new
  checkpoints; it is mandatory for all future validation/test claims.

These are distinct issues. CIF parsing, DataLoader throughput, and CUDA device
placement are no longer the active blockers. Do not start the full run,
relaxation, DFT, or DFPT from the current supporting result.

Gate B tests random tensor representatives at fixed tensor orbits using
velocity equivariance and distributional comparisons (C2ST/MMD), Gate C is the
three-seed method screen, and Gate D is the only stage allowed to consume
relaxation/DFPT budget. A finite objective or a smoke sample is never a pass.
