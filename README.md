# GaugeFlow

GaugeFlow is a standalone implementation of tensor-orbit-conditioned crystal
generation. It does not import FlowMM at
runtime. The former local FlowMM working tree has been removed; the upstream
baseline and the preserved local T2C patch set are documented under
`../legacy_backups/flowmm_local_2026-07-14/`.

## Current experimental status (2026-07-14)

GaugeFlow is in **Gate A**, the small-real-subset conditioning check. Gate A
has not passed and no full 4,000/499/499 training result is claimed. The
original four-method 400-step result is frozen as a negative v1 archive at
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
1.0066 against the required 1.2. Gate A therefore remains unresolved. The full
decision additionally requires a pre-qualified frozen external tensor-oracle
ensemble, the training-set orbit-tensor-error distribution, and the registered
physical micro-audit. See `reports/performance_data_scientific_audit.md`.

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

Train the active method with a Cartesian tensor target cache:

## TensorOrbit-JARVIS-v1 data artifact

GaugeFlow owns a frozen local evaluation artifact under
`data/tensororbit_jarvis_v1/`: the 4,000/499/499 v1 split and the
Reynolds-projected Cartesian tensor targets. It is a data artifact, not a
runtime dependency on PiezoJet or any predictor checkpoint. GaugeFlow keeps
zero-response crystals as physical negatives and never emits target-CIF
stabilizers as model inputs.

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
  --preprocessed-cache artifacts/tensororbit_jarvis_v1_preprocessed_v1.pt \
  --device cuda
```

## What is currently blocked

- **Current Gate A failure:** the learned velocity is condition-sensitive, but
  400-step samples from different targets are not sufficiently separable.
- **Generator substrate failure:** Gate A4 passed analytic path closure but
  failed the trivial endpoint-ID type and geometry qualification. Resolve the
  atom-type manifold/decoder and joint substrate under a new protocol before
  returning to tensor conditioning; do not add conditional losses as a
  workaround.
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
