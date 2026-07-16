# GaugeFlow research iteration history

This document replaces the executable archive of exploratory Gate A--A11,
P5-D0/C0, substrate-v2, and vNext Q0/Q1 experiments. Those experiments were
valuable for diagnosis but are not part of the revised production model or its
final paper evidence. Their complete code, configurations, reports, tests, and
data pointers remain recoverable from the annotated Git tag
`archive/pre-production-cleanup-20260716` at commit
`0dbcfbabd997b3e32a18ed391e28adb1fe4f3ffc`.

The active repository intentionally keeps only the revised hybrid-production
surface, its S0 qualifications, and the current TensorOrbit-JARVIS-v2 data
protocol. Historical files must not be copied back as runtime fallbacks.

## Data generations

| Generation | Role | Outcome |
|---|---|---|
| TensorOrbit-JARVIS-v1 | Early 4,000/499/499 flow and Gate A substrate | Retired. It supported diagnostic experiments but is not valid for future validation/test claims. |
| Formula-grouped candidate split v2 | Formula-group separation used while activating v2 | Retained because the current v2 build and qualification manifests reference it. |
| TensorOrbit-JARVIS-v2 | Current external tensor-oracle qualification dataset | Retained. Future validation/test work must use a source-verified v2 protocol. |
| TensorOrbit-JARVIS-v2 full-O(3) source-verified build | Current parity-aware data audit | Retained. Full-O(3) crystal compatibility is distinguished from the SO(3) orbit of the polar rank-three tensor. |

The retired v1 tensor cache and Gate-specific selection manifests are available
from the archive tag. They are intentionally absent from the current working
tree so a future trainer cannot silently select them.

## Conditional-flow iterations

### Gate A and A.1

The four early conditioners (`raw_tensor`, `direct_irrep`,
`stabilizer_pooling`, and `orbit_alignment`) showed that the condition could
change teacher-forced velocity. Gauge/orbit alignment improved representative
consistency, but free-running samples did not form distinct target-conditioned
distributions. Common-noise and own-target ranking diagnostics showed that a
locally condition-sensitive vector field was not sufficient to make the tensor
a necessary trajectory branch variable.

### Gate A2

Per-message-block FiLM, an explicit conditional residual field, graphwise
condition dropout, and counterfactual ranking were tested under fixed budgets.
The condition changed all three velocity heads, but generated between/within
separation did not pass. Increasing conditional-injection machinery was
therefore stopped rather than promoted into production.

### Gate A3

An InN/BN two-target early-branching test produced condition-dependent
continuous trajectories, but the final atom-type argmax often selected elements
unrelated to either endpoint. This established the key distinction between
continuous control and a discrete chemical branch change.

## Generator-substrate iterations

### Gate A4

The analytic probability-path closure passed for type, coordinates, lattice,
and their joint state, excluding a simple time-direction/Euler/SPD-log error.
The learned endpoint-ID substrate did not qualify: type-only decoded
composition was 0.000, geometry retrieval 0.5625, and joint retrieval 0.625
against thresholds 0.95/0.90/0.90. The failure was in the learned generator and
decoder rather than the analytic endpoint path.

### Gates A5--A10

These experiments separated quotient geometry, discrete type transitions,
composition counts, time conditioning, and species-aware evaluation:

- A5 repaired simplex/quotient diagnostics but did not qualify the generator.
- A6 replaced repeated argmax with an absorbing categorical path; composition
  reached 0.75 and still failed.
- A7 generated exact composition counts and reached composition accuracy 1.0,
  while chemical site assignment remained wrong.
- A8 repaired time-blind endpoint-ID injection; atom accuracy improved to
  0.59375 but remained below qualification.
- A9 used one frozen source-weighted objective and did not qualify.
- A10 showed with species-aware periodic matching that the residual error was
  a real chemical-sublattice mismatch, not merely CIF row order.

### Gate A11 and substrate-v2

The InN/BN fixed geometry has species-mixed full-O(3) site orbits, so a
deterministic scalar decoder cannot recover arbitrary CIF row labels. This made
fixed-CIF site accuracy an invalid primary gate. Exact count-constrained
assignment enumeration with the state-dependent residual automorphism group
passed its read-only mathematical checks. A later fixed-geometry,
supplied-composition substrate-v2 decoder achieved exact proper-SO(3) quotient
assignment and species-aware matching, but it was not a composition generator,
tensor conditioner, or end-to-end GaugeFlow result. These implementations are
retired rather than mistaken for the current production generator.

## P5 coordinate and synthetic-control iterations

The P5 sequence tested whether the old continuous coordinate flow could learn a
single endpoint before tensor control was reconsidered.

- D0/D0.1/D0.2 exposed failures to fit fixed batches and initially even a
  single sample, motivating a direct audit of the coordinate head, translation
  gauge, metric features, and gradient chain.
- Translation quotienting and periodic distance/RBF features repaired the
  elementary single-sample issue but did not make the old vector field stable
  over a complete time trajectory.
- D0.4--D0.8 tested full-time conditioning, endpoint bridges, finite flow maps,
  rollout correction, semigroup consistency, and contraction. Long-span and
  composed rollouts remained below the frozen qualification requirements.
- C0 identified that dynamic nearest-image, permutation, and translation-gauge
  choices can switch branches. A fixed source--endpoint lift made the target
  mathematically coherent but did not qualify the old generator.
- The exact synthetic rank-three teacher was useful as an oracle-free test, but
  the old flow did not demonstrate successful tensor-conditioned generation.

The lesson retained in production is structural: define the quotient/path once,
keep discrete and continuous manifolds explicit, and qualify a tensor-free
reverse generator before adding a conditioner. The old `flow.py`, `model.py`,
and their experiment-specific helpers are not retained as compatibility code.

## vNext Q0/Q1 iteration

The vNext analytic-flow branch was an intermediate redesign. Q0 was blocked
because historical P5 weights had not been saved; Q0.1 could only establish a
partial legacy diagnostic and never reconstructed the missing checkpoint. The
branch was superseded by the revised hybrid-diffusion manuscript architecture.
Its source tree, gate configurations, frozen-manifest machinery, and tests are
retired. The archive tag is the sole reproduction path.

## Harmonic and Cartesian conditioner iterations

Harmonic/Hopf code is kept only under `gaugeflow.production.archive_harmonic`
where needed for the paper's diagnostic reference; it is never a runtime
fallback. The 24-frame-only Cartesian atlas (S0.3-v1) failed and remains frozen
as a paper result. The weighted 24x7x24 Cartesian prior (S0.4-v1) passed its
scientific/numerical checks but failed the frozen 20 ms latency limit at
41.89 ms. S0.4.1 preserved the same 4,032-candidate prior and qualified the
runtime at 14.62 ms in the official report (13.09 ms in a later no-write smoke).

S0.3, S0.4, S0.4.1, and the current production Cartesian atlas remain in the
active repository because the manuscript directly relies on them.

## Retired performance artifacts

Early profiler and cProfile traces occupied roughly 905 MB. They established
that per-forward candidate deduplication and Python-side frame work dominated
the atlas path. The actionable result was implemented as cached cubature and a
proof-gated unique generic fast path. Raw traces, `.prof` files, and pre/post
top-20 tables are retired; the qualified S0.4.1 metrics are retained.

## Current scientific boundary

The current tree proves mathematical interfaces and a qualified Cartesian-atlas
runtime. S1a-I0 v1/v1.1 showed that a raw cosine-VP lattice score can produce a
numerically unstable high-noise clean-state inversion; v1.2 ruled out a simple
training-budget explanation. The v1.3 clean log-volume/log-shape
parameterization passed the bounded CUDA trainer/reverse-sampler closure with a
fixed-loss ratio of 0.0811, zero sampling failures, and zero terminal masks.

This is software-path evidence only. Real-data S1a and decoded generation
quality have not been run, the production blueprint remains P1 rather than a
full space-group/Wyckoff sampler, and the project does not claim
tensor-conditioned sample separation. No tensor fine-tuning, learned oracle,
relaxation, DFT, or DFPT is authorized by the implementation closure.
