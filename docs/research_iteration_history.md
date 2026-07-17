# GaugeFlow research iteration history

This document replaces the executable archive of exploratory Gate A--A11,
P5-D0/C0, substrate-v2, and vNext Q0/Q1 experiments. Those experiments were
valuable for diagnosis but are not part of the revised production model or its
final paper evidence. Their complete code, configurations, reports, tests, and
data pointers remain recoverable from the annotated Git tag
`archive/pre-production-cleanup-20260716` at commit
`0dbcfbabd997b3e32a18ed391e28adb1fe4f3ffc`.

The active repository intentionally keeps only the revised hybrid-production
surface and the current TensorOrbit-JARVIS-v2 data protocol. The later
S0/S1a-I0 executable evidence, harmonic reference and per-run reports are
recoverable from `archive/pre-runtime-cleanup-20260717`. Historical files must
not be copied back as runtime fallbacks.

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

Harmonic/Hopf code is retained only in the archive tag as a paper diagnostic;
it is not part of the installed package. The 24-frame-only Cartesian atlas
(S0.3-v1) failed and remains frozen
as a paper result. The weighted 24x7x24 Cartesian prior (S0.4-v1) passed its
scientific/numerical checks but failed the frozen 20 ms latency limit at
41.89 ms. S0.4.1 preserved the same 4,032-candidate prior and qualified the
runtime at 14.62 ms in the official report (13.09 ms in a later no-write smoke).

Only the current production Cartesian atlas remains active. The S0.3, S0.4 and
S0.4.1 runners/configs/reports are represented by this history, the manuscript
tables and the archive tag.

## Retired performance artifacts

Early profiler and cProfile traces occupied roughly 905 MB. They established
that per-forward candidate deduplication and Python-side frame work dominated
the atlas path. The actionable result was implemented as cached cubature and a
proof-gated unique generic fast path. Raw traces, `.prof` files, pre/post top-20
tables and standalone S0.4.1 report files are retired; the qualified metrics
remain in the manuscript and this history.

## H0-E parent-occurrence follow-up

H0-E-v1 remains frozen failed at `125/1024 = 0.12207` against its preregistered
`0.15` candidate-coverage threshold. A post-freeze semantic audit found that
the implementation had named the final reconstruction error as a residual
bound. These are different quantities:

\[
d_{\rm rec}=d_{\rm PBC}(x_{\rm child},\hat x_{\rm child}),\qquad
r_{\rm RMS}=\sqrt{N^{-1}\sum_i\|r_i\|^2}.
\]

The active compiler now records and checks both. Rebuilding the 125 discovered
candidates in memory gave median/p95/max true residual RMS
`0.001546/0.04078/0.08516` Angstrom, so every v1 candidate still qualifies;
the historical failure remains coverage-only.

Two no-write successor diagnostics tested whether the missing coverage was a
cheap parent-construction artifact. The first projected only the conventional
Gram matrix onto finite volume-preserving monoclinic, orthorhombic,
tetragonal, cubic, hexagonal and rhombohedral strata. It recovered the
synthetic pure-strain example but added `0/64` candidates on a fixed,
crystal-system-balanced panel of v1 no-candidate rows. The second jointly
enumerated lattice rotations and species-preserving site permutations, closed
each finite affine extension, and used a batched universal-cover Reynolds
projection. It also added `0/64`; the bounded extra actions found on the panel
did not survive tight parent certification. Extending the spglib proposal
ladder to `0.9` Angstrom found three candidates, but only one satisfied the
unchanged OPD/residual contract.

Consequently metric-only projection, blind tolerance extension and the
experimental affine-extension search were removed from the active code. They
must not become runtime fallbacks. A correct complete successor needs an
explicit maximal group--subgroup/Wyckoff-splitting catalogue followed by a
joint site-and-strain projection. This is the same crystallographic object
used by PSEUDO, AMPLIMODES and ISODISPLACE, not a lattice-only approximation:

- Kroumova et al., *PSEUDO: a program for a pseudosymmetry search*, J. Appl.
  Cryst. 34 (2001), [doi:10.1107/S0021889801011852](https://doi.org/10.1107/S0021889801011852).
- Orobengoa et al., *AMPLIMODES: symmetry-mode analysis on the Bilbao
  Crystallographic Server*, J. Appl. Cryst. 42 (2009),
  [doi:10.1107/S0021889809028064](https://doi.org/10.1107/S0021889809028064).
- Campbell et al., *ISODISPLACE: a web-based tool for exploring structural
  distortions*, J. Appl. Cryst. 39 (2006),
  [doi:10.1107/S0021889806014075](https://doi.org/10.1107/S0021889806014075).

The representation policy is unchanged: site actions are stored as a node
permutation plus a Cartesian `3x3` rotation; homogeneous strain is the exact
six-coordinate Kelvin image of symmetric Hencky strain; periodic closest
vectors share one QR factorization; OPD projectors and component orbits are
cached. These are mathematical equivalences and measured accelerations. No
dense `3N x 3N` action, exhaustive per-material PyXtal search, or approximate
metric-only parent is retained.

The post-audit implementation also applies the Reynolds operator by linear
reductions of the compact component orbits instead of allocating a complete
residual orbit for every one/two-mode subset. A fixed synthetic WSL float64
microbenchmark (`|G|=384`, `N=128`, two components, 128 reductions) measured
`1.51x` lower wall time and `1.88x` lower traced peak allocation, with maximum
absolute difference `3.1e-16`. Element masses are cached and the parent-cell
inverse is reused across terminal certifications. These changes do not alter
the candidate set or acceptance equations.

The frozen H0-E-v1 runners were removed from the active tree after commit
`f6f0262bfe9bbd983213467b20e66bce5fcb8485`; that commit remains the exact v1
reproduction surface. No incomplete v2 runner is retained before its
group--subgroup/Wyckoff compiler and protocol are frozen.

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
