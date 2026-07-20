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

### H0-E-v2 E0 maximal embedding catalogue

The next version began with a separately frozen E0 prerequisite rather than
silently rerunning the v1 occurrence pilot. PyXtal 0.6.1 contributes only its
MIT-licensed, source-hashed maximal t/k subgroup and Wyckoff-splitting tables.
The GaugeFlow compiler rationalizes each affine basis/origin transform, checks
all Seitz operations in one broadcasted array, canonicalizes physically
unordered child-orbit labels and removes source multiplicity from candidate
measure.

All 1,103 maximal t records and 2,641 k records with index at most four passed.
The 3,744 raw rows reduce to 2,843 affine embeddings and 2,845 relation
variants; 901 duplicate rows remain provenance only. An independent exhaustive
audit reidentified every one of the 230 source settings with spglib and
recomputed maximum rotation/periodic-translation errors of `2.22e-16` and
`4.44e-16`. E0 is qualified and permits only the bounded parent-occurrence E1
pilot. H0-E and H0 remain unqualified.

A post-E0 no-write equivalence diagnostic also conjugated all source edges to
primitive settings, `T_prim = P_G^-1 T_conv P_H`. Every edge still passed;
every t determinant became exactly one and every k determinant became its
subgroup index, with common rational denominator at most 12. This is the
preferred compact coordinate system for E1 site/strain projection, but it does
not rewrite the frozen E0 artifact.

The pre-gate E1 projector now uses that primitive representation directly. It
Reynolds-projects the row-lattice metric, fixes a proper Cartesian gauge,
solves exact periodic species-preserving assignments and verifies the complete
permutation group law before averaging sites. A positive-control tetragonal
BaTiO3 distortion is recovered as cubic `Pm-3m` (SG 221), while incompatible
species are rejected. Synthetic nonsymmorphic general-position perturbations
for SG 14, 19, 62 and 194 also return to their exact parent groups with group
errors of order `1e-15`; SG 62 remains an active regression test. Four
deterministic low-space-group rows from the frozen
v1 no-candidate panel produced no certified maximal-t parent in a no-write
smoke; this is insufficient to freeze or judge E1 and no tolerance was changed.

The same implementation replaces a cubic broadcasted Seitz-table search by
batched products plus a modulo-lattice integer key, followed by a full
float64 closure verification. Against the previous brute-force reference, all
230 standard primitive space groups produced identical tables; cumulative CPU
time was `0.3103 s` versus `0.03853 s` (`8.05x`). Species blocks now batch all
group-element periodic costs through one shared exact-CVP factorization before
the unchanged Hungarian solves; on the 48-operation BaTiO3 control this was
`25.81 ms` versus `19.33 ms` (`1.34x`) with identical permutations and error.
These are exact representation/vectorization changes, not candidate pruning or
an approximate symmetry test.

### H0-E-v2 E1a maximal-t parent occurrence

E1a froze 64 deterministic, split/system/site-balanced rows from the 899 H0-E
v1 materials with no candidate. It used every reverse-indexed E0 maximal-t
embedding in the exact PyXtal/ITA Hall setting, transformed once to primitive
coordinates. During qualification two implementation semantics were corrected
before the run: a t edge is identified by `cell_index=1`, not by a unit point-
group `subgroup_index`; and a raw orbit defect has the triangle bound twice the
one-sided distance to the parent fixed set, so only the final projected source
displacement is compared with the frozen 0.2 Angstrom threshold.

The committed run found zero new candidate materials against the minimum of
three and is frozen failed. All 64 joins and setting conversions succeeded,
forward/reverse enumeration agreed, and the independent reverse-order rebuild
reproduced the result. Seventeen rows have no incoming maximal-t edge. Among
430 edges on the remaining rows, 236 have orbit defect at least 0.6476
Angstrom, 191 fail the full species-permutation group law, and three projected
embeddings are duplicate setting variants for one SG 12 material with Hencky
norm 0.4898, above the frozen 0.15 bound. Thus a matcher-only patch cannot
rescue the gate. E1b and H1a were not started; any multi-step or cell-changing
successor requires a new H0-E-v3 proposal rather than changing E1a history.

### H0-E-v3 K0 cell-changing maximal-k occurrence

E1a also established that a deeper translationengleiche chain is not a useful
successor: for `H subset M subset G`, `Fix(G) subset Fix(M)`, so distance to a
deeper parent fixed space cannot be smaller than distance to the first maximal
parent fixed space. K0 was therefore frozen as a new H0-E-v3 mechanism test,
not as continuation of the prohibited E1b branch. It reconstructs the full
parent action in a child supercell from the exact quotient
`Z^3 / B Z^3`, performs one species-preserving group-law assignment, Reynolds
projects once, and quotients back to the primitive parent.

The implementation passed synthetic P1, inversion and off-diagonal-basis
positive controls. The fixed 64-row real panel nevertheless yielded zero new
candidates over all 578 maximal-k edges. A read-only rejection decomposition
found 265 site-count/index incompatibilities, 108 assignment group-law
failures, and 205 orbit-defect failures with minimum `1.22584 Angstrom` against
the frozen `0.4 Angstrom` prefilter. Independent reverse-order reconstruction
reproduced the full negative result. K0 and H0-E-v3 are frozen failed; no H1
stage was started.

### H0-E-v4 O0-v2 occupational-order mechanism closure

The t/k failures exposed a representation error rather than a need for wider
geometric tolerances: parent projection required the full parent action to
preserve terminal species before chemical ordering was allowed to lower the
symmetry. The v4 implementation separates the species-free parent geometry
from an ordered full-vocabulary coloring and computes its exact stabilizer in
the parent-supercell node action. Production reconstruction uses the same
stabilizer intersection and no longer has a parent-species copying path.

After versioned removal of one parent-occurrence-incompatible material, the
frozen 63-row O0-v2 panel evaluated 962 edges and found 10 materials with 13
qualified occurrences. All geometric, coloring, subgroup, integer-element and
forward/reverse checks passed; an independent reverse-order audit reproduced
every result. O0-v2 therefore qualifies the mechanism only. H0-E and H0 remain
unqualified, and the only permitted successor is a separately frozen held-out
O1 coverage protocol.

### H0-E-v4 O1-v1 held-out occupational census

O1 froze the complete 835-row clean remainder of the v1 zero-candidate
universe after removing every O0 source ID; it did not draw a second selected
panel. The 125 v1-positive, 63 clean O0 and 835 O1 rows form a disjoint exact
partition of the 1,023-material clean universe. The unchanged aggregate
coverage rule required at least 19 new O1 materials.

The formal four-process CPU run evaluated all 13,370 canonical E0 material
edges and found 224 candidate materials with 454 unique
`(material_id, embedding_key)` paths. There were zero processing failures,
nonfinite values, partial occupancies or duplicate canonical paths. Every
coloring reconstructed exactly, every coloring stabilizer was a valid subgroup
with order equal to the observed child, and all candidates remained inside the
0.2-Angstrom displacement and 0.15 Hencky limits. Aggregate coverage is
`359/1023 = 0.350929`.

An independent reverse-material/reverse-catalogue auditor rebuilt all 835 rows,
13,370 edges and 454 occurrences, including every stored array and scalar.
O1-v1 therefore qualifies H0-E-v4 and H0-v4. It authorizes only a separately
frozen real-data H1a; H1b and H2--H6 have not started.

### Real-data H1a and coordinate-field attribution

The P1 cache rebuilt and independently audited all 675,204 Alex-MP-20
structures. Joint tensor-free training then used the complete 540,164-row
train split for 20,000 updates. Element marginals, lattice volume, finite
positive cells, formula uniqueness, mask termination and sampling reliability
passed, while generated local geometry did not: nearest-neighbour median was
2.172 Angstrom versus 2.698 Angstrom in the train reference and normalized
nearest-neighbour Wasserstein was 0.953 against 0.75. H1a is frozen failed.

One exact train pass of coordinate-only Rao--Blackwell quotient DSM reduced
validation monotonically to 0.54928 but missed 0.35. Raw/EMA and
train/validation comparisons rejected EMA lag and ordinary generalization gap
as the main cause. A formal repeated-species representative difference was
also rejected causally after likelihood weighting placed at most 5.42e-14 mass
on the alternative matching; no permutation-path surrogate was added.

A signed pairwise reciprocal residual was then separately qualified before
training. Its numerical symmetry errors were of order 1e-16 and its 64-graph
BF16 operator benchmark achieved 490.77 graphs/s at 1.73 GiB. A second exact
one-pass run improved validation only to 0.53354 and low-noise endpoint RMS to
0.04494 Angstrom, still above both thresholds. Branch subtraction showed that
the residual was active rather than disconnected, so its insufficient benefit
is a negative representation result. The module and tests were removed from
the current runtime; commit 154e6c9 retains the exact implementation.

The subsequent fixed-state studies did not replace the full-data run. A
corrected translation-quotient Jacobian had full physical rank `30/30`, but a
damped Gauss--Newton step predicted to remove `99.9337%` of the one-state loss
was `3.1575` active-parameter norms long. Within the measured nonlinear trust
region the best preregistered step removed only `0.1388%`. Thus the target is
linearly expressible but its weak directions lie beyond the useful local
curvature radius.

An explicit Helmert quotient then removed the three translation zero modes
before solving the 225-parameter affine coordinate readout. It again spanned
`30/30` physical directions, projected the target with relative residual
`1.12e-15`, and achieved `5.39e-8` through the unchanged FP32 production
forward. The negative result was scale: condition number `3.496e7`, entropy
effective rank `2.23`, and a minimum-norm update of `2079.20` from an initial
readout norm of `0.80036`. With the backbone frozen, exact readout MSE on
1/4/16/64 states was respectively `1.55e-27`, `1.43e-14`, `0.09947`, and
`0.55232`; beyond a tiny panel, state-dependent backbone features are required.

Four bounded successors were rejected. Graphwise vector/edge unit scaling
reduced the required update to `6.14` but missed spectral, throughput, and
translation guardrails. Unregularized 16-state variable projection inflated
the head norm from `9109` to `4.83e7` and destabilized BF16 backbone updates.
A screened quotient-Laplacian operator was efficient but left the spectrum and
required step essentially unchanged. Finally, a function-preserving power-of-two
`1024x` reparameterization reduced the exact-solve norm to `2.03`, yet the
same-state 1,024-step AdamW run ended at MSE `0.40491`, worse than the historical
`0.34414`, because Adam normalization and global clipping cancel a pure constant
scale without decorrelating the basis. All four candidates were removed from
active runtime/config/test dispatch and remain only in reports and Git history.

The preregistered combination was then stopped before training. A fixed
power-of-two `1024x` chart preserved the function to `5.96e-7`, retained design
rank `225/225`, reduced the stored exact-solution norm to `8.894`, and achieved
FP32 MSE `0.099467` with backbone-gradient norm `3.889`. It did not change the
effective unscaled norm (`9107.83`) or cancellation geometry. In BF16 the MSE
rose to `10.9886` (`110.47x` FP32), the gradient norm to `23468.3`
(`6033.9x`), and gradient cosine became `-0.1572`. The vector and edge
contributions had norms `272.59` and `271.00` but summed to only `16.83`, a
`32.31x` cancellation ratio. No optimizer step ran and all parameters were
restored exactly. Scaled variable projection was therefore rejected; commit
`231126d` retains the frozen runner, protocol and tests.

A branch-minimality audit next tested deletion rather than adding machinery.
After an explicit Helmert quotient removed exactly three translation modes,
vector-only and edge-only designs were each locally full rank `30/30` with
one-state target projection residual below `1.8e-13`. Vector-only was relatively
BF16-stable but its 16-state FP32 MSE was `0.56437`, low-time endpoint RMS
`0.05046` Angstrom, and solution norm `1022.67`. Edge-only reached FP32 MSE
`0.13474` and endpoint RMS `0.02401` Angstrom but missed the frozen `0.12` bound,
required norm `1325.83`, and remained BF16-unstable: MSE `10.2160`, gradient
norm `16794.1` versus `4.295` in FP32, and cosine `-0.1419`. Neither branch
qualified. No optimizer step ran and production retained the combined head.

The initial debug execution had estimated quotient rank from an FP32 mean-zero
matrix and counted three roundoff translation modes. Commit `7c9cacb` replaced
that diagnostic with the exact Helmert basis without changing any threshold,
state, seed or model setting; only the corrected result is retained as evidence.

A fixed target-free block Gram--Schmidt chart then tested whether the full
combined span could be retained while removing its parameter anisotropy. The
graph-equal weighted Gram condition number became `1.000000004`, maximum Gram
error was `4.96e-10`, span-prediction error was `1.35e-10`, and the orthogonal
solution norm was only `3.2299`. FP32 MSE and low-time endpoint RMS remained
`0.099464` and `0.020287` Angstrom, while the chart operator cost `0.0255 ms`
and `0.360 MiB` on the fixed CUDA panel.

This exact algebraic success did not stabilize the feature path. The equivalent
raw norm remained `9108.38`; BF16 MSE was `9.7679`, gradient norm `14670.5`,
and FP32/BF16 gradient cosine `0.1278`. The audit therefore rejected post-hoc
orthogonalization before training. It performed zero optimizer steps and left
production unchanged. The causal boundary moves upstream: a successor must
generate compact, scale-controlled Cartesian carriers before quantization and
readout rather than reparameterize an already ill-scaled feature family.

The next operator therefore formed Cartesian moments before the readout. Sixteen
bounded scalar channels generate a polar first moment `m`, an even symmetric-
traceless second moment `Q`, and the Cayley--Hamilton-closed polar family
`(m,Qm,Q^2m)`. Together with the existing 32 vector channels this yields 80
RMS-balanced carriers. The audit used no coordinate target and no optimizer
step. Every one of 16 real states reached full translation-quotient rank; the
worst condition number was `14657.96`, O(3) covariance error `6.76e-6`, and
translation mean error `1.54e-7`. BF16/FP32 carrier and probe-gradient cosines
were `0.99598` and `0.99269`, with gradient-norm ratio `1.00121`. The 12,192-edge
CUDA operator cost `3.043 ms` and `11.609 MiB`. This qualifies only a clean
production integration replacing the old readouts, not target fitting or H1a.

That first clean integration was then tested and rejected before training. It
had the exact `4,479,161` parameters, no legacy readout keys, FP32/BF16 output
cosine `0.99623`, gradient cosine `0.98573`, and strong CUDA performance
(`1066.85 graphs/s`, `506.24 MiB`). However, the target-free production
output-energy gradient norms were `373.27/407.55`, exceeding the frozen `100`
bound in both precisions. The defect is therefore absolute Jacobian scale after
coupling the normalized carrier to the fractional output, not BF16 direction
loss. Zero optimizer steps ran. The integration was removed from active
production and remains at commit `e25f432`; carrier-order and parameter-group
gradient attribution is required before a successor.

## Current scientific boundary

The current tree proves mathematical interfaces and a qualified Cartesian-atlas
runtime. S1a-I0 v1/v1.1 showed that a raw cosine-VP lattice score can produce a
numerically unstable high-noise clean-state inversion; v1.2 ruled out a simple
training-budget explanation. The v1.3 clean log-volume/log-shape
parameterization passed the bounded CUDA trainer/reverse-sampler closure with a
fixed-loss ratio of 0.0811, zero sampling failures, and zero terminal masks.

Real-data H1a has now run and failed for local-coordinate fidelity despite
passing coarse chemistry/lattice and sampler-safety checks. The production
blueprint remains P1 rather than a full space-group/Wyckoff sampler, and the
project does not claim tensor-conditioned sample separation. No tensor
fine-tuning, learned oracle, relaxation, DFT, or DFPT is authorized.

### Cartesian tangent correction and full-split pretraining

The compact Cartesian carrier was ultimately retained after the apparent
production-gradient failure was traced to a tensor index-type error.  The
reverse sampler consumes a tangent drift, not a covector: for row coordinates
`r=fL`, the physical chart is `v_r=v_f L` and its inverse is a batched solve
for `v_f=v_r L^-1`.  The old `L^T` pullback was removed from active runtime.
Successive no-training audits then repaired periodic-lift arithmetic, replaced
atomic reductions by target-contiguous linear-time `segment_reduce`, and fixed
the precision boundary of geometry-sensitive blocks.  The final qualification
had exact repeat determinism, BF16/FP32 output cosine `0.999806`, loss-gradient
cosine `0.997593`, `516.03 graphs/s`, and `185.73 MiB` on the RTX 4060 Ti.

Commit `6591015` preregistered one seed-5705, 8,441-step, exact-one-pass
coordinate-only experiment before its execution.  It then consumed all
540,164 training structures with finite logs and checkpoints.  Fixed
validation decreased monotonically from `34.43436` to `24.24037`, but its
ratio `0.70396` missed the frozen `0.5` bound.  The `t=.005` endpoint RMS
improved from the archived covector result `0.05672 A` to `0.04207 A`, but
still missed `0.04 A`.  The `t=.1` teacher-forced RMS (`0.06143 A`), rollouts
from `t=.1/.2` (`0.06589/0.09861 A`), zero sampling failures, and zero tensor
candidates all passed.  The protocol is therefore failed without threshold,
seed, or step changes.  Joint initialization and every later Gate remain
closed.

### Tangent readout-span attribution

A separately committed zero-step audit then asks whether the corrected model
merely failed to optimize its final 80-to-1 global carrier readout. It captures
the exact centered Cartesian carrier at steps 0/1250/5000/8441 on fixed,
disjoint 128-graph train and validation panels, five times, and two noise
replicates. Graph-equal float64 minimum-norm fits are rank 80/80 throughout;
the carrier/head reconstruction error is `9.54e-7`, parameters are bitwise
unchanged, and no tensor candidate or optimizer step occurs.

At step 8441 the current head explains `57.28%/45.47%` on train/validation. A
train-optimal global head reduces train loss only `5.23%` and increases
validation loss `3.46%`. A validation-label oracle head, used only as an
offline span ceiling, explains `49.61%`, below the frozen `75%` threshold and
consistently about `47--53%` at every audited time/replicate. The oracle span
gains `44.94` percentage points from initialization, so the backbone learns,
but its cross-state Cartesian carrier family remains insufficient. The
classification is `backbone_span_limited`, not a disconnected or unoptimized
global head. Any successor must alter one feature-formation mechanism rather
than add steps, seeds, harmonic branches, or joint objectives.

### Factorized angular moments and the volume-normalized tangent chart

The bounded successor maintains a 64-dimensional scalar state per periodic
edge and forms eight first- and second-order Cartesian moment channels at each
message block. Expanding the contractions gives the explicit degree-one/two
triplet kernels, but the implementation concatenates the 3 vector and 6 STF
components into one target-sorted segment reduction. It therefore retains
`O(E*C)` time and edge/node-linear storage and never creates a triplet index.
The active implementation qualifies at `489.10 graphs/s` and `182.86 MiB` on
the RTX 4060 Ti, with BF16/FP32 output and gradient cosines
`0.999916/0.999038`.

One exact pass over all 540,164 train structures improves the validation ratio
from `0.70396` to `0.63864` and reaches `0.03916 A` at `t=.005`, but the ratio
still fails. A fixed 256-graph causal audit rejects short-range RBF, degree,
self-image, and element-pair attributions. Raw Cartesian graph error correlates
with atom count by `0.579--0.643`; dividing the tangent by `V^(1/3)` reduces
this to `0.163--0.231`. The resulting dimensionless chart learns
`V^(-1/3)v_r` and restores `v_r` before the exact `v_f=v_rL^-1` pullback,
leaving the torus path and sampler unchanged. Its exact-one-pass run reaches
ratio `0.58940`, `t=.005/.1` RMS `0.040084/0.05675 A`, and rollout `.1/.2`
RMS `0.05963/0.08444 A`, with zero failures and zero tensor candidates. It
remains a failed H1a result.

A single degree-three STF extension was then tested under the same seed, steps,
data, chart, and sampler. It improves the ratio only to `0.57240` and
`t=.005` to `0.03938 A`, while reducing measured training throughput. This is
useful negative evidence that angular order alone does not pay for its added
complexity. The cubic parameter and runtime path were removed; production
retains only the vectorized degree-one/two operator. No joint initialization,
later Gate, tensor condition, oracle, relaxation, DFT, or DFPT was run.

### Dynamic edges, explicit triplets, and induced slots

A fixed causal sequence then tested whether the remaining gap came from stale
edge state or from compressing all incoming edges into one moment set. Updating
the persistent edge state from current node, vector, radial, graph-state, and
time context at every block, together with small `1e-2` nonzero orthogonal
residual initialization, improves the one-pass ratio to `0.54417`. The low
noise and rollout checks pass, but the `0.5` validation gate does not.

An explicit shell-complete TopK triplet kernel produces ratio `0.56794`, worse
than the dynamic predecessor. Besides the negative result, its neighbor order
can switch under small noisy-coordinate perturbations. A soft R=8 induced-slot
operator uses all edges and is causally active, but its unbalanced run reaches
only `0.54583`; by the final checkpoint deep layers allocate as much as `0.951`
of assignment mass to one slot.

The final local experiment applies six fixed vectorized alternating row/column
normalizations per center, with no auxiliary loss. Its zero-step qualification
has exact global occupancy `0.125`, effective slots `7.916/8`, immediate
assignment/value gradients, BF16/FP32 output/gradient cosines
`0.99994/0.99956`, `218.31 graphs/s`, and `215.72 MiB`. The exact-one-pass run
reaches ratio `0.533141`: an improvement of only `0.011025` over the dynamic
predecessor, below both the `0.5` gate and the preregistered `0.02` material
improvement. Teacher-forced RMS at `t=.005/.1` is `0.037761/0.053899 A`;
rollout RMS from `.1/.2` is `0.054275/0.076667 A`; failures are zero.

Ablating the induced branch worsens validation loss by `82.61%`, so it is used.
Nevertheless it fails all required specialization checks: maximum slot mass
`0.195789 > 0.14`, minimum representation effective rank `1.351 < 2`, and
maximum inter-slot cosine `0.999738 > 0.95`. Shallow assignments are nearly
uniform but produce nearly parallel slot values; deeper learned logits become
sharp enough that six fixed balancing iterations no longer attain their mass
target. This closes the local-aggregation hypothesis. R=16, more balancing
iterations, and additional local operators are not run. TopK, induced slots,
matched initialization, and their runners/configuration dispatch are removed
from active code; the result remains in this history, its compact report, and
Git provenance.

### Middle-noise reciprocal attribution

The next attribution was preregistered before results in commit `70ecfea` and
then run read-only on the frozen dynamic-edge seed-5705 step-8441 checkpoint.
It did not retrain the generator or change H1a. Across
`t={.35,.425,.5,.575,.65}`, same-composition endpoint retrieval averages
`0.403150 < 0.75` and falls from `0.53543` to `0.25984`. The mean normalized
low/high reciprocal-residual ratio is `1.053482 < 1.15`, with `0/5` supporting
times. A closed-form frozen 12-channel low-k ridge probe improves held-out MSE
by only `0.002257`; its matched high-k control is `-0.000682`, giving
`0.002939 < 0.03` low-minus-high improvement. Low-band graph coverage remains
`0.9766--0.9883`, so empty low-frequency shells do not explain the result.

All three independent checks therefore fail and the frozen decision is
`do_not_implement_reciprocal_carrier`. An independent auditor recomputes the
CSV metrics, decision, protocol hash, and artifact hashes and finds no
checkpoint or optimizer artifact in the report. No reciprocal carrier is
implemented and neither failed reciprocal residual is restored. The evidence
redirects future, separately versioned H1a diagnosis toward middle/high-noise
conditional target variance, finite data exposure, probability-path
information, or staged/self-conditioned coordinate generation. It does not
authorize H1b, H2--H6, tensor/oracle training, relaxation, DFT, or DFPT.

An earlier independent Bridge worktree had already reached the same NO-GO on
the volume-normalized step-8441 checkpoint. Its middle-noise low-shell excess
over an atom-permutation null is `0.007755 < 0.10`; the held-out low-frequency
probe explained fraction is `-0.001368`, with only `0.000695` advantage over
random Fourier and `-0.001368` relative to a graph-only token. That audit uses
zero optimizer steps, preserves the checkpoint fingerprint, and passes
translation, permutation, O(3), and GL(3,Z) checks. The main-worktree result is
therefore confirmatory and adds endpoint-identifiability evidence; it is not a
reason to rerun or tune low-k diagnostics. Combined with TopK/slot failure, the
remaining bounded hypothesis is that the model is forming increasingly rich
features on a noise-corrupted neighbor relation. This motivates a separately
frozen clean-topology oracle/probe before any new operator or extra-pass run.

### Complete all-pair clean-topology attribution

The first clean-topology audit was invalid for scientific attribution because
it carried clean labels only on the current noisy production edge set and
covered `0.58261` of clean coordination mass. It is preserved as an audit
failure rather than interpreted as a negative topology result. The versioned
v2 audit repairs only candidate support: it evaluates every directed non-self
atom pair (`N(N-1)<=380`), uses exact float64 periodic CVP for clean labels,
and aggregates all production periodic images with a fixed soft mixture.

Coverage is exactly `1.0`. Across `t=.4/.5/.6`, mean clean/noisy soft Jaccard
is `0.50413` and hard topology switch fraction is `0.26469`. A clean-topology
oracle carrier improves held-out coordinate residual energy by `0.10716`,
compared with `-0.00354` for the current noisy-topology control. Frozen node
and edge states predict clean coordination with middle-noise AUC `0.87923` and
explained fraction `0.61362`. However, substituting probe probabilities into
the oracle linear carrier yields `-0.04391` improvement, or `-0.40976` times
the oracle gain. The decision is therefore
`probe_predictive_but_topology_correction_not_residual_causal`. No production
branch or optimizer step is added.

### Fixed dynamic-architecture two-pass exposure curve

A separately committed protocol then trains the unchanged 5,034,297-parameter
dynamic coordinate model from scratch at seed 5705 for exactly two complete
passes over the 540,164-structure train split. Model, optimizer, EMA, batch
size, time/noise sampling and data-generator order are unchanged. The trainer
records a complete post-global-clip gradient partition and saves checkpoints
at nominal `0/.25/.5/1/2` passes. Validation ratios are respectively
`1.00000/0.73837/0.63348/0.54371/0.49103`. The one-pass point differs from
the archived `0.54417` by only `0.00046`, so the reproduction precondition
passes.

The one-to-two-pass relative validation improvement is `0.096876`. This is
above the preregistered `<=0.05` representation-plateau rule but below the
`>=0.10` undertraining rule, so the frozen classification is `ambiguous`.
Teacher-forced endpoint RMS continues to improve from one to two passes at
`t=.005/.1/.5` (`.03835->.03546`, `.05484->.04980`, and
`.40305->.37155 A`) while remaining uninformative at `t=.9`. Every active
coordinate module retains finite gradients through the second pass. The
post-hoc two-pass ratio below `0.5` does not rewrite H1a because the diagnostic
does not rerun the complete historical acceptance suite.

The read-only exposure-conditioned rerun is complete. Its two-pass middle
clean-oracle gain is `0.09293`, retaining `0.6640` of the quarter-pass effect,
with time-resolved values `0.04099/.09577/.14203` at `t=.4/.5/.6`. This is the
preregistered `mixed` case: exposure absorbs much of the lower-middle-noise
residual but not the high-noise effect. It authorizes neither more exposure nor
a full ACF branch.

### Quotient-Tweedie self-conditioning attribution

A final zero-optimizer attribution uses the same two-pass EMA checkpoint,
512/256 v2 panels, all-pair topology, and original per-time noise streams. It
forms `xhat_0=P_Q[x_t+sigma*predicted_scaled_score]` without reading the clean
endpoint. One clean-oracle-fitted Cartesian carrier coefficient is shared by
all validation topology fields so that variant-specific readout fitting cannot
manufacture causality.

At `t=.6`, the Tweedie field improves topology MSE by `0.31269` over the noisy
field but reaches only `AUC=0.77003 < 0.8`. Its periodic endpoint RMS is
`0.65437 A`. More decisively, its shared-carrier residual improvement is
`-0.04955`, with structure-bootstrap 95% interval
`[-0.06020,-0.03890]`, whereas the clean oracle reproduces `+0.14203`. The
frozen linear probe reaches `AUC=0.81964` but remains non-causal at `-0.05405`.
The decision is
`self_conditioned_topology_not_predictive_revisit_conditional_variance`.
No production ACF, staged Tweedie branch, optimizer step, additional exposure,
sampler search, tensor condition, oracle, relaxation, DFT, or DFPT is added.

### Coordinate clean-side-information contract repair

Two follow-up attributions first show that neither per-variant optimal ridge
carriers nor a matched nonlinear pair-to-vector MLP can turn probe/Tweedie
topology into a held-out residual improvement. The nonlinear incremental gain
is only `0.00537` with a structure-bootstrap interval crossing zero; both MLP
readouts overfit their training panel. This closes deterministic topology
conversion rather than motivating another topology branch.

A separate code-path audit then finds a real task-contract mismatch. Every
coordinate-only rollout holds the true element tokens and lattice fixed, but
the historical coordinate-only DSM trainer still corrupted both variables and
optimized only the coordinate loss. The resulting model was trained on a
strictly harder, mismatched conditional distribution. The repair records
`coordinate_clean_side_information=true` in checkpoint metadata, bypasses the
categorical and VP lattice corruption in both training and fixed-noise
validation, and makes score diagnostics read the same metadata. Coordinate
noise, target, chart, architecture and sampler remain unchanged.

The preregistered seed-5705, 2,111-step screen passes every check. At matched
0.25-pass exposure, validation ratio changes from `0.7383705` to `0.4938223`;
`t=.6` explained fraction changes from `0.1302396` to `0.3907047`. Conditional
rollout RMS from `t=.1/.2` is `0.07684/0.12153 A`, with zero failures. This
attributes a major part of the coordinate-only failure to corrupted observed
side information. It does not retroactively make the old Tweedie carrier
causal, change historical H1a, or authorize later Gates.

The follow-up exact-one-pass protocol changes only exposure. From scratch at
the same seed it presents all 540,164 training structures once in 8,441 steps.
All frozen checks pass: validation ratio `0.332188`, absolute improvement
`0.161635` over the quarter-pass screen, t=.005/.1 endpoint RMS
`.037560/.049194 A`, t=.6 explained fraction `.635090`, and reverse-SDE-100
rollout RMS `.051235/.070390 A` from t=.1/.2 with zero failures. Throughput is
`267.57 graphs/s` and peak allocated CUDA memory is `4917.15 MiB`.

This qualifies the conditional-coordinate substrate with observed chemistry
and lattice. It does not change the historical free-joint H1a result. It also
changes how earlier representation ablations may be interpreted: they remain
valid for their archived mismatched task, but cannot by themselves establish a
representation limit under the repaired contract. Retired branches are not
restored.

### J0 side-information sensitivity and J1 independent clocks

J0 uses the qualified one-pass EMA checkpoint with fixed validation structures
and coordinate noise. At `t_F=.5`, controlled element corruption increases
coordinate score MSE by `5.33520x`, lattice corruption by `5.16257x`, and both
by `9.93850x`. Per-node permutation changes `35.855%` of tokens, the shuffled
lattice control retains graph coverage, and all tensor-candidate counts remain
zero. The coordinate field therefore uses both observed side modalities.

J1 is committed separately at `c4a4201a746969ff87dcbdf7d115a267d18946de`.
It adds independent Fourier clocks `(t_F,t_A,t_L)` and one linear fusion before
the unchanged per-block time FiLM. The coordinate probability path, target,
loss, optimizer, EMA and dynamic Cartesian backbone are unchanged. Each
64-graph batch has exact regime counts `13/13/13/13/12` for clean-clean,
noisy-element, noisy-lattice, diagonal and independent interior states.

The seed-5705, 2,111-step run passes both frozen scientific bounds. Held-out
final/initial ratios are `0.47273`, `0.51407`, `0.56107`, `0.57304`, and
`0.64015` in the five regimes. Clean-clean is below `0.518513`; diagonal is
below `0.664533`. All clock/fusion gradients are positive and finite,
throughput is `247.65 graphs/s`, peak allocation is `4714 MiB`, and the tensor
atlas is bypassed. The composite result supports continuing unified multimodal
diffusion and rejects an immediate hard-chain decision, but it does not isolate
the clock effect from the changed five-regime task mixture or the 3.9% parameter
increase and does not qualify free joint H1a. Separate confidence intervals for
adjacent regimes overlap, so their ordering is descriptive until a
structure-paired difference bootstrap is run.

The next frozen controls are C0 (only `t_F`, with a disconnected
parameter-matching clock bank), C1 (`t_F` plus `(t_A+t_L)/2`) and the existing
C2 (three named clocks), all using the identical mixture, seed, exposure and
capacity. A separate zero-step audit measures per-regime pre-clip norms,
gradient cosines, clip scales and module energy shares before any clipping
policy can change.

J2 requires qualified E1 element and L1 lattice reverse generators followed by
joint M1 training to cross true/generated on-policy side states. J1 is
coordinate-only, so those heads are untrained; no J2 state, joint sample,
tensor result, relaxation, DFT or DFPT is fabricated from it.

### J1 matched attribution and gradient geometry

The parameter-matched controls use the exact J1 mixture, seed, 2,111-step
exposure and 5,232,057 parameters. C0 sees only `t_F`, C1 sees `t_F` and
`(t_A+t_L)/2`, and C2 is the existing three-clock checkpoint. Final MSE for
C0/C1/C2 is `0.55303/0.54216/0.52814` at clean--clean,
`0.62187/0.61225/0.59945` for noisy element,
`0.62257/0.61921/0.62028` for noisy lattice,
`0.67028/0.66935/0.66934` for diagonal, and
`0.75772/0.75505/0.75190` for interior.

The frozen clock-attribution Gate fails. C2-minus-C0 paired 95% intervals are
`[-0.00929,0.00703]` for diagonal and `[-0.01619,0.00446]` for interior, so
neither is strictly below zero. C2 is significantly better for clean and
element-only states and non-inferior by observation elsewhere; separate clocks
remain an information-preserving interface, but they are not the identified
cause of J1's noisy/noisy improvement. The fixed multi-regime task mixture is
the shared sufficient change in this comparison.

The raw C2 step-2111 zero-optimizer audit then evaluates 40 gradients (five
regimes over eight structure microbatches). Median hypothetical clip scale is
`0.26609`, above the frozen severe boundary `0.2`. Every regime-pair median
cosine is positive, the largest negative fraction is `0.125`, and no pair meets
the 75% persistent-conflict rule. Mean gradient energy is `69.48%` coordinate
readout, `18.22%` input/time, `11.52%` dynamic edge/angular, `0.57%` base
blocks, and `0.19%` time fusion. Global clipping is retained; no optimizer
modification is authorized. E1/L1 protocol design may proceed, but no E1/L1,
M1 or J2 run is inferred from these coordinate-only audits.

### Unified product-process interpretation after J1 attribution

The matched controls change the abstraction, not the frozen numbers. Their
result does not support “three named clocks repair noisy/noisy learning.” It
supports training a family of partially observed tasks rather than only the
synchronized diagonal. The production design is therefore written as one
typed reverse field on the heterogeneous product state `X=(A,F,L)` with noise
coordinate `(t_A,t_F,t_L)`. Joint, coordinate-conditional, CSP, staged and
alternating generation are paths through this cube. E1 and L1 qualify field
components, M1 qualifies their shared training, and J2 qualifies on-policy
paths; none is a permanent architecture branch.

The existing equal five-regime sampler is now factored into the production
`FiveRegimeTaskMeasure`. This is a behavior-preserving implementation change:
the `13/13/13/13/12` counts, random-number order, coordinate target, loss and
model capacity are unchanged, while regime identifiers remain audit-only.
The paper also records the exact nested-corruption Fisher/tower identity. It is
currently a theorem and future audit target, not a newly added loss. The
proposed element--lattice--coordinate reverse path was corrected to
`(1,1,1)->(0,1,1)->(0,1,0)->(0,0,0)` in `(A,F,L)` order; the sequence in the
external feedback would otherwise clean coordinates before lattice.

### E1 element-only reverse qualification

E1 then isolates the element component with clean coordinate and lattice side
information, the complete chemical vocabulary, seed 5705 and 2,111 updates.
Four bounded mechanisms fail. Absorbing-mask diffusion lowers teacher-forced
NLL but reaches free site accuracy `0.03843` and exact composition `0/256`.
Uniform D3PM permits token correction yet reaches only `0.06175` site accuracy,
`0.08144` count overlap and exact composition `0/256`. Supplying the same
terminal site logits with oracle target counts raises site accuracy to
`0.70861` and exact assignment to `0.30859`, localizing the main defect to the
global species multiset rather than Hungarian ranking.

A graph-composition head does not repair the free chain: site accuracy/count
overlap/exact composition are `0.05944/0.08684/0`. Its information audit shows
that at `t=.25` the current noisy-token histogram already has overlap `0.85913`
while the learned head compresses it to `0.68352`. The subsequent exchangeable
histogram residual preserves that sufficient statistic and correctly repairs
the low-noise boundary (`t=.25` overlap `0.87534`, exact composition `0.27734`,
clean-token-oracle exact `0.89062`). It still fails at high noise and in free
reverse (`t=.9` overlap `0.08530`; free overlap/site accuracy
`0.06831/0.03396`; exact composition `0/256`).

The combined result rejects another readout head, wider local features, more
exposure or sampler tuning on an independent site-token state. Occupation must
next be tested as `A=(C,Y)`, with an explicit unordered composition `C` and a
count-constrained site assignment `Y`. A read-only audit supports an exact
sparse representation: 540,164 train graphs, at most 20 atoms, 76 active
elements, `99.7408%` with at most four species and a maximum of seven. This
authorizes only composition identifiability and exact synthetic-kernel audits;
no later Gate is opened.
