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
