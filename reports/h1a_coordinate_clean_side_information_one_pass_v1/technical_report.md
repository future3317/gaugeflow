# H1a conditional-coordinate substrate: one-pass qualification

Status: **passed within its preregistered conditional scope**.

This experiment isolates coordinate generation from de-novo composition and
lattice generation.  The denoiser observes the clean element tokens and clean
lattice, while only fractional coordinates follow the translation-quotient
torus path.  Seed 5705 was trained from scratch for exactly one shuffled pass:
8,441 optimizer steps and 540,164 graph presentations.  No tensor condition,
topology branch, sampler change, extra capacity, pretrained weights, or
post-result threshold change was used.

## Frozen result

| Check | Frozen requirement | Observed | Pass |
|---|---:|---:|:---:|
| validation final/initial coordinate loss | <= 0.463822 | 0.332188 | yes |
| improvement over the 0.25-pass screen | >= 0.03 | 0.161635 | yes |
| endpoint RMS at t=.005 | <= 0.04 A | 0.037560 A | yes |
| endpoint RMS at t=.1 | <= 0.08 A | 0.049194 A | yes |
| explained score fraction at t=.6 | >= 0.50 | 0.635090 | yes |
| rollout RMS from t=.1 | <= 0.50 A | 0.051235 A | yes |
| rollout RMS from t=.2 | <= 1.00 A | 0.070390 A | yes |
| sampling failures | 0 | 0 | yes |
| throughput | >= 220 graphs/s | 267.57 graphs/s | yes |
| peak allocated CUDA memory | <= 5200 MiB | 4917.15 MiB | yes |
| tensor candidates | 0 | 0 | yes |

The EMA validation coordinate loss changes from 0.972845 at step 0 to
0.481502 at step 2,111 and 0.323167 at step 8,441.  The full numerical record
is `result.json`; plotted values are unsmoothed and are not substituted for the
JSON acceptance source.

## Root-cause attribution

### Code and task contract: confirmed primary cause of the recent failure

Historical coordinate-only sampling fixed the true elements and lattice, but
historical coordinate-only training corrupted both while optimizing only the
coordinate loss.  At the same seed and 2,111-step exposure, repairing this
contract changed the validation ratio from 0.738371 to 0.493822 and the t=.6
explained fraction from 0.130240 to 0.390705.  Extending the repaired task to
one exact data pass yields the qualified result above.  This is direct causal
evidence that the earlier coordinate conclusion was dominated by a
training--inference mismatch rather than a broken cache or an intrinsically
unlearnable coordinate field.

The active implementation now has one explicit contract flag recorded in the
checkpoint.  With it enabled, categorical corruption and VP lattice
corruption are bypassed, while the torus noise, quotient target, Cartesian
tangent chart, architecture, and reverse sampler are unchanged.  Element and
lattice heads receive zero gradients.  `condition_present=false` bypasses the
Cartesian atlas and produces zero candidates; the small logged `tensor_atlas`
gradient belongs only to the learned constant null token.

The one-pass evaluator also contained a separate software defect: it required
a historical hash key owned only by the quarter-pass protocol, even though the
one-pass protocol inherits that evidence through the frozen screen protocol
and result.  The evaluator now constructs protocol-specific hash contracts and
has regression tests.  The plotting script was updated to read the current
nested production JSONL schema and current result keys instead of the retired
schema.

### Data: qualified and sufficient for this isolated task

The Alex-MP-20 source contains 675,204 valid structures.  The active
formula/prototype/StructureMatcher-disjoint split contains
540,164/67,520/67,520 train/validation/test structures.  All rows were rebuilt;
the maximum source-equivalence error is 8.10e-15 A and the maximum FP32 cache
error is 2.79e-6 A.  This run consumed every training structure exactly once.
Consequently there is no evidence that damaged structures, an empty effective
dataset, or a small-data regime explains the conditional-coordinate failure.

More data exposure did matter: the repaired ratio improved from 0.493822 at a
quarter pass to 0.332188 at one pass.  This does not authorize indefinite
training; it establishes that the previously repaired task had not exhausted
the available valid data.

### Representation: functional, but old ablations are contract-confounded

The compact dynamic persistent-edge Cartesian model is non-degenerate on the
correct task: input/state embeddings, message blocks, dynamic angular edge
updates, and the coordinate readout all receive finite gradients; teacher-
forced score quality and free-running closure improve together.  The result
rules out a missing output direction, disconnected coordinate head, or a
sampler-only explanation for this task.

TopK triplets, induced slots, reciprocal features, and most topology
attributions were evaluated before the clean-side-information repair.  Their
frozen results remain valid statements about the historical mismatched task
and budget, but they are not clean evidence that those representations are
universally ineffective under the repaired contract.  Restoring every retired
branch would be wasteful and would reintroduce code redundancy.  Any future
representation comparison must start from this qualified contract and add
only one residual mechanism justified by a remaining measured error.

### Method and theory: the unresolved problem is de-novo joint generation

This qualification is conditional coordinate generation, not free crystal
generation.  The historical joint model independently corrupts element type,
lattice, and coordinates and asks one 5.03M-parameter field to infer all three
coupled clean variables.  Clean elements and lattice reduce coordinate-score
conditional variance sharply, so the remaining joint failure is evidence
against assuming that this simultaneous parameterization is an efficient
learning problem for the current backbone.  It is not evidence that score
matching on a torus is theoretically invalid.

The next non-degenerate design should use the exact probability-chain
factorization

\[
p(N,C,L,F,A)=p(N,C)\,p(L\mid N,C)\,p(F\mid N,C,L)\,
p(A\mid F,L,N,C),
\]

where `N` is site count, `C` composition, `L` lattice, `F` a species-free or
composition-conditioned periodic geometry, and `A` the occupational
assignment.  This factorization does not restrict the representable joint
distribution when every conditional is expressive; it changes the statistical
and optimization geometry.  The qualified model in this report directly
supplies the `p(F|N,C,L)` substrate.  Composition counts must be generated, not
read from the target, and the final assignment must be sampled as a complete
permutation-equivariant categorical object rather than independent site
argmaxes.  Parent symmetry may be a prior or latent route, not a hard terminal
constraint.

The previous Tweedie-topology result is also theoretically unsurprising.  For
nonlinear, discontinuous topology extraction `g`, in general

\[
g(\mathbb E[F_0\mid F_t]) \ne \mathbb E[g(F_0)\mid F_t].
\]

Thus a better posterior-mean coordinate estimate need not give a calibrated
bond/topology posterior, and a predictive topology probe need not provide a
causal coordinate residual.  No ACF or deterministic Tweedie topology branch
is authorized by that result.

## Relation to original literature

- MatterGen jointly corrupts atom types, coordinates, and lattice, but couples
  this with a substantially larger GemNet-dT-style directional backbone and
  Alex-MP-20-scale pretraining.  Joint corruption is therefore a demonstrated
  option, not evidence that a much smaller field has the same optimization
  behavior: <https://huggingface.co/papers/2312.03687>.
- FlowMM shows that manifold-aware flow matching and physically chosen base
  distributions can reduce integration cost and simplify lattice learning;
  it does not validate driving a flow ODE with the present diffusion-trained
  score: <https://huggingface.co/papers/2406.04713>.
- CrysBFN motivates variance-reduced distribution-parameter dynamics and
  entropy conditioning for periodic variables, supporting a future
  probability-path audit rather than another local feature branch:
  <https://huggingface.co/papers/2502.02016>.
- CDVAE predicts aggregate composition, lattice, and atom count before
  conditional coordinate/type denoising, providing an empirical precedent for
  separating global variables from local refinement:
  <https://huggingface.co/papers/2110.06197>.

## Boundary and next decision

The conditional-coordinate substrate is qualified.  Historical free joint
H1a remains failed.  This result does not authorize tensor conditioning,
oracle training, relaxation, DFT, or DFPT.  The next implementation should be
a separately specified triangular de-novo generator using the same qualified
coordinate field, followed by a small closure test for generated composition,
lattice, assignment, and geometry.  It should not revive the old simultaneous
joint path as a silent fallback.
