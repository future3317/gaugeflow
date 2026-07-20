# H1a J1 independent-modality-time attribution

## Answer first

J1 passes every frozen check.  A single coordinate denoiser with explicit
element, lattice and coordinate clocks retains the qualified clean-side task
and materially improves the historical noisy/noisy diagonal task after the
same 2,111-step exposure.  The evidence supports continuing the unified
multimodal hybrid-diffusion hypothesis; it does **not** qualify free joint
generation and does not authorize ACF, tensor conditioning, relaxation, DFT or
DFPT.

This result shows that the composite intervention---a fixed five-regime task
mixture plus explicit modality clocks---handles teacher-forced side-state
uncertainty better while retaining the clean-side task.  It does not yet
isolate clocks from the changed task mixture or the 3.9% parameter increase,
and therefore does not establish that a single clock caused the historical
free-joint failure.  There is still no evidence that the wrapped torus path,
score target, coordinate carrier, data cache or diffusion theory is
intrinsically invalid.

## What was corrected before J1

The preceding clean-side qualification establishes only

\[
p(F\mid A,L,N),
\]

where \(A\) is the clean per-node element-token list.  It does not establish
\(p(F\mid C,L,N)\) from composition counts alone and cannot by itself identify
simultaneous joint diffusion as the cause of free-generation failure.

The zero-training J0 audit then demonstrated that the qualified coordinate
field genuinely uses both side modalities at \(t_F=0.5\): controlled element
corruption increased coordinate score MSE by \(5.335\times\), controlled
lattice corruption by \(5.163\times\), and both together by \(9.939\times\).
This ruled out the degenerate explanation that the clean-side result merely
made the task easier while the network ignored chemistry or lattice.

The Tweedie-topology result is also scoped correctly: it rejects the current
\(g(\hat F_0)\) plug-in carrier, not a directly supervised posterior
\(q_\phi(Z_0\mid F_t,A_t,L_t)\) in principle.  J1 therefore adds no topology
module.

## Mechanism

The coordinate score is parameterized as

\[
s_F^\star\!\left(F_{t_F};A_{t_A},L_{t_L},t_F,t_A,t_L\right)
=\nabla_{F_{t_F}}\log p\!\left(F_{t_F}\mid A_{t_A},L_{t_L}\right).
\]

Three independent Fourier embeddings are concatenated and linearly fused:

\[
h_t=W_t\,[\phi_F(t_F)\Vert\phi_A(t_A)\Vert\phi_L(t_L)].
\]

The fused token replaces only the old scalar-time token.  It enters every
existing message block through the unchanged time FiLM, the coordinate control
gate and terminal graph context.  The torus coordinate path, analytic target,
dynamic Cartesian message operator, loss, optimizer, EMA and reverse process
are unchanged.  An independent-time model fails closed unless both side times
are supplied; a shared-time model rejects unequal side times rather than
silently ignoring them.

Every 64-graph training batch uses the fixed 13/13/13/13/12 allocation:

| regime | \(t_A\) | \(t_L\) | graphs |
|---|---:|---:|---:|
| clean--clean | 0 | 0 | 13 |
| noisy element | \(t_F\) | 0 | 13 |
| noisy lattice | 0 | \(t_F\) | 13 |
| diagonal | \(t_F\) | \(t_F\) | 13 |
| interior | independent stratified | independent stratified | 12 |

No target CIF metadata, material ID, target space group, endpoint token,
tensor, clean coordinates or future state enters the denoiser.

## Frozen protocol and execution

- protocol: `h1a_j1_independent_modality_times_v1`
- implementation/run commit: `c4a4201a746969ff87dcbdf7d115a267d18946de`
- dataset: qualified Alex-MP-20 P1 cache, 540,164/67,520/67,520 split
- training: seed 5705, batch 64, 2,111 steps, 135,104 presentations
- precision/device: BF16 learned matmuls with FP32 geometry, RTX 4060 Ti 16 GB
- model size: 5,232,057 parameters (5,034,297 predecessor)
- validation: same 256 held-out structures and fixed noise for checkpoints 0
  and 2,111; structure-paired 2,000-replicate bootstrap

The two decision thresholds were frozen before training:

\[
r_{\mathrm{clean,clean}}\le 1.05(0.493822)=0.518513,
\qquad
r_{\mathrm{diag}}\le 0.90(0.738371)=0.664533.
\]

## Results

| side-state regime | initial MSE | final MSE | ratio | bootstrap 95% |
|---|---:|---:|---:|---:|
| clean--clean | 1.11722 | 0.52814 | **0.47273** | [0.43557, 0.51057] |
| noisy element | 1.16609 | 0.59945 | **0.51407** | [0.47853, 0.55080] |
| noisy lattice | 1.10553 | 0.62028 | **0.56107** | [0.52322, 0.60308] |
| diagonal noisy/noisy | 1.16805 | 0.66934 | **0.57304** | [0.53773, 0.60740] |
| independent interior | 1.17457 | 0.75190 | **0.64015** | [0.60063, 0.68463] |

All modality-clock gradient norms were finite and positive at every logging
boundary.  At the final step they were 0.03839 (coordinate), 0.01759
(element), 0.03002 (lattice) and 0.03534 (fusion).  Training throughput was
247.65 graphs/s and peak allocated CUDA memory was 4,714 MiB.  The tensor atlas
was bypassed exactly.

The original run logger stored per-regime graph quadratic energy before the
final `/3` coordinate-MSE normalization, although the optimized scalar loss and
all validation results used the correct normalization.  The post-run logger is
corrected; this diagnostic display issue did not affect gradients, checkpoints
or the decision.

## Interpretation

The observed ordering

\[
r_{00}<r_{A}<r_{L}<r_{AA=LL}<r_{\mathrm{interior}}
\]

is consistent with graded cross-modal difficulty rather than a collapsed time
encoder.  Lattice uncertainty is more damaging than element uncertainty under
this budget, while asynchronous interior states are hardest.  The separate
bootstrap intervals of adjacent regimes overlap, however; adjacent differences
must be bootstrapped on paired structures before they are called significant.
The run shows that the 5.23M-parameter network can learn all five contracts
without sacrificing the clean-side corner, but task mixture, clock identity
and nominal capacity remain confounded.

The 97.4% global clipping rate is an important optimization diagnostic.  It
does not invalidate J1 because all models use the preregistered clip and every
clock receives finite gradients, but future matched joint training should
report pre/post-clip norms and module shares as done here.  J1 does not grant
permission to tune the clip threshold retrospectively.

## Decision boundary and next gates

J1 first authorizes a parameter-matched clock attribution and a zero-optimizer-
step gradient-geometry audit.  The former compares single-clock, side-summary
and separate-clock models under the same five-regime mixture and exact
5,232,057-parameter budget.  The latter tests whether the 97.4% clipping rate
reflects persistent cross-regime gradient conflict or only large norms; it does
not change the clip threshold.

Only after those diagnostics may clean-side element and lattice reverse heads
be separately qualified as E1 and L1, followed by joint multi-head M1.  A valid
J2 then needs their on-policy states at the same reverse-clock time.  The
current J1 coordinate-only checkpoint leaves the element and lattice readouts
untrained; it must not be misused as such a generator or used to manufacture a
terminal-state/forward-noise surrogate.  Until those upstream heads are
qualified, J2 remains unexecutable.

If later J2 shows tolerable upstream error, continue one unified multimodal
hybrid diffusion.  If upstream side-state errors dominate, the minimal hard
chain to compare is

\[
p(N,C)\,p(L\mid N,C)\,p(F\mid\widetilde A(C),L),
\]

where \(\widetilde A(C)\) is a randomly ordered element-token multiset.  No
separate post-hoc site-assignment head is implied by this factorization.
