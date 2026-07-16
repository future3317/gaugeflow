# Cartesian stratified gauge atlas v1

Status: **S0.4-v1 remains `failed_no_advance`; S0.4.1 runtime successor passed; untrained**.

This version replaces the active production denoiser's finite-Hopf harmonic
conditioner with `gaugeflow.production.cartesian_gauge_atlas`.  It does not
alter frozen S0.1--S0.2 reports: their Hopf/harmonic implementation now lives
under `gaugeflow.production.archive_harmonic` and is importable only by the
historical audit/tests.  There is no runtime switch or fallback from the atlas
to the archived conditioner.

## Cartesian query algebra

For each normalized Cartesian edge direction `n`, the active query encoder
uses the symmetric-trace-free tensors

\[
n_a,\qquad n_an_b-\delta_{ab}/3,\qquad
n_an_bn_c-\{n_a\delta_{bc}+n_b\delta_{ac}+n_c\delta_{ab}\}/5.
\]

They are the Cartesian realizations of the degree-one, degree-two, and
degree-three SO(3) irreps.  Learned scalar edge weights aggregate them at
nodes; only invariant norms enter subsequent scalar message blocks.  A
Cartesian epsilon contraction produces the polar rank-two contribution, and
four explicit rank-three embeddings yield two graph-level rank-three queries.
No spherical-harmonic or Clebsch--Gordan call occurs on the active denoiser
path.

## Stratified atlas and posterior

The current state and the input rank-three tensor each yield a symmetric
Cartesian frame covariant. Let their right-handed eigenframes be `F_x` and
`F_e`. The 24 proper signed-permutation matrices are atlas centres, not a
complete SO(3) quadrature. The production finite prior is

\[
\mathcal A(x,e)=
\{F_xP_LQ_qP_RF_e^\top:
P_L,P_R\in\mathcal P^+_3,\ q=1,\ldots,7\},
\]

where `P^+_3` is the proper signed-permutation group of size 24 and `Q_q` are
seven fixed non-identity chart nodes. The generic set therefore contains
4,032 rotations and is closed under independent eigenframe relabeling on both
sides. The conditioner scores every candidate through Cartesian rank-three
contractions and marginalizes with a softmax; it never chooses one candidate
as a canonical orientation.

The eigenspaces define a **descriptor-frame ambiguity group**, not the physical
crystal or tensor stabilizer. In general only

\[
G_x\subseteq\operatorname{Stab}(C_x),\qquad
H_e\subseteq\operatorname{Stab}(C_e)
\]

is justified. The normalized lower and upper eigenvalue gaps are mapped by a
compact cubic smoothstep over `[eta/2,2 eta]`. Their products form a partition
of unity over generic, lower-doublet axial, upper-doublet axial, and
descriptor-isotropic charts. Hence a chart enters and leaves with zero value
and zero first derivative rather than through a hard `gap < eta` branch.

A double eigenvalue keeps the same full-group cubature and additionally
marginalizes the affected side over an eight-node SO(2) rule. A physically
zero tensor uses invariant-only conditioning. A nonzero rank-three descriptor
whose quadratic covariance is isotropic is different: it retains directional
information and is evaluated with the fixed global Cartesian cubature. This
fallback is a finite replacement prior, not a claim that covariance isotropy
is the tensor's full physical stabilizer and not a hidden harmonic runtime
path.

## Discrete measure and multiplicity correction

The raw tuple enumeration is not treated as an unweighted set. It defines the
state-dependent discrete measure

\[
\nu_{x,e}=\sum_{R\in\mathcal A(x,e)}w_R(x,e)\,\delta_R,
\qquad \sum_Rw_R=1.
\]

The partition-of-unity mass of each chart pair is distributed uniformly over
that pair's raw candidates. Rotations equal to tolerance `1e-7` are then
deduplicated and their masses are summed. The posterior and aligned tensor are

\[
p_\theta(R\mid x,e)=
\frac{w_R(x,e)\exp(s_\theta(R;x,e)/\tau)}
{\sum_{R'}w_{R'}(x,e)\exp(s_\theta(R';x,e)/\tau)},
\]

\[
\bar e(x,e)=\sum_Rp_\theta(R\mid x,e)\,\rho_3(R)e.
\]

Consequently enumeration order and measure-preserving duplicate expansion do
not change either posterior or aligned tensor. The current prepared diagnostic
finds 4,032 raw / 4,032 unique generic rotations, 32,256 / 8,064 one-sided
axial rotations, and 258,048 / 16,128 rotations in the worst two-sided axial
case. The equality tests give zero FP64 posterior and aligned-tensor change.

## Finite-prior covariance theorem

Assume the discrete measure obeys the pushforward relation

\[
\nu_{gx,\rho_3(h)e}
= (L_gR_{h^{-1}})_\#\nu_{x,e}
\]

and the scalar score obeys

\[
s_\theta(gRh^{-1};gx,\rho_3(h)e)=s_\theta(R;x,e).
\]

Changing variables `R'=gRh^{-1}` in the finite weighted sum preserves its
normalizer and gives

\[
\begin{aligned}
\bar e(gx,\rho_3(h)e)
&=\sum_R p_\theta(R\mid x,e)
  \rho_3(gRh^{-1})\rho_3(h)e\\
&=\rho_3(g)\sum_R p_\theta(R\mid x,e)\rho_3(R)e\\
&=\rho_3(g)\bar e(x,e).
\end{aligned}
\]

This is an exact theorem about a covariant finite measure. It neither assumes
nor concludes convergence to Haar measure. The descriptor-isotropic fixed-node
fallback is therefore audited separately as a finite-prior approximation and
must not be used to overstate exact arbitrary-rotation covariance at that
singular stratum.

## Prepared numerical checks

The S0.4 runner now includes candidate multiplicity/order/expansion checks,
generic-to-axial and axial-to-isotropic perturbations, finite gradients,
candidate-count switches, axial refinement, and synthetic relative-rotation
coverage. The prepared axial refinement has successive normalized differences
`0.035965, 0.005285, 0.00000391` for `K=8,16,32,64`. On eight fixed synthetic
rotations, the largest nearest-candidate geodesic distance is `0.24778 rad`.
These values were initially pre-run implementation diagnostics and were later
reproduced within the official S0.4-v1 run.

The production condition orbit remains proper SO(3).  Improper operations are
not atlas candidates and remain exclusively in the full-O(3) physical Reynolds
compatibility router.

## Versioned qualification

S0.3-v1 is frozen as the failed 24-frame-only implementation. S0.4-v1 defines
the Cartesian atlas as a replacement prior and was formally run on 2026-07-16.
Its candidate deduplication/multiplicity, descriptor-frame ambiguity,
soft-stratum continuity, axial refinement, synthetic coverage, and mixed-
precision checks all passed. The only failure was the frozen CUDA latency threshold:
`41.89 ms/forward > 20 ms/forward` on the RTX 4060 Ti. The official decision is
therefore `failed_no_advance`; no training is authorized. The earlier
4,032-candidate measurements remain exploratory pre-audit evidence. The
archived Hopf difference is named
**archived-reference prior displacement** and is diagnostic only.

The separately versioned S0.4.1 runtime protocol leaves that result immutable.
Profiling identified the per-forward quantized `torch.unique` sort as redundant
on an interior generic chart: the prequalified base 4,032-node rule is unique,
and two-sided multiplication by proper rotations is bijective. The runtime now
caches the base cubature and takes this proven-unique path only for
generic--generic or descriptor-isotropic--descriptor-isotropic interiors.
Axial and mixed chart pairs still use the full multiplicity-corrected
deduplication. Against the original deduplicated measure, aligned tensor,
sorted-posterior, and prior errors are `2.17e-15`, `4.80e-16`, and `0`.
Official RTX 4060 Ti latency is `14.62 ms/forward` and peak memory is `15.19 MB`,
passing the unchanged `20 ms / 64 MB` thresholds. This authorizes S1a
implementation/qualification preparation only; no training has started.

These operator/runtime results do not authorize tensor-conditioned training, real
tensor/oracle work, relaxation, DFT, or DFPT. A later training gate remains
separately versioned and must be explicitly authorized.
