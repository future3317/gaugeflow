# GaugeFlow-base product-space integration contract

## Status

`draft_before_implementation`.  This document defines the object that replaces
the historical independent site-token reverse path.  It is not evidence for an
A1 checkpoint.  The existing `p(C|N)`, remaining-count assignment, lattice,
and conditional-coordinate Gates remain component evidence only until the
integration Gate below passes.

## State and probability law

For a sampled node count `N`, let `C` be the unordered integer composition,
`B` a carrier stratum, `A` the complete site occupation, `L` the lattice, and
`F` the fractional translation-quotient coordinates.  The required object is

\[
p(N,C,B,A,L,F)=p(N)\,p(C\mid N)\,p(B\mid N,C)\,
p_\theta(A,L,F\mid C,N,B).
\]

`B` is an explicit parent carrier when one is available.  Its action and
occupation blocks define the parent-quotient assignment law.  The universal
flexible/P1 stratum is a genuine state, not a fallback.  In that stratum the
continuous geometry and assignment are sampled jointly; it is mathematically
wrong to pretend that an unlabeled P1 site list alone can define a non-uniform,
permutation-equivariant pre-geometry assignment law.

The first runtime implementation therefore uses a composition-conditioned
product-space reverse process for the flexible stratum, and the established
parent-action quotient law whenever an explicit carrier is selected.  A future
learned `p(B|N,C)` remains distinct from this integration task.

## Exact discrete reverse state

For graph `g`, draw a uniformly random auxiliary site permutation `pi_g`.
At discrete reveal depth `r`, the categorical state is

\[
Y_r=(R_r,\tilde A_r,C),\qquad
R_r=\{\pi_1,\ldots,\pi_r\},
\]

where `tilde A` equals its revealed element on `R` and is `MASK` elsewhere.
The remaining integer state is

\[
c_r=C-\operatorname{hist}(\tilde A_r)\in\mathbb N^{118}.
\]

The denoiser receives only `(Y_r,F_t,L_t,C,N,B)` and never a target endpoint,
material identifier, reveal order, or parent target assignment.  Its next-site
logits `ell_theta` define the normalized exact-count kernel

\[
p_\theta(a_{\pi_{r+1}}=z\mid Y_r,F_t,L_t,C,N,B)=
\frac{\mathbf 1[c_r(z)>0]c_r(z)\exp \ell_{\theta,\pi_{r+1},z}}
{\sum_q\mathbf 1[c_r(q)>0]c_r(q)\exp \ell_{\theta,\pi_{r+1},q}}.
\]

The explicit remaining-count factor is the exchangeable base measure: zero
logits recover uniform sampling over the legal multiset assignments.  After
`N` steps all remaining counts are identically zero.  Thus count
preservation is a property of every sample, not a terminal projection.  The
uniform auxiliary permutation is marginalized by training and sampling; it is
not a site label or a model feature, so the law remains permutation equivariant.
For parent carriers, legal block reveals replace single-site reveals and use the
same remaining-count state with block multiplicities.

Continuous `(F,L)` reverse-SDE steps are interleaved with the discrete reveal
grid.  Their score field is explicitly conditioned on the dense integer
composition `C` through a permutation-invariant graph token.  This makes the
reverse process a model of `p(A,L,F|C,N,B)`, rather than an unconstrained
site-token process repaired after sampling.

## Training objective

For clean `A_0,L_0,F_0`, sample a noise time `t`, a random reveal order, and
the corresponding partial exact-count state.  The A1 product loss is

\[
\mathcal L_{\rm A1}=
\overline{\mathcal L}_{F}+
\overline{\mathcal L}_{L}+
\overline{\mathcal L}_{A},
\]

where each term is a per-graph normalized negative log likelihood/score loss
in its declared path measure.  `L_A` is the masked next-reveal cross entropy
under the legal remaining-count support above.  The three coefficients are one;
any later reweighting requires a new frozen protocol.  `p(N)` and the
absolute-likelihood `p(C|N)` checkpoint are frozen at first integration so that
their already qualified calibration cannot be silently overwritten.

## Required integration Gate

Before A1 training, a no-training and short-gradient Gate must demonstrate:

1. `p(C|N)` samples exactly `N` atoms with zero invalid compositions.
2. Every discrete reverse trajectory has exact composition at every reveal
   depth, terminates with no masks, and has zero failures.
3. Relabeling sites and conjugating a parent action relabels probabilities and
   samples; the flexible path is separately node-permutation equivariant.
4. The complete small-`N` distribution agrees with subset dynamic programming
   and sums to one.
5. Continuous state validation preserves finite right-handed lattices and the
   translation quotient while the discrete state is interleaved.
6. The 34.28M shared backbone has finite forward/backward gradients through
   composition conditioning and assignment logits; no target-only field is
   reachable from the runtime input object.

Only after this Gate passes may an A1 checkpoint be trained.  A passing
integration Gate does not establish OOD carrier generalization, free parent
sampling, material stability, tensor control, relaxation, DFT, or DFPT.
