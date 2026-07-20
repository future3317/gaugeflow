# E1.1 count-projected self-correcting categorical path

## Evidence that selects the mechanism

E1 learned a nontrivial teacher-forced element estimator (mean validation NLL
ratio `0.62856`) but failed free reverse generation: site accuracy was
`0.03843` and exact composition accuracy was `0/256`.  The subsequent frozen
exposure audit separates two effects.  With all sites masked, top-1 accuracy is
only about `0.084--0.088`.  More decisively, at observed element time `0.4845`,
on-policy accuracy on the remaining masked sites is `0.02970`, whereas replacing
already revealed tokens by their true values raises the same prediction to
`0.42574`.  About `95.6%` of on-policy revealed tokens are wrong.  An absorbing
reverse kernel copies all of those errors forever.

The selected repair therefore changes one coherent object, the element
probability path and terminal quotient decoder.  It does not add another local
geometry carrier, training seed, target-composition input, or sampler search.

## Mathematical definition

Let (K=118), (U=\mathbf 1\mathbf 1^\top/K), and
(a_t=\cos^2(\pi t/2)).  The forward categorical kernel is

\[
\bar Q_t=a_t I+(1-a_t)U.
\]

It has the uniform element distribution at the noisy endpoint, and unlike an
absorbing state it gives every intermediate token nonzero probability of being
revised.  The implementation never materializes a (K\times K) matrix per
site.  For (s<t), define (a_{t|s}=a_t/a_s).  Given current token (x_t=\ell)
and clean posterior (p_\theta(x_0=k\mid x_t)), the exact D3PM posterior is

\[
p_\theta(x_s=j\mid x_t=\ell)
=\sum_k p_\theta(k\mid x_t)
\frac{\bar Q_s(k,j)Q_{t|s}(j,\ell)}{\bar Q_t(k,\ell)}.
\]

Because every kernel is a diagonal-plus-rank-one matrix, this sum is evaluated
in (O(NK)), not (O(NK^2)).  At (s=0), the expression reduces exactly to
the predicted clean posterior.

The same site posteriors define a model-predicted graph composition

\[
\hat q_g(k)=\frac1{N_g}\sum_{i\in g}p_\theta(A_i=k\mid X_t),
\]

with an equal-weight composition cross entropy against the training label.
The label is an objective target only; it is never a denoiser input.  At the
terminal step, largest-remainder rounding converts (N_g\hat q_g) into integer
counts summing exactly to (N_g), and a (N_g\times N_g) Hungarian solve assigns
the model-predicted species slots to sites using the site logits.  Thus exact
count preservation uses only generated probabilities and node count, never the
target formula or target counts.

## Relation to prior work

- Austin et al., *Structured Denoising Diffusion Models in Discrete
  State-Spaces* (D3PM, 2021), <https://hf.co/papers/2107.03006>, supplies the
  general discrete transition/posterior construction.
- Shi et al., *Simplified and Generalized Masked Diffusion for Discrete Data*
  (MD4, 2024), <https://hf.co/papers/2406.04329>, shows why absorbing masked
  diffusion and cross-entropy training are strong baselines, while also noting
  finite-step conflicts when several tokens are generated together.
- Wang et al., *Remasking Discrete Diffusion Models with Inference-Time
  Scaling* (ReMDM, 2025), <https://hf.co/papers/2503.00307>, identifies the
  inability to revise an unmasked token as the failure-to-remask property and
  derives correction-capable reverse processes.

GaugeFlow does not copy a language sampler.  Its additional material-specific
step is permutation-invariant graph-composition prediction followed by exact
integer count projection and site assignment for (N\le20).

## Frozen boundary

E1.1 keeps seed `5705`, `2,111` optimizer steps, the full 118-element
vocabulary, clean coordinate/lattice side information, and the unified
separate-clock Cartesian backbone.  Failure cannot be rescued with more steps,
seeds, a target-specific vocabulary, target composition, L1/M1, tensor work,
or physical calculations.
