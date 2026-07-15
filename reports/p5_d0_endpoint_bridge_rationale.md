# P5-D0 endpoint-bridge rationale

## Observed obstruction

P5-D0.4 uses a deterministic straight coordinate interpolant from 64 distinct
sources to one quotient endpoint. At the terminal time, all sources have the
same translation-equivalence class of geometry, while the raw target velocity
still equals the source-dependent displacement. A translation-invariant model
therefore cannot make raw terminal velocity a single-valued function of its
input. Increasing message-passing capacity cannot remove this conditional
target variance.

This is consistent with D0.4: velocity MSE rises to `0.07480` at `t=1`, while
the endpoint RMS there is numerically zero because no time remains. The full
trajectory fails well before this endpoint, so the original vector decoder is
also augmented with a direct metric edge-displacement field.

## Implemented idea: quotient endpoint bridge

For the translation quotient projection \(P\), D0.5 trains

\[
r_t=P\operatorname{Log}_{x_t}(x_1), \qquad
\mathcal L=\lVert r_\theta(x_t,t)-r_t\rVert^2.
\]

Unlike the straight-path raw velocity, \(r_1=0\) for every source. The target
is therefore single-valued at the collapsed endpoint. For a uniform grid, the
sampler applies

\[
x_{k+1}=x_k+\left(1-\frac{1-t_{k+1}}{1-t_k}\right)r_\theta(x_k,t_k),
\]

with torus wrap and graphwise translation projection. If the residual is
exact, this is the exact linear bridge contraction; it avoids explicitly
dividing a neural prediction by \(1-t\) in the last Euler step.

The coordinate decoder is augmented by

\[
r_i^{\mathrm{edge}}=\sum_{j\ne i}\alpha_{ij}(x,t)\,\Delta r_{ij},
\]

where \(\Delta r_{ij}\) is the closest-image Cartesian displacement and
\(\alpha_{ij}\) receives node features plus a distance RBF. It is translation
invariant and SO(3)-covariant, while preserving the existing vector-message
field as an additive term.

## Literature basis

- Lipman et al., *Flow Matching for Generative Modeling* (2023),
  [arXiv:2210.02747](https://arxiv.org/abs/2210.02747): flow matching permits
  alternate probability paths; the learned marginal field is determined by
  the chosen path/coupling.
- Tong et al., *Improving and generalizing flow-based generative models with
  minibatch optimal transport* (2023), [arXiv:2302.00482](https://arxiv.org/abs/2302.00482):
  source-target coupling is a primary modeling decision.
- Albergo, Boffi, and Vanden-Eijnden, *Stochastic Interpolants* (2023),
  [arXiv:2303.08797](https://arxiv.org/abs/2303.08797): interpolant choices
  define distinct valid flow/diffusion constructions.
- Song, Meng, and Ermon, *Denoising Diffusion Implicit Models* (2021),
  [arXiv:2010.02502](https://arxiv.org/abs/2010.02502): endpoint prediction
  supports deterministic bounded update parameterizations.

D0.5 is a small, pre-registered substrate qualification only. It is not a
tensor-conditioned result and does not alter P5-D0.4.
