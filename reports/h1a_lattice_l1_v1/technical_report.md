# Coordinate-free lattice qualification

## Interface repair

The prior lattice heads consumed a graph context derived from periodic edges and
fractional coordinates. That violated the intended factorization
`p(L|C,N,parent)` by allowing a later random variable to enter the lattice law.
The repaired context is

\[
h_L=[\operatorname{mean}_i e(A_i),\ N^{-1/2}\sum_i e(A_i),\
h_t(0,0,t_L),\ h_{\rm state}(L_t,N,C)],
\]

and the two heads predict the clean standardized log-volume residual and the
five-dimensional whitened trace-free log shape. The dedicated training and
sampling interfaces do not accept coordinates and do not construct a radius
graph. The complete hybrid forward uses the same lattice context, so L1 is not
a parallel runtime or fallback.

Repository tests verify invariance to node ordering and coordinate changes,
equality of the full and lattice-only outputs, zero message-block execution, and
zero updates to coordinate/element readouts during a lattice-only optimizer step.
The complete repository closure at implementation time was 335 passed tests,
ruff clean, mypy clean, and no duplicate/unreachable production definitions in
the redundancy audit.

## Probability path and evaluation

Let `z_V` be the standardized residual of log volume after conditioning on
`log N`, and let `z_S` be the whitened five-dimensional trace-free log-metric
coordinate. Both use

\[
z_t=\alpha(t)z_0+\sigma(t)\epsilon
\]

with the cosine VP schedule and direct `z_0` prediction. Training samples only
`t_L`; element and coordinate clocks are fixed at zero. Free sampling starts
from independent standard Gaussian volume/shape latents and applies the same
100-step reverse-SDE kernel used by the joint continuous sampler.

Teacher-forced ratios are measured against the legal zero predictor on the same
4,096 validation structures and noise draws. Free-running distribution metrics
compare one generated lattice per clean validation composition against the held-out
lattice distribution, normalized by each reference interquartile range. Density
uses the exact clean composition and generated volume.

All frozen checks pass. Volume and density are substantially inside their bounds;
the shape-latent W1 (`0.49417`) is the tight check. At high noise, teacher-forced
shape prediction approaches the composition-conditional uncertainty floor
(`0.95481` of the zero baseline at `t=0.9`), while reverse stochastic sampling
still recovers the aggregate held-out shape distribution within the frozen bound.

## Boundary and next action

This result qualifies `p(L|C,N,P1)` only. It does not use or qualify a sampled
parent symmetry family, and it does not test coordinates conditioned on generated
lattices. The next permitted operation is a separately frozen generated-side
coordinate exposure Gate comparing clean-lattice, generated-lattice, clean-
assignment, and generated-assignment error budgets. Joint base pretraining and
the 30M/60M/100M capacity screen remain downstream of that Gate.
