# Chart-consistent fast sampling for GaugeFlow

## Status and boundary

This document is a design and qualification plan, not an active sampler claim.
The current H1a coordinate mechanism must finish under its frozen 100-step
evaluation.  Nothing here changes that result, authorizes H1b/H2--H6, or
replaces the production reverse sampler before a separate qualification.

## Why an image diffusion solver is not a drop-in replacement

One GaugeFlow denoiser call updates three different state spaces:

1. an absorbing 118-element categorical process with an exact finite-step
   reverse bridge;
2. fractional coordinates on a periodic translation quotient, with a wrapped
   Brownian score and a quotient Brownian reverse step;
3. log-volume and trace-free log-shape latents with a variance-preserving
   Gaussian posterior.

DPM-Solver obtains low-NFE accuracy by exploiting the semi-linear form of a
continuous Euclidean diffusion ODE.  Applying the same algebra directly to the
hard element tokens or to unwrapped fractional coordinates would change the
scientific state space.  Conversely, replacing the substrate with CrysBFN
would be a new generator rather than an acceleration of GaugeFlow.  The useful
lesson from these methods is therefore finite-transition learning and
nonuniform time allocation, not their image-space update formula.

The production sampler already supports a finite transition in each component.
Its cost comes from evaluating the graph denoiser at every transition.  The
first question is consequently how far the existing learned posterior can be
jumped before its state dependence makes the approximation inaccurate.

## F0: training-free NFE and grid qualification

Use one frozen checkpoint and common initial noise/random numbers.  Evaluate
`100, 50, 32, 20, 16, 10, 8, 4` denoiser calls.  Do not select a schedule from
the final benchmark.  Use a fixed qualification panel, then freeze one schedule
for a held-out confirmation panel.

Common random numbers must be invariant to the number of transitions.  In
particular, repeatedly calling `multinomial` from one stateful generator is not
a valid coupling because a shorter grid consumes a different random stream.
For each node, pre-sample one reveal threshold \(U_i\) and one 118-class Gumbel
vector.  The token becomes eligible for reveal when the analytic categorical
survival crosses \(U_i\); its element is selected from the current clean logits
with the fixed Gumbel vector and then remains fixed.  Coordinate/lattice base
noise is shared exactly.  The primary integration-error comparison uses the
deterministic continuous update.  A separate ancestral confirmation uses
nested grids and one pre-sampled master Brownian path, aggregating its fine
increments for each coarse interval; independent per-grid bridge noise would
confound solver error with Monte Carlo variation.  This makes step-count
comparisons causal rather than RNG comparisons.

Compare:

- the existing uniform-log-alpha grid;
- uniform model time;
- a shortest-path grid selected only on the qualification panel.

For candidate times \(t_0>\cdots>t_M\), define a measured one-jump cost

\[
c(i,j)=\mathbb E\left[
  d_Q(f_j^{(i\to j)},f_j^{\rm ref})^2
  +d_L(\ell_j^{(i\to j)},\ell_j^{\rm ref})^2
  +\lambda_{\rm cat}\,D_{\rm KL}
       (p_j^{\rm ref}\Vert p_j^{(i\to j)})
\right].
\]

The best exactly-\(K\)-transition schedule is then the dynamic-programming
shortest path

\[
C_K(j)=\min_{i<j}\{C_{K-1}(i)+c(i,j)\}.
\]

The cost table can be evaluated in batches over candidate endpoint times.  It
does not add a learned controller, alter the denoiser, or search continuous
hyperparameters.  The categorical term compares posterior probabilities, not
integer token labels.

Report wall time, denoiser calls, peak memory, terminal masks, composition
validity, lattice guardrail failures, periodic/translation-quotient coordinate
error, lattice error, and distribution-level crystal metrics.  A low pointwise
RMS is insufficient if sample diversity or composition changes.

## F1: quotient hybrid transition-map distillation

Only if F0 shows that coarse jumps, rather than the generator itself, are the
dominant error should a student learn a finite transition

\[
\Phi_\theta(z_s,s,u),\qquad 0\le u<s\le 1,
\]

from rollouts of the frozen qualified teacher.  The student receives Fourier
features of both source and destination noise levels through every message
block.  A single scalar image-style step embedding is not sufficient because
the map length is part of the requested function.

### Coordinates

The coordinate head predicts a translation-free fractional tangent

\[
\Delta_f=P_{\rm tr}D_\theta(z_s,s,u),\qquad
\hat f_u=\operatorname{wrap}(f_s+\Delta_f).
\]

Training compares \(\hat f_u\) and the teacher state with the same periodic
translation-quotient metric used by production.  It must not use an unwrapped
Euclidean MSE or a time-local nearest-image fallback.  Rotation covariance is
preserved because the learned Cartesian tangent, its fractional chart change,
the linear translation projector, and the torus exponential are all
equivariant/covariant operations already qualified in production.

### Lattice

The student predicts the destination clean estimates or finite updates in the
qualified standardized log-volume and trace-free log-shape charts.  Shape is
projected through the blueprint projector after every transition.  No raw
matrix interpolation is introduced.

### Elements

Element states retain the exact absorbing finite-step bridge.  The learned
quantity remains the clean 118-class posterior.  Distillation uses posterior
KL/cross-entropy and fixed common categorical random numbers; it never regresses
integer atomic numbers in a Euclidean loss.  Terminal masks must remain zero.

### On-policy correction and semigroup consistency

A direct teacher-state loss alone creates exposure bias.  Therefore train on
both teacher states and one student-produced state, with gradients through the
student composition:

\[
L_{\rm map}=d_{\mathcal Z}
  (\Phi_\theta(z_s^T,s,u),z_u^T)^2,
\]

\[
L_{\rm rollout}=d_{\mathcal Z}
  (\Phi_\theta(\Phi_\theta(z_s^T,s,u),u,v),z_v^T)^2,
\]

\[
L_{\rm semi}=d_{\mathcal Z}
  (\Phi_\theta(z_s^T,s,v),
   \Phi_\theta(\Phi_\theta(z_s^T,s,u),u,v))^2.
\]

Here \(d_{\mathcal Z}\) is component-wise: quotient distance for coordinates,
qualified latent distance for lattice, and distributional divergence for the
categorical posterior.  Each component is normalized by its teacher transition
energy before fixed equal weighting, so lattice magnitude cannot silently
dominate coordinates or chemistry.

This explicitly addresses the long-span and composition defects observed in
the archived D0.6/D0.7 finite-map experiments: multiscale spans, direct endpoint
supervision, on-policy states, and semigroup checks are qualifications, not
optional afterthoughts.

## Proposed acceptance contract

Qualify `16`, `8`, and `4` NFE against the frozen 100-step teacher with common
random numbers.  Pre-register tolerances before training.  At minimum require:

- zero sampling and terminal-mask failures;
- unchanged translation, rotation, permutation, and cell-consistency tests;
- lattice guardrail and composition validity no worse than the teacher within
  a fixed confidence interval;
- matched distributional coverage/diversity, not merely teacher-state RMS;
- tensor-free qualification first; tensor representative covariance is tested
  only after tensor conditioning is independently authorized;
- measured RTX 4060 Ti latency and peak memory, including graph construction.

One-step generation is not the initial target.  Four to eight transitions are
the defensible first operating point because multistep consistency work shows
that a small number of stages is markedly easier to train and more accurate
than forcing a single global map.

## Orthogonal per-call acceleration

Reducing NFE and reducing the cost of one NFE are separate experiments.  After
the sampler is qualified, the periodic radius multigraph may use an exact
Verlet-style candidate cache: build at `cutoff + skin`, update all current
distances in one vectorized kernel, filter at the unchanged production cutoff,
and rebuild once a certified bound on atomic plus lattice-induced Cartesian
motion exceeds half the skin.  Because the final active-edge predicate is still
the exact cutoff predicate, this can be reference-tested against a fresh graph
at every step.  A stale-neighbor fallback or tolerance-widened acceptance set
is prohibited.  This optimization is expected to help late reverse times,
where displacements are small, and is intentionally not mixed into the NFE
qualification.

## Literature basis

- Lu et al., [DPM-Solver](https://arxiv.org/abs/2206.00927), derives dedicated
  high-order solvers for continuous diffusion ODEs and demonstrates the value
  of schedule-aware low-NFE integration.  Its Euclidean formula is not used
  directly for the hybrid GaugeFlow state.
- Pei et al., [Optimal Stepsize for Diffusion Sampling](https://arxiv.org/abs/2503.21774),
  formulates reference-trajectory schedule calibration with dynamic
  programming.  GaugeFlow's proposed cost replaces an image-space error by a
  hybrid quotient/lattice/categorical metric and requires a held-out schedule
  confirmation panel.
- Song et al., [Consistency Models](https://arxiv.org/abs/2303.01469), and Heek
  et al., [Multistep Consistency Models](https://arxiv.org/abs/2403.06807),
  motivate directly learned finite maps and the practical 4--8-step regime.
- Boffi, Albergo, and Vanden-Eijnden,
  [How to build a consistency model: Learning flow maps via self-distillation](https://arxiv.org/abs/2505.18825),
  provides a flow-map/tangent-condition view and separates Lagrangian rollout
  learning from purely local Eulerian matching.
- Lacombe and Vaidya,
  [Progressive Distillation of Equivariant Latent Diffusion Models](https://arxiv.org/abs/2404.13491),
  provides evidence that diffusion acceleration can preserve geometric
  equivariance when the distillation target and representation do so.
- Wu et al., [A Periodic Bayesian Flow for Material Generation](https://arxiv.org/abs/2502.02016),
  demonstrates that crystal-specific periodic modeling can reach useful
  results at about 10 network evaluations; it is a comparison and design
  signal, not a runtime dependency or substrate replacement.
- Miller et al., [FlowMM](https://arxiv.org/abs/2406.04713), motivates treating
  fractional coordinates and lattice variables on their actual periodic and
  Riemannian spaces rather than importing a flat image representation.
