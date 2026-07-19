# H1a persistent-edge causal attribution v1

This diagnostic freezes the active volume-normalized `l<=2` checkpoint and
does not change the denoiser, probability path, loss, sampler, data, or any
historical H1a decision.  It combines a fresh reproduction of the first 1,000
optimizer steps with fixed-checkpoint layerwise vector probes and analytic
local-environment counterexamples.

## Gradient startup

The seed-5705 replay presents 64,000 training graphs using the original data
order, noise generator, BF16 policy, optimizer, and coordinate DSM objective.
The zero-initialized output maps receive gradients at step 1.  Every serially
upstream path first receives a nonzero gradient at step 2:

| group | first nonzero raw-gradient step |
|---|---:|
| angular scalar/vector output maps | 1 |
| coordinate edge output map | 1 |
| adaptive mixer `U` | 1 |
| angular coefficient projection | 2 |
| edge update | 2 |
| angular scalar/vector internal maps | 2 |
| coordinate edge internal map | 2 |
| adaptive mixer `V` | 2 |

All 1,000 steps and gradients are finite.  Thus the mathematical gradient
delay is real, but it lasts one optimizer step rather than a material fraction
of the one-pass budget.  Small nonzero initialization is a valid confound
removal experiment, not an already established explanation of the failed
learning curve.

## Current-checkpoint span

The table fits one graph-equal FP64 scalar readout to the final 80 Cartesian
carrier channels on fixed validation states.  These are in-panel span ceilings,
not deployable predictors.

| diffusion time | 16-state explained | 64-state explained | current model explained | effective rank (64) |
|---:|---:|---:|---:|---:|
| 0.005 | 0.7583 | 0.5696 | 0.4101 | 17.59 |
| 0.1 | 0.6882 | 0.5939 | 0.5080 | 18.86 |
| 0.2 | 0.6663 | 0.5962 | 0.5253 | 20.34 |
| 0.4 | 0.6461 | 0.4942 | 0.4388 | 19.06 |
| 0.5 | 0.5236 | 0.4422 | 0.3941 | 15.88 |
| 0.6 | 0.3946 | 0.3050 | 0.2679 | 10.67 |
| 0.9 | 0.2031 | 0.0692 | 0.0102 | 3.58 |

The edge-direction probe grows across message depth, for example from
`0.1785` to `0.2801` explained energy at `t=0.2` and from `0.3042` to
`0.3461` at `t=0.5`, but remains below the final carrier.  Vector-state span
also grows, whereas the node-position scalar probe is nearly flat.  This is
evidence that later node/vector context contains useful information that is
not fully refreshed into the persistent edge state; it does not prove that
dynamic refresh alone will pass H1a.

The descriptor-isotropic/axial/generic split is computed from the noisy radius
graph covariance and is therefore a descriptor ambiguity diagnostic, not a
claim about the physical crystal point group.  No stratum is silently removed
from training.

## Direct moment-collision counterexample

An octahedral six-neighbour set and an equal-count triangular-prism set are
both centered spherical two-designs.  In FP64 both have zero first moment and
zero second STF moment to `1e-10`, so a constant-state `l<=2` factorized
operator maps both to numerical zero.  Their fixed fourth-order angular
fingerprints differ by `0.360312`.  Tetrahedral, cubic, and cuboctahedral
families are reported alongside this matched pair in `span_result.json`.

This establishes non-injectivity of the shared low-order moment bottleneck
without inferring it from aggregate validation loss.  It does not imply that
mechanically adding only `l=3`, or any one higher order, is sufficient.

## Decision

The next bounded mechanism may combine per-layer dynamic edge refresh with a
fixed `1e-2` orthogonal initialization of the serial residual output maps.  The
initialization removes a real but one-step optimization confound; dynamic
refresh directly tests the observed edge/node/vector coupling gap.  Data,
seed, optimizer, DSM target, time sampler, radial graph, angular order, and
reverse sampler must remain unchanged.

The `t=0.9` result is not evidence for a missing global feature: DSM predicts
the conditional mean score and may have large irreducible conditional target
variance when endpoint identity is lost.  A reciprocal global branch remains
unauthorized.  It requires a separate `t=0.35--0.65` oracle-recoverability
curve, residual reciprocal-shell spectrum, and frozen low-frequency linear
probe after the local mechanisms are evaluated.
