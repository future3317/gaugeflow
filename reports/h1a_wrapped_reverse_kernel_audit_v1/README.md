# H1a wrapped reverse-kernel audit v1

Status: **completed; generic panel passed but did not identify a production
kernel**.

The preceding causal audit found a useful teacher-forced quotient score but a
nonconvergent free rollout.  This no-training gate therefore separates score
calibration from the finite reverse transition on a four-site, three-dimensional
translation quotient.  It uses one fixed endpoint and an equal two-endpoint
mixture, exact analytic heat-kernel scores, common random numbers, and the
pre-registered `25/50/100/200` step grid.

The present sampler is the Euclidean SMLD ancestral approximation of Song et
al. (2021, Eq. 47).  Its conditional Gaussian variance is exact for an
unwrapped Euclidean bridge when the clean endpoint is known; it is not an exact
finite bridge after wrapping, quotienting translation, or replacing the clean
endpoint by a marginal score.  The `exact` wording in the current runtime
docstring is consequently treated as a hypothesis under audit, not as an
established property.

The score-only alternatives follow the reverse-SDE and predictor--corrector
formulas of Song et al. (2021) and their intrinsic geodesic-random-walk form in
De Bortoli et al. (2022).  A finite wrapped bridge that samples endpoint,
translation, and winding posteriors is included only as an endpoint-aware
reference.  Jo and Hwang (2024) motivates the explicit bridge-mixture
interpretation, but the reference is prohibited from production because a
real generated state has no known clean endpoint.

Primary sources:

- Song et al., *Score-Based Generative Modeling through Stochastic
  Differential Equations*, arXiv:2011.13456.
- De Bortoli et al., *Riemannian Score-Based Generative Modelling*,
  arXiv:2202.02763.
- Jo and Hwang, *Generative Modeling on Manifolds Through Mixture of Riemannian
  Diffusion Processes*, arXiv:2310.07216.
- Jiao et al., *Crystal Structure Prediction by Joint Equivariant Diffusion*,
  arXiv:2309.04475 (wrapped-normal crystal predictor--corrector precedent).

No threshold, method, sample count, endpoint, or seed may change after the
first result is written.

## Result

The endpoint-aware wrapped reference passed all frozen checks.  On this
generic endpoint panel, however, all four score-only methods also reached
`1.0` endpoint recovery with zero cut-locus failures at 200 steps.  Their
largest two-endpoint mixture-weight deviations were `0.02539` (ancestral),
`0.01855` (reverse SDE), `0.02051` (predictor--corrector), and `0.01465`
(probability-flow Heun), all below `0.05`.  GPU FP64 execution took 224 s.

This positive result has a narrow interpretation: the exact quotient score
and every tested integrator close a generic, ordered four-site path.  In the
last nonzero-to-zero variance step, the quotient Tweedie map has an unambiguous
local mode and sends the state to machine precision of an endpoint.  The panel
therefore does not reproduce the `0.433/0.612` fractional branch failures seen
for special real four-site structures, and it cannot rank or select a
production kernel.

No sampler is changed from this result.  The next no-training audit must keep
the same analytic score but use a fixed, hashed panel of real validation
endpoints and stratify failures by translation/permutation symmetry and
cut-locus margin.  That is a scope extension, not a threshold revision to this
completed protocol.  H1a remains failed and all later Gates remain stopped.
