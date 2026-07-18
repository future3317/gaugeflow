# H1a Rao--Blackwell quotient target v1

Status: **frozen, not run**.

## Identified defect

The product-torus corruption is invariant to a graphwise common fractional
translation, and the denoiser only observes periodic relative geometry.  The
retired target nevertheless evaluated an independently sampled per-site
wrapped-normal score before removing its graph mean.  That projected random
variable is an unbiased score-matching target, but it retains variance from an
unobservable common-translation lift.  At `sigma=0.25` in a fixed exact
three-site audit, its target energy was `0.31317`, whereas the exact quotient
score energy was `0.06421`; their cosine was only `0.54090`.

For displacement `d=x-y`, the translation-quotient heat kernel factorizes
because the production covariance is isotropic in fractional coordinates:

```text
p_Q([x] | [y], sigma)
  = product_{a=1}^3 integral_0^1
      product_i kappa_sigma(d_ia - tau_a) d tau_a.
```

Differentiating under each integral gives the visible quotient score.  The new
target is equivalently the Rao--Blackwell conditional expectation of the old
projected target given the quotient state.  Therefore it changes neither the
forward probability path nor the reverse sampler; it only removes nuisance
variance that no translation-invariant network can predict.

## Production implementation

Each one-dimensional integral uses a 32-node periodic trapezoidal cubature
shifted to the circular mean of that graph and coordinate.  Kernel log density,
site score, graph reduction, posterior normalization, and posterior averaging
are batched with tensor broadcasting and `index_add_`; there are no graph,
site, or quadrature loops in the production path.  Complexity is `O(NQ)` for
`Q=32`.  There is no legacy target fallback and no model parameter change.

Before training, the implementation satisfied:

- exact adaptive three-site image oracle: maximum FP64 error `2e-10` at
  `sigma=0.005, 0.10, 0.25`;
- 32 versus 64 nodes for 20 sites: maximum tolerance `2e-9` through
  `sigma=0.50`;
- translation and node-permutation invariance and horizontal zero graph mean;
- batched versus independent graph equivalence and the analytic narrow-kernel
  limit.

On one real 64-graph/596-site RTX 4060 Ti batch, target construction took
`5.05113 ms` versus `3.52084 ms` for the retired estimator and used `7.84033
MiB` additional peak memory.  A complete BF16 training-step benchmark measured
`329.62 graphs/s` and `1696.43 MiB` peak allocated memory while another process
held GPU memory; these are capacity checks, not cross-run speed claims.

## Frozen execution

Only seed `5601` is run during mechanism iteration: 20,000 steps, batch 64,
unchanged 4.47M-parameter model, optimizer, loss weights, exponential torus
schedule, and 100-step sampler.  The screen requires finite training, final
coordinate validation loss at most `0.47`, total validation ratio at most
`0.65`, generated nearest-neighbour median at least `2.3 A`, zero sampling
failures/masks, and valid positive-volume lattices.

A separately frozen train-reference distribution audit then requires normalized
nearest-neighbour Wasserstein at most `0.75` in addition to the existing H1a
safety and marginal-distribution checks.  No seed replacement or second seed is
allowed during this iteration.  Two further fixed seeds are reserved for the
eventual selected production baseline and paper statistics.

## Frozen result

Seed `5601` completed all 20,000 steps with finite losses and gradients.  The
screen **failed**: final fixed-validation coordinate loss was `0.50350` against
`0.47`, and the 128-sample nearest-neighbour median was `2.26062 A` against
`2.3 A`.  Total validation ratio passed at `0.54325`; all samples had finite
positive-volume lattices, no terminal masks, no sampling failures, and no
tensor candidates.

The separately frozen 256-sample train-reference qualification also failed.
Normalized nearest-neighbour Wasserstein was `0.95302` against `0.75`, nearly
unchanged from `0.95972` for the preceding exponential-schedule model.  Volume
Wasserstein (`0.06443`), element marginal JSD (`0.01434`), lattice validity,
minimum-distance safety and formula uniqueness passed.  Node-count JSD was
`0.01144` versus `0.01`; because node counts are sampled from the exact
empirical prior, that small 256-sample deviation is finite-sample noise and is
not the primary scientific failure.

The correct conclusion is that nuisance-target marginalization modestly
improved teacher-forced coordinate loss and the small screen median, but did
not repair free-running local-geometry fidelity.  H1a remains failed and H1b
is not authorized.  No additional iteration seed or training extension is
run; the next action is the pre-registered read-only causal audit.
