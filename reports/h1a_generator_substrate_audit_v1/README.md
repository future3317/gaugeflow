# H1a generator-substrate audit

**Decision: the reviewed substrate defects are confirmed and their replacement
implementation passes the bounded mathematical/software checks. Real-data H1a
learning is not yet qualified.**

The qualified 675,204-row cache was held fixed. No tensor condition, oracle,
relaxation, DFT or DFPT was used.

## Evidence before repair

The frozen 4,096-row graph sample contained 424,160 directed distinct-site
pairs under the old closest-image representation. The complete 8 A periodic
radius multigraph contains 3,739,440 edges, including 363,118 non-zero
self-image edges and 3,283,773 additional image multiplicities. The old graph
also sent 2,232 pairs whose closest image exceeded the cutoff into a biased
message MLP. All 152 sampled one-site cells had a nonempty physical periodic
graph after repair.

The finite 4 A Cartesian wrapped Gaussian did not match the uniform torus
sampler prior. Across all 675,204 cells, only 17.1763% had first-Fourier-mode
residual at most `1e-5`; the median residual was `0.00316` and the maximum was
`0.8780`. This is a probability-path mismatch, not bad source data.

Before lattice standardization, one fixed 16-graph batch gave shared-backbone
gradient norms `16.55 / 0.60 / 15.33 / 0.49` for coordinate, element, volume
and shape objectives (the archived JSON preserves the exact ordering and
pairwise cosines).

## Current replacement

- The graph is the strict set
  `E={(i,j,n): ||(f_j-f_i+n)L|| < 8 A}`, including every image and every
  `(i=i,n!=0)` self image. Reciprocal-column image bounds are complete for
  triclinic cells, candidate construction is on-device, and no per-edge CPU
  sphere decoder remains in the denoiser.
- Current log volume, log shape, `log(N)`, `1/N`, current composition and local
  periodic degree are explicit state features. State FiLM enters every message
  block and all output heads.
- Coordinates use a cell-independent Brownian process on the fractional torus.
  It is a product Markov path with the independently diffused lattice. At
  `sigma_max=1` and `t=0.999`, the leading Fourier residual is about
  `2.7e-9`, below the frozen `1e-8` check and consistent with the uniform start.
- The lattice VP process acts on the standardized training-split variables
  `z_v=(log(V)-log(N)-3.127444)/0.362635` and the five whitened trace-free
  log-shape directions. The normalizer is invertible and stored in checkpoint
  metadata.
- Periodic coordinate aggregation uses variance-preserving `1/sqrt(degree)`
  normalization while an explicit degree token retains coordination count.
  The post-repair shared-gradient norms are `3.51 / 1.00 / 1.90 / 1.33` for
  coordinate, element, volume and shape, so no searched loss weights were
  introduced.

Exact brute-force skew-cell graph tests, one-site self-image tests, fractional
path cell-independence, analytic one-step endpoint recovery, lattice
standardization round trips, translation/cell covariance, checkpoint recovery
and joint sampling all pass. A 10-step real-cache CUDA smoke and two-sample
reverse rollout completed with finite losses and zero sampling failures; these
are software checks only.

The next allowed experiment is the frozen three-seed
`h1a_p1_generator_pilot_v1`. Passing it permits only a longer tensor-free H1a
experiment. H1b--H6 remain blocked.
