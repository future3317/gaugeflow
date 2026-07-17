# H1a tensor-free benchmark v1

**Decision: failed; H1b remains closed.** This report freezes the first full
tensor-free benchmark exactly as run. It does not authorize blueprint learning,
tensor conditioning, an oracle, relaxation, DFT, or DFPT.

The benchmark generated 256 structures from each of the three 5,000-step EMA
checkpoints and compared them with 8,192 fixed held-out Alex structures. All
768 reverse trajectories completed, no terminal MASK tokens remained, every
lattice was finite with positive volume, and the hard 0.5 A close-contact
guardrail passed both per seed and in aggregate. Element marginals, normalized
volume, formula uniqueness, and the held-out volume envelope also passed.

| Metric | Result | Frozen bound | Check |
|---|---:|---:|:---:|
| sampling failures | 0 | 0 | pass |
| terminal masks | 0 | 0 | pass |
| finite positive lattices | 1.00000 | 1.00000 | pass |
| minimum distance >= 0.5 A | 0.97917 | >= 0.97000 | pass |
| element marginal JSD | 0.02195 | <= 0.10000 | pass |
| normalized volume Wasserstein | 0.10056 | <= 0.50000 | pass |
| formula uniqueness | 0.99609 | >= 0.50000 | pass |
| node-count JSD | 0.27505 | <= 0.01000 | fail |
| normalized nearest-distance Wasserstein | 2.62598 | <= 0.75000 | fail |

The nearest-neighbor failure is a real generator failure. Generated structures
have median nearest distance 1.2711 A, versus 2.7591 A in the fixed test
reference (5th percentiles 0.6293 A and 1.8906 A). The model usually avoids an
outright collision but does not reproduce the local packing distribution.

The node-count check, in contrast, was specified against the wrong reference.
The sampler intentionally uses the empirical **training** node-count prior,
whereas the formula/prototype-disjoint split creates a large train-to-test
shift. The true train-to-test node-count JSD is already 0.26641; for example,
8-node cells comprise 7.83% of train and 52.57% of test. It is therefore
inconsistent to draw node counts from train and require a JSD below 0.01 to
test. The failed v1 result and its threshold remain unchanged. A future
versioned benchmark must evaluate unconditional distribution matching against
train and reserve held-out test structures for novelty, leakage, and
generalization diagnostics.

Post-failure low-noise calibration further attributes the physical failure to
an underfit coordinate field, not merely the reverse integrator. At
`t=0.005/0.01`, the three checkpoints predict only about 12--18% of the
conditional target norm with cosine about 0.28--0.35; at `t>=0.1` the direction
correlation is approximately zero. The corresponding one-step endpoint RMS is
0.72--0.73 A at `t=0.005` and already exceeds 1 A at `t=0.01`. These values are
diagnostic rather than additional acceptance criteria.

Finally, each 5,000-step run consumed only 80,000 graph presentations at batch
16, about 0.148 passes over the 540,164-structure training split. The source
protocol qualified a bounded software/learning substrate; it was not a
converged full-data training budget. Further work must first audit time/head
gradient coverage and then freeze a bounded learning curve before changing the
architecture or adding a physical penalty.
