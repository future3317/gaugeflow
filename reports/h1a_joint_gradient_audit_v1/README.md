# H1a joint-objective gradient audit

This is a read-only post-failure attribution of the 5,000-step seed-5201 EMA
checkpoint. It does not change the failed H1a benchmark or authorize H1b.

The audit uses 16 fixed training graphs, four fixed noise replicates, FP32
autograd, and eight preregistered times. It measures the coordinate, element,
volume, and shape losses separately, their gradient norms on shared parameters,
and their shared-gradient cosine with the coordinate objective.

| t | coordinate loss | zero-score loss | coordinate shared grad | element shared grad | volume shared grad | shape shared grad |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.78942 | 0.89710 | 1.09343 | 0.00000 | 2.28859 | 4.62245 |
| 0.010 | 0.70585 | 0.80216 | 0.87684 | 0.00000 | 2.58221 | 4.46989 |
| 0.020 | 0.72930 | 0.79828 | 0.48464 | 0.00000 | 2.40682 | 4.89065 |
| 0.050 | 0.54333 | 0.55006 | 0.09789 | 21.89938 | 2.17243 | 4.56566 |
| 0.100 | 0.13172 | 0.13658 | 0.04622 | 10.04399 | 2.24875 | 5.01199 |
| 0.200 | 0.00514 | 0.00514 | 0.01160 | 7.42567 | 2.46933 | 4.57298 |
| 0.500 | 0.00032 | 0.00000 | 0.01651 | 5.29619 | 2.67306 | 3.22075 |
| 0.900 | 0.00022 | 0.00000 | 0.01870 | 4.09353 | 1.20091 | 1.09337 |

The coordinate conditional target carries appreciable energy almost entirely
below `t=0.1`. Uniform random time gives this region less than 10% of graph
presentations. At `t=0.05`, a rare masked token makes the element shared
gradient roughly 224 times the coordinate shared gradient; volume and shape
also dominate. Because observed total gradient norms are normally far above
the global clipping threshold of one, these other heads attenuate the useful
coordinate update. At `t<=0.02` the coordinate shared gradient is finite and
substantial, so this audit does not support an architectural
non-representability diagnosis.

Coordinate-versus-other shared-gradient cosines are generally near zero (the
largest systematic value is the approximately `-0.18` volume cosine at the
three earliest times). Scale and sample allocation, rather than strong
destructive directional conflict, are therefore the primary diagnosed
mechanism.

The next bounded repair is consequently restricted to three correctness and
efficiency changes: true null-condition bypass, skipping unused atlas geometry
queries in tensor-free batches, and randomized stratified uniform-time samples.
The latter preserves the uniform-time marginal objective while guaranteeing
low-discrepancy coverage in every minibatch. It is paired with a one-epoch-scale
learning curve because 5,000 steps exposed only 0.148 training-set passes.
