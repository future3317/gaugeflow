# H1a coordinate-generator diagnosis

The failed H1a learning curve is attributable to an unlearned coordinate
score, not to a broken analytic path or a coarse reverse integrator. This is a
read-only result and does not change the frozen H1a failure.

Across all three seeds and times `0.01--0.95`, the learned score explains only
`-0.18%` to `+0.10%` of the sampled conditional-score variance. Its norm is
only `0.23%--2.33%` of the target norm and its cosine with the target stays near
zero. The score-based endpoint estimator has `1.08--2.75 A` RMS, whereas the
same estimator supplied with the analytic sampled-lift target closes to
`1.7e-7--4.1e-7 A`. The target and path algebra therefore close, but the
network has learned an almost-zero coordinate field.

A fixed 4,096-structure training reference has no minimum periodic distance
below `0.5 A` (minimum `0.7534 A`, median `2.6982 A`). The generated
close-contact tail is consequently not supported by the training data.
Increasing the reverse discretization does not remove it:

| Reverse steps | fraction >= 0.5 A | minimum distance (A) |
|---:|---:|---:|
| 10 | 0.90625 | 0.3843 |
| 25 | 0.90625 | 0.2920 |
| 50 | 0.96875 | 0.4537 |
| 100 | 0.84375 | 0.2654 |
| 200 | 0.81250 | 0.2605 |

These sensitivity samples share the initial random stream but not a coupled
Brownian refinement, so individual rows are not a model-selection result.
Their non-monotone behavior and degradation beyond 50 steps rule out the claim
that simple step refinement fixes the substrate.

The implementation-level cause is label variance. The current target is the
score of one sampled Gaussian lift. On a torus, the conditional density is a
wrapped normal and its score is the posterior average over all integer lifts.
At medium/high noise the wrapped density is almost uniform and its true score
is almost zero, while a sampled-lift target remains random with order-one
normalized energy. The observed zero prediction is therefore the regression
response to a low-signal, high-variance target.

The next repair is limited to a vectorized exact wrapped-normal target followed
by the translation-horizontal quotient projection. This is the periodic score
used in DiffCSP (Jiao et al., 2023, arXiv:2309.04475), with image/Fourier dual
evaluation to avoid a large periodic-image loop. FlowMM (Miller et al., 2024,
arXiv:2406.04713) independently motivates treating fractional coordinates on a
torus rather than a Euclidean lift. Target Score Matching (De Bortoli et al.,
2024, arXiv:2402.08667) diagnoses low-noise DSM variance, but its known clean
density-score assumption is unavailable here and is not imported.
