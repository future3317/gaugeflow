# H1a coordinate-path metric audit v1

Status: **completed; the physical-metric mismatch hypothesis was not
activated. H1a remains failed.**

This no-training audit used 4,096 fixed train graphs, four noise replicates and
the active fractional-torus schedule.  It loaded no checkpoint and changed no
model, sampler or probability path.

The fractional path induces a broad physical noise distribution: at every
audited time the per-graph physical RMS has p95/p05 between `3.0779` and
`3.1342`.  A non-orthogonal unimodular shear changes its induced Cartesian
covariance by median relative Frobenius distance `0.71774`.  These facts confirm
that the path is a fixed Niggli-chart prior rather than an intrinsic Cartesian
Brownian metric.

They do not identify the H1a failure.  The preregistered third criterion did
not pass: log physical-noise RMS correlates with log per-atom cell scale by only
`0.3792--0.3923`, below `0.75`.  At `t<=0.5`, the exact quotient score's
Tweedie endpoint RMS is about `1.75e-7 A`; the active analytic target and
translation quotient therefore close at the informative low-noise times.
The large `t=0.9` endpoint ambiguity is expected after torus mixing and is not
used as a low-noise closure failure.

According to the frozen decision rule, the active path is retained for the
next causal experiment.  The next experiment must determine whether the
unchanged denoiser/head/loss can memorize a fixed finite set and then a fixed
set with resampled time/noise.  This report does not authorize a metric-path
replacement, more seeds, H1b, tensor conditioning or later Gates.

Exact values are stored in `result.json`.
