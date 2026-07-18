# H1a Rao--Blackwell causal audit v1

Status: **frozen, not run**.

The single-seed target-correction screen failed at final coordinate validation
`0.50350 > 0.47` and generated nearest-neighbour median `2.26062 A < 2.3 A`.
Its fixed 256-sample train-reference audit also failed with normalized
nearest-neighbour Wasserstein `0.95302 > 0.75`, essentially unchanged from the
preceding exponential-schedule result `0.95972`.

This read-only audit separates two remaining mechanisms before any new code is
proposed.  It measures time-resolved conditional score calibration on 128 fixed
validation graphs and compares 25/50/100/200-step stochastic rollouts for 32
common-stream samples.  Weak low-noise calibration implicates the learned
field; strong calibration with nonconvergent step refinement instead
implicates the finite torus reverse kernel.  It changes no weights, sampler,
threshold, seed, or Gate status.
