# H1a Rao--Blackwell causal audit v1

Status: **completed; H1a remains failed**.

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

## Result

The EMA field is useful on forward-noised validation states.  At
`t = 0.005, 0.01, 0.02, 0.05, 0.10, 0.20`, the decoded endpoint RMS is
respectively `0.0461, 0.0459, 0.0506, 0.0541, 0.0670, 0.1007 A`, while the
field explains `25.5--37.7%` of the exact quotient-score energy.  The
near-machine-zero oracle endpoint RMS on the same states verifies the endpoint
decoder used by this audit.  The `t=0.9` value is not used to diagnose field
capacity: at that noise level the torus heat kernel is close to uniform, its
score energy is very small, and endpoint inversion is ill-conditioned.

Free-running refinement is not convergent.  With common-stream sampling, the
generated nearest-neighbour median is `2.0798, 2.2231, 2.1110, 1.8324 A` for
`25, 50, 100, 200` steps.  The terminal coordinate increment decreases, but
the structural statistic neither converges nor approaches the train-reference
median `2.6872 A`.  This separates numerical step size from correctness of the
finite reverse transition: simply adding steps is not a valid repair.

## Decision

The Rao--Blackwell target improved the teacher-forced field but did not repair
the rollout distribution.  The next admissible action is a separately frozen,
no-training analytic audit of the wrapped torus reverse kernel, including
cut-locus branch failures and common-random-number step refinement.  Production
sampler code may change only after one mechanism passes that analytic audit.
No additional training seed or training step is authorized, and H1b, tensor
conditioning, oracle work, relaxation, DFT and DFPT remain stopped.
