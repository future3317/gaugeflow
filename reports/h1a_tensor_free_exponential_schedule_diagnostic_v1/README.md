# Exponential torus schedule distribution diagnostic

This no-training diagnostic uses the failed seed-5401 checkpoint and exactly
the same 8,192-structure train reference, 8,192-structure held-out novelty
reference, 256 generated samples, and 100-step stochastic sampler as the
corrected H1a benchmark.

The diagnostic **failed**.  The normalized nearest-distance Wasserstein value
improved from 1.97287 for the prior linear schedule to 0.95972, but remained
above the frozen 0.75 limit.  The generated median moved from 1.6031 Å to
2.19392 Å versus 2.69824 Å in the train reference.  Node-count JSD was 0.01229
against 0.01; because the node prior itself is unchanged and only 256 samples
were drawn, this small miss is secondary to the coordinate-distribution miss.

The result supports the schedule change as a useful but insufficient repair.
It does not reverse the failed screen, authorize additional seeds, or open
H1b-H6.  The next isolated mechanism must target smooth periodic vector-field
expressivity rather than training duration, loss weights, or another schedule.
