# H1a explicit node-count law

Status: **PASS** for the IID flexible-carrier probability-calibration axis.

GaugeFlow-base now has an explicit source for `N`: the closed-form categorical
maximum-likelihood law fitted only on the 486,340 IID-fit structures.  All
counts `1..20` have nonzero train support.  A separately optimized
unconditional softmax would have the same categorical MLE and adds no
expressive power, so the production implementation uses the exact empirical
law rather than an unnecessary learned duplicate.

Frozen IID evidence:

- calibration/test NLL: `2.4176006 / 2.4176006`;
- uniform NLL: `2.9957323`; gain: `0.5781317`;
- paired bootstrap UCB95: `-0.5701493 / -0.5700022`;
- Jensen--Shannon divergence: `1.00281e-4`;
- integer-support Wasserstein distance: `0.0243824`;
- support coverage: `1.0`.

For 200,000 fixed-seed free draws, the sampled-law JSD was `1.22775e-5`,
integer-Wasserstein was `0.0109932`, and there were zero invalid counts and zero
sampling failures.  Replaying the same seed reproduced every draw exactly.

The original formula/prototype-disjoint validation and test panels are OOD
stress evidence, not pass criteria.  Their JSD values (`0.3470 / 0.2663`) and
integer-Wasserstein values (`1.3834 / 1.6586`) show a strong node-count shift;
the IID qualification must not be presented as OOD generalization.

Boundary: this result qualifies only train-only `p(N)` for the flexible-carrier
branch.  It does not qualify a parent-blueprint law, lattice, coordinates,
joint generation, tensor conditioning, relaxation, DFT or DFPT.

Files:

- `result.json`: complete frozen metrics, checks and categorical probabilities;
- `node_count_diagnostics.png`: IID/OOD node-count comparison.

Implementation commit: `0bd3c8974767d5e9bd4cf12e50318484d60d0f73`.
