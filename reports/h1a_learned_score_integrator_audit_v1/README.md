# H1a learned-score integrator audit v1

Status: **completed; no integrator repair qualified**.

Three independent analytic audits show exact endpoint closure for generic,
ordinary real, and high-symmetry InN/BN states.  This read-only audit therefore
holds the seed-5601 EMA network fixed and changes only the score integrator on
forward-noised validation states.  Atom types and lattice are kept at their
clean oracle context so the result isolates coordinate-field rollout
robustness rather than joint categorical/lattice errors.

The existing ancestral kernel is the baseline.  Reverse-SDE GRW,
predictor--corrector GRW, and probability-flow Heun must improve the `t=0.2`
100-step endpoint RMS by at least 20%, avoid more than 5% degradation at
`t=0.1`, remain stable from 100 to 200 steps, and produce no failures.  These
relative thresholds, validation rows and noise streams were frozen before the
first result.

Passing permits only a free-generation diagnostic with the existing
checkpoint; it does not permit retraining or a later Gate.

## Result

No candidate passed the frozen `t=0.2` improvement requirement.  At 100 steps,
the endpoint-RMS ratios to the current ancestral kernel were `1.00001`
(reverse SDE), `1.00978` (predictor--corrector), and `1.00579`
(probability-flow Heun).  Their `t=0.1` ratios were `1.00001`, `1.01058`, and
`0.99435`; all passed the low-noise guardrail but none supplied a meaningful
repair.  Reverse SDE additionally failed the 100-to-200-step stability bound.
All trajectories remained finite.

More importantly, starting from a genuine forward-noised validation state at
only `t=0.1`, every integrator ends near `1.78--1.81 A` mean endpoint RMS after
100 steps.  The exact score closes the same path, while the preceding
teacher-forced audit showed much smaller one-step errors.  This is learned
field error accumulation/off-path instability, not evidence that the Gaussian,
GRW, PC, or Heun discretization is the primary bottleneck.

The production sampler is retained unchanged.  The next separately frozen
mechanism may address coordinate optimization/representation.  In particular,
the existing gradient audit already shows that element and lattice objectives
dominate the shared gradient and global clipping in the low-noise interval;
that evidence supports a coordinate-only pretraining qualification before any
new sampler or conditional mechanism.
