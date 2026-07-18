# H1a 16-state coordinate variable projection v1

Status: **failed and rejected. H1a remains failed.**

The frozen test alternated an exact graph-equal least-squares solve of the 225
affine coordinate-readout parameters every 16 steps with BF16 AdamW updates of
only the nonlinear backbone.  It used the same first 16 fixed states, seed,
path, model capacity and 1,024-step budget as the prior memorization panel.

The method is numerically self-destabilizing.  The initial exact head solution
already has norm `9109.38`; its BF16 training forward reports loss `10.29` and
backbone gradient norm `2.08e4`.  During alternating updates the design
condition number grows from `2.19e5` to `3.61e9`, while the final head solution
norm reaches `4.83e7`.  Intermediate losses reach tens of thousands despite
remaining finite.

The final FP32 solve gives coordinate MSE `0.09460`, explained fraction
`0.87412`, and low-time endpoint RMS `0.02004 A`.  All three scientific
thresholds (`0.001`, `0.995`, `0.01 A`) fail.  There are no sampling failures
or tensor candidates.

This rejects unregularized variable projection and its positive-feedback loop
between a huge affine solution and drifting nonlinear features.  The protocol
does not permit post-hoc ridge, solve-frequency or precision searches.  The
active production head remains unchanged; a successor must improve quotient
node-mode conditioning without solving for enormous readout weights.

Exact curves and solve spectra are in `result.json`.
