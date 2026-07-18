# H1a scaled-readout one-state memorization v1

Status: **failed; the standalone reparameterization is rejected. H1a remains
failed.**

The test used exactly the historical first fixed state, seed, noise, BF16
forward, AdamW, global clipping, 4.47M-parameter model and 1,024 steps.  The
only difference was the qualified function-preserving `1024x` final-readout
parameterization.

Final coordinate MSE is `0.40491`, explained fraction `0.45616`, and low-time
endpoint RMS `0.03649 A`; frozen thresholds are `0.001`, `0.995` and `0.01 A`.
The historical unscaled one-state MSE was `0.34414`, so this is not even a
practical improvement.  Sampling failures and tensor candidates remain zero.

Although the exact scaled-readout solution has norm `2.03`, Adam moves those
parameters by only `0.03936`.  Constant gradient scaling is largely cancelled
by Adam's normalization and global clipping, while the `3.5e7` correlated
condition number remains.  A pure powers-of-two reparameterization therefore
does not repair learning and must not be used alone.

Any successor must address correlation as well as magnitude.  The only
mechanistically supported combination is a separately frozen scaled variable
projection, preceded by a BF16 exact-solve stability check; no scale, ridge,
step or seed search is authorized.

Exact metrics and the full curve are in `result.json`.
