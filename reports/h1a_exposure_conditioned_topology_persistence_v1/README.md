# H1a exposure-conditioned topology persistence v1

Status: **completed; mixed**.

The exact all-pair clean-topology v2 audit was evaluated at the frozen
0/0.25/0.5/1/2-pass checkpoints. Every checkpoint used the same 512 train and
256 validation structures, four times, five seeds, image mixture, RBF, ridge,
and structure bootstrap. The audit performed zero optimizer steps and all
checkpoint parameter fingerprints were unchanged. Complete results at step 0
and 2,111 were hash-validated and reused; the interrupted step-4,221 artifact
was rejected and recomputed.

| passes | middle clean-oracle gain | retention basis | probe explained fraction | learned carrier gain |
|---:|---:|---:|---:|---:|
| 0.25 | 0.13995 | 1.000 | 0.50805 | -0.06787 |
| 0.50 | 0.14208 | 1.015 | 0.56290 | -0.04874 |
| 1.00 | 0.10683 | 0.763 | 0.61491 | -0.04354 |
| 2.00 | 0.09293 | 0.664 | 0.65384 | -0.04325 |

The two-pass gain is above the exposure-dominant cutoff (`0.05`) and its
retention is above `0.50`, but it misses both topology-persistent thresholds
(`gain>=0.10`, `retention>=0.75`). All three middle-time bootstrap lower bounds
remain positive. The frozen decision is therefore `mixed`.

The aggregate hides a clear time dependence. At two passes the clean-oracle
gain is `0.04099` at `t=0.4`, `0.09577` at `t=0.5`, and `0.14203` at `t=0.6`.
Exposure largely absorbs the lower-middle-noise residual, while a substantial
high-noise topology residual persists. Meanwhile topology probe predictability
increases with exposure, but the fixed linear carrier remains harmful.

This does not authorize a full ACF/latent-pair production branch or added data
passes. The evidence instead motivates one separately frozen, time-localized
self-conditioning/conditional-variance diagnostic: determine whether a
Tweedie endpoint estimate can supply the persistent high-noise coordination
signal before training a new topology module.
