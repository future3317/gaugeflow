# H1a coordinate memorization audit v1

Status: **completed; failed at the exact-state stage; the resampled-time stage
was not run. H1a remains failed.**

The unchanged 4.47M-parameter denoiser was trained on the same 64 fixed
structures, times, noisy states and coordinate targets for 1,024 steps.  This
is a single-seed causal audit, not production pretraining.  The model used the
active quotient score, PBC radius multigraph, BF16 path, optimizer, loss and
global clipping.  No tensor candidates were constructed.

The loss decreased from `0.78902` to a raw final evaluation of `0.28273`, so
the coordinate head and gradient chain are active.  It nevertheless missed all
frozen exact-state criteria:

| check | observed | threshold | result |
|---|---:|---:|:---:|
| coordinate MSE | 0.28273 | <= 0.001 | fail |
| explained fraction | 0.63712 | >= 0.995 | fail |
| low-time endpoint RMS | 0.03297 A | <= 0.01 A | fail |

Gradient norms were normally above the global clip threshold and reached
`9.50`, but the persistent smooth loss decrease does not by itself identify
clipping as causal under AdamW.  The result establishes that the current
shared structural representation cannot rapidly memorize 64 arbitrary visible
coordinate-score states.  It does not yet distinguish a one-state
forward/head defect from a representation-capacity or sample-efficiency limit.

According to the frozen rule, the independent resampled-time stage stopped
before training.  The next bounded audit reuses this 64-state result and tests
only 1/4/16-state panels at the same 1,024-step budget.  No result here
authorizes more steps, production initialization, H1b, tensor conditioning or
later Gates.
