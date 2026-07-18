# H1a coordinate memorization scaling audit v1

Status: **completed; failed already on one fixed state. H1a remains failed.**

The unchanged seed-5914 model was independently restored and trained for the
same 1,024 steps on nested panels containing 1, 4 and 16 exact noisy states.
The completed 64-state result was referenced rather than rerun.  Every panel
used the same model, optimizer, BF16 path, target, clipping and frozen
thresholds.

| fixed states | coordinate MSE | explained fraction | low-time endpoint RMS |
|---:|---:|---:|---:|
| 1 | 0.34414 | 0.53777 | 0.03336 A |
| 4 | 0.18782 | 0.78898 | 0.02380 A |
| 16 | 0.16265 | 0.78357 | 0.02449 A |
| 64 (referenced) | 0.28273 | 0.63712 | 0.03297 A |

All panels miss `MSE <= 0.001`, explained fraction `>=0.995`, and low-time
endpoint RMS `<=0.01 A`.  The single-state graph contains 11 sites and is not a
trivial one-site quotient.  Its pre-update training loss oscillates between
roughly `0.23` and `0.59` late in optimization, and the post-update evaluation
is `0.34414`; finite gradients and decreasing envelopes confirm that the path
is active but do not show stable exact fit.

The non-monotone panel ordering rejects a simple finite-table capacity law.
The next allowed action is no-training tangent analysis of this exact first
state: quotient output rank, target projection residual, NTK conditioning and
modulewise gradient energy.  That analysis must precede any head,
preconditioning or optimizer change.  This result does not authorize more
steps, another seed, production initialization, H1b, tensor conditioning or
later Gates.
