# H1a fixed-panel affine coordinate-readout audit v1

Status: **small panels fit exactly; 16/64-state panels fail with frozen
features. H1a remains failed.**

The audit reconstructed the active final coordinate head as a graph-equal
weighted design matrix of shape `1818 x 225`.  Multiplying it by the actual
model parameters reproduces the production output within `8.50e-7`, so the
captured vector, edge and bias bases are faithful.

| fixed states | exact readout MSE | explained fraction | endpoint RMS (A) |
|---:|---:|---:|---:|
| 1 | `1.55e-27` | `1.0000` | `1.66e-7` |
| 4 | `1.43e-14` | `1.0000` | `1.08e-7` |
| 16 | `0.09947` | `0.8676` | `0.9112` |
| 64 | `0.55232` | `0.2911` | `0.8918` |

One and four states pass their frozen thresholds because the affine readout has
enough degrees of freedom.  At 16 states the design is overdetermined and full
column rank; at 64 states all 225 directions remain active, but frozen random
features cannot represent the distinct targets.  Static one-shot readout
calibration is therefore rejected as a complete repair.

Together with the failed Adam memorization and exact one-state readout, this
supports a bounded variable-projection test: periodically solve the small
affine head exactly, then update only the nonlinear backbone at the reduced
least-squares objective.  The envelope theorem means the optimal-head
derivative need not be backpropagated through the solve.  This adds training
work but no inference parameters or runtime branch.

Exact values are in `result.json`; no tensor or later Gate is authorized.
