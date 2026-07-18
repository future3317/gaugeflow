# H1a Cartesian-tangent readout-span audit v1

Status: **qualified diagnostic; classified as `backbone_span_limited`. H1a
remains failed.**

This audit reads the preregistered seed-5705 checkpoints at steps
`0/1250/5000/8441` without gradients or optimizer updates. On fixed,
disjoint 128-graph train and validation panels, five times and two noise
replicates, it captures the 80-channel centered Cartesian carrier and solves a
single graph-equal float64 minimum-norm readout. A train-fitted head measures
generalizable head optimization; a validation-fitted head is an offline span
ceiling only and is never a runtime parameter or checkpoint.

All diagnostic qualifications pass: carrier/head reconstruction error is
`9.54e-7`, every train and validation design is rank `80/80`, all metrics are
finite, tensor candidates are zero, parameters are bitwise unchanged, and
optimizer steps are zero.

## Result

| step | current train explained | current val explained | train-opt/train current | train-opt/val current | val-oracle explained |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.00005 | 0.00033 | 0.91767 | 1.04709 | 0.04673 |
| 1,250 | 0.14099 | 0.07847 | 0.89793 | 1.02743 | 0.14322 |
| 5,000 | 0.48258 | 0.37448 | 0.93896 | 1.01244 | 0.43010 |
| 8,441 | 0.57282 | 0.45469 | 0.94769 | 1.03459 | 0.49608 |

The current head is already close to the best global train readout: refitting
it reduces train loss by only `5.23%` and *increases* validation loss by
`3.46%`. Thus the failed checkpoint is not waiting for a better global linear
head. Even a validation-label oracle head explains only `49.61%` of target
energy, below the preregistered `75%` span threshold. This remains
`46.96--53.02%` at every audited time/replicate. The oracle span does improve
by `44.94` percentage points from initialization, so feature learning is
active, but insufficient after the frozen pass.

The design remains algebraically full rank while validation effective rank
changes from `30.32` at initialization to `21.32` at the final checkpoint.
The conclusion is therefore not a missing Cartesian direction or head
disconnect; it is a learned cross-state carrier-span limitation. The only
admissible successor is one separately frozen feature-formation mechanism. It
may use scalar state to mix the existing Cartesian carriers, but may not add a
harmonic branch, extra seed/step, joint initialization, tensor condition, or
later-Gate work.
