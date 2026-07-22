# Stage-D equivalent-view D0 audit

This zero-training audit adapts FlowMimic-style online pairing to exact crystal
object symmetries. It uses the selected Stage-C 30k EMA, 32 real Stage-D train
graphs (380 atoms), identical random response heads, and the same normalized
full-tensor plus icosahedral response-probe objective in every arm.

| Transform set | Loss rel. residual | Gradient cosine | Gamma rel. residual | Decision |
| --- | ---: | ---: | ---: | --- |
| proper rotation + origin + permutation | `4.74e-7` | `1.0000000` | `3.35e-7` | use |
| elementary `SL(3,Z)` basis change only | `3.28e-5` | `0.997784` | `5.14e-2` | exclude |
| improper rotation + handedness compensation | `2.01e-6` | `0.999990` | `2.53e-2` | exclude |
| all transforms together | `2.31e-5` | `0.999403` | `8.04e-2` | exclude |

Every arm preserves target Frobenius norm and lattice volume to about `1e-6`
or better. The failed arms expose an input-chart issue rather than an invalid
physical transformation: the current backbone's lattice-shape context is not
strictly invariant to a non-canonical integer basis representation. Improper
Cartesian actions require an orientation-reversing basis compensation to keep
the row lattice right-handed and inherit the same issue.

Stage-D training therefore uses one online view sampled from proper `SO(3)`,
fractional origin shifts, and within-graph atom permutations. It does not use
`GL(3,Z)` or improper augmentation, and it does not add a paired second forward
pass. Full polar parity remains covered by the Cartesian head unit tests. This
audit qualifies the augmentation and invariant objective implementation only;
it is not predictive response evidence and does not authorize Stage E or F.
