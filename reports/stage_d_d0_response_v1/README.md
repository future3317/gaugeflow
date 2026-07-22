# Stage-D D0 response-field auxiliary screen

D0 compares the complete Cartesian response objective against the same
objective plus a weight-0.1 six-axis icosahedral response-field auxiliary. Both
arms start from the selected Stage-C 30k EMA and share seed 5731, sample order,
proper-rotation/origin/permutation augmentation streams, BF16 execution, and
2,000 updates.

| Validation quantity | Baseline | + probe | Relative change |
|---|---:|---:|---:|
| response-field probe loss | `0.35080761` | `0.35095587` | `+0.0423%` |
| complete piezoelectric tensor loss | `0.29784991` | `0.29819625` | `+0.1163%` |
| other active-task macro | `0.49522593` | `0.49515542` | `-0.0142%` |

The frozen primary requirement was at least 5% response-field improvement. The
probe arm instead changes the metric by only `+0.0423%` in the wrong direction;
retention checks pass, but the mechanism check fails. Stage-D therefore keeps
the simpler complete-tensor baseline and does not tune the auxiliary weight.

This D0 run used the initial cache in which the elastic head was fully masked.
The result remains a valid test of whether the auxiliary adds information to
the fully supervised piezoelectric objective, but it is not a Stage-D predictive
qualification. Before formal D training, the cache was rebuilt with 2,893
audited JARVIS elasticity targets. All non-elastic tensors remained bitwise
unchanged, and a new three-step baseline CUDA smoke passed with finite elastic
loss `0.71929`, total validation loss `0.85943`, and `3.35 GiB` peak allocated
memory.

![Stage-D D0 paired screen](stage_d_d0_response.png)
