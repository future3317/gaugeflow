# H1a P1 tensor-free learning curve

**Frozen decision: failed at H1a; diagnose the coordinate generator.** This
result does not authorize H1b, tensor conditioning, an oracle, relaxation,
DFT, or DFPT.

The experiment used the qualified 675,204-structure Alex cache, three fixed
seeds, 5,000 updates per seed, the frozen 192/32/four-block model and no tensor
condition. The protocol and every checkpoint are bound to protocol SHA-256
`8c170213c24480545471a61b8628d50e1f5ea229d95aee77f648698406125b17`.

| Seed | final / initial validation | samples with minimum distance >= 0.5 A | minimum distance (A) | failures | masks |
|---:|---:|---:|---:|---:|---:|
| 5201 | 0.68196 | 0.93750 | 0.25460 | 0 | 0 |
| 5202 | 0.69105 | 0.84375 | 0.08088 | 0 | 0 |
| 5203 | 0.68027 | 0.95312 | 0.21688 | 0 | 0 |

The mean validation ratio, `0.68443`, passes the frozen `0.75` bound and all
three seeds pass the per-seed `0.85` bound. All lattices are finite,
right-handed, and positive-volume; the categorical process terminates without
masks; the tensor-free bypass creates no atlas candidates; and there are no
sampling failures.

The protocol nevertheless fails both pre-registered local-geometry checks.
The mean acceptable-distance fraction is `0.91146 < 0.95`, and seed 5202 has
`0.84375 < 0.90`. Coordinate validation loss also plateaus near `0.876` after
step 1,000 while element and lattice heads continue improving. The supported
conclusion is therefore that the joint substrate learns composition and
lattice statistics but has not qualified its local coordinate score/reverse
trajectory. The thresholds remain unchanged; the next work is a read-only
time-resolved score, endpoint-estimator, generated-neighbor, and sampler-step
audit at H1a.
