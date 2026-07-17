# Qualified H1a P1 scaled-score substrate

**Decision: qualified for a separately frozen full tensor-free H1a
benchmark.** This result does not authorize H1b, tensor conditioning, an
oracle, relaxation, DFT, or DFPT.

The only changes relative to the preceding failed wrapped-score experiment are
an algebraically equivalent scaled-score output and the analytically qualified
uniform-variance reverse grid. The network predicts
`sigma(t) * grad log p_t`; the sampler divides by `sigma(t)`. Thus its
unweighted MSE is exactly the former variance-weighted raw-score MSE, but the
low-noise output and parameter gradient remain order one.

| Seed | total validation ratio | coordinate validation ratio | fraction with minimum distance >= 0.5 A | minimum distance (A) | failures |
|---:|---:|---:|---:|---:|---:|
| 5201 | 0.63808 | 0.34058 | 0.984375 | 0.27681 | 0 |
| 5202 | 0.65400 | 0.41222 | 0.984375 | 0.43898 | 0 |
| 5203 | 0.63683 | 0.30611 | 0.984375 | 0.26842 | 0 |

The mean total and coordinate ratios are `0.64297` and `0.35297`. Every
training/validation quantity is finite, all 192 reverse trajectories finish,
all categorical masks are resolved, every lattice is finite and positive
volume, and the tensor-free path constructs zero Cartesian-atlas candidates.
The aggregate and every-seed close-contact guardrail pass their frozen
`0.95`/`0.90` bounds.

The three remaining sub-0.5 A samples are retained and reported; this is a
qualification of the bounded learning substrate, not a claim of final crystal
quality. The next H1a action is a larger fixed-sample comparison to held-out
Alex structures for composition, lattice, nearest-neighbor, validity and
distribution coverage. Only that benchmark can decide whether H1a as a whole
is ready to freeze before H1b.
