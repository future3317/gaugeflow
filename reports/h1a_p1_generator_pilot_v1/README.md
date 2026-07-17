# H1a P1 generator pilot

**Decision: qualified for a separately frozen longer tensor-free H1a learning
curve. This is not full H1a qualification.**

The pilot used the qualified 675,204-row Alex cache, three fixed seeds, 1,000
steps per seed, batch size 16, the same 192/32/four-block model, and no tensor
condition. Learned layers used BF16 on the RTX 4060 Ti while probability-path,
lattice and periodic-neighbor kernels remained FP32.

| Seed | Initial validation | Step-1000 validation | Ratio | Sampling failures | Terminal masks |
|---:|---:|---:|---:|---:|---:|
| 5201 | 7.9490 | 6.7016 | 0.8431 | 0/32 | 0 |
| 5202 | 7.9843 | 6.7579 | 0.8464 | 0/32 | 0 |
| 5203 | 8.0678 | 6.7871 | 0.8413 | 0/32 | 0 |

The mean final/initial validation ratio is `0.84358`, below the frozen `0.90`
threshold, and every seed is below `1.0`. All logged losses and gradients are
finite. All 96 generated lattices are finite, right handed and positive-volume;
the tensor-free bypass produced exactly zero Cartesian-atlas candidates.

This short run does not establish final structural quality. The fraction with
minimum periodic distance at least 0.5 A is `0.9375`, `0.90625` and `0.9375`
for the three seeds. The corresponding worst distances are 0.398, 0.135 and
0.274 A. These close contacts are expected to remain a principal guardrail for
the longer learning curve; they were not silently added to the already frozen
pilot acceptance criteria.

No H1b, tensor condition, oracle, relaxation, DFT or DFPT work is authorized by
this result.
