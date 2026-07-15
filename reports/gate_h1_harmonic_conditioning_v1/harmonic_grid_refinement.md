# H1 harmonic-grid refinement operator audit

This is a deterministic numerical operator check, not training or a conditional-generation gate. The largest declared grid is a reference, not exact SO(3) integration.

- device: `cuda`
- seed: `20260715`
- reference grid: `240`

- CUDA forward/backward smoke: `{'direct_irrep_complete_v1': True, 'harmonic_alignment_v1': True}`

| grid | mean posterior entropy | relative aligned-irrep difference to reference |
|---:|---:|---:|
| 24 | 3.12603283 | 2.01429749e+00 |
| 60 | 4.04510975 | 5.95571697e-01 |
| 120 | 4.74015522 | 1.72405109e-01 |
| 240 | 5.43258476 | 0.00000000e+00 |

No threshold is declared here. A later causal training protocol must pre-register its grid and an acceptance tolerance before inspecting generation results.
