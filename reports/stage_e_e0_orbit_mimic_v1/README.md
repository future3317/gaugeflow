# Stage-E E0 common-noise orbit-mimic screen

This one-seed mechanism screen uses the same Stage-C 30k checkpoint, response
cache, noisy states, clocks, and validation panel in every arm. It does not
qualify free tensor-conditioned generation or Stage F.

| arm | fine loss | typed orbit residual | posterior information | target-swap separation | null typed drift |
|---|---:|---:|---:|---:|---:|
| baseline | 2.067480 | 4.379e-4 | 2.598e-5 | 0.103512 | 0.181775 |
| common-noise orbit mimic | **1.818112** | **1.997e-4** | **2.373e-4** | 0.238860 | 0.203300 |
| orbit mimic + soft retention | 2.193481 | 4.776e-4 | 6.691e-6 | **0.256773** | 0.048221 |
| atlas-only exact-null | 2.657143 | 3.559e-3 | 1.408e-5 | 0.147015 | 3.773e-10 |
| low-rank exact-null | 2.806377 | 2.825e-4 | 1.173e-5 | 0.137671 | 4.473e-10 |
| centered-block exact-null | 2.471778 | 1.468e-3 | 1.871e-4 | 0.206684 | 4.473e-10 |

Common-noise orbit mimic is the only arm that improves endpoint denoising,
reduces representative residual by more than 50%, increases atlas information,
and strengthens target-swap separation. Soft retention and three structurally
exact-null repairs trade away the conditional mechanism; none is selected.

The retained deployment contract is therefore two explicit model roles:

- the frozen Stage-C checkpoint is the only null/unconditional generator;
- the common-noise orbit-mimic checkpoint is condition-required and may never
  serve as a null fallback.

This routing makes unconditional retention exact without constraining the
non-null tensor branch toward the unconditional field. The next experiment is a
paired conditioned rollout using the selected E0 checkpoint and the frozen
Stage-D model as an independent evaluator. F remains blocked until that direct
condition Gate passes.
