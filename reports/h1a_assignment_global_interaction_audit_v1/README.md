# H1a assignment global-interaction audit v1

This is a zero-training expressivity audit of the frozen failed Q1 unary
assignment family. It does not qualify assignment or authorize a successor
training run by itself.

Decision: `pair order is sufficient but the transferable descriptor is not; refine only the target-free pair descriptor and do not train`.

| metric | value |
|---|---:|
| exact enumeration coverage | 0.995595 |
| exact unary-collision carriers | 331 |
| exact pair-orbital resolved fraction | 0.936556 |
| transferable pair resolved fraction | 0.039275 |
| transferable pair mean target ceiling | 0.357654 |
| relabel failures | 0 |
| target-orbit containment failures | 0 |

The exact pair-orbital statistic is explicitly an upper bound. It may use
the carrier's complete pair-orbit partition but never a prototype ID. The
transferable statistic merges pair orbits whenever their target-free action
descriptors coincide; only that family is eligible to motivate a shared scorer.

Boundary: This audit cannot qualify assignment, generated-C, p(N), L1/M1, tensor work, relaxation, DFT or DFPT.
