# H1a assignment orbital-set expressivity audit v1

Decision: `second-order orbital sets remain insufficient; do not train and audit the smallest higher-order or assignment-level latent statistic`.

| metric | value |
|---|---:|
| exact enumeration coverage | 0.995595 |
| exact unary-collision carriers | 331 |
| orbital-set resolved fraction | 0.042296 |
| orbital-set mean target ceiling | 0.364685 |
| relabel failures | 0 |

The representation keeps exact pair orbitals as an unordered set and
uses no orbit index. This audit is about identifiability only; pairwise
global normalization remains a separate blocker before training.

Boundary: This audit cannot qualify assignment, its normalization, generated-C, p(N), L1/M1, tensor work, relaxation, DFT or DFPT.
