# Stage-E Conditional Interface Closure v1

## Decision

The old Stage-E lattice catastrophe is best classified as **multiple interacting causes**: the old VP posterior was wrong, but correct VP alone does not remove the generated-lattice failure mode. E3 tensor conditioning strongly amplifies the oracle_ca generated-lattice path without the exposure adapter; the JARVIS generated-side lattice adapter suppresses the catastrophic outliers, but it does not qualify Stage-E because it degrades oracle_c/free lattice and local geometry metrics.

ABC does **not** need retraining. GaugeFlow-base A1 remains qualified under correct VP, and the Stage-C Pareto-minimax selection remains the 30k relative / 40523 global checkpoint.

## A/B/C factorial

| System | Arm | tensor RMSE | volume W1 | NN W1 | finite lattice | failures |
|---|---:|---:|---:|---:|---:|---:|
| A/B no adapter, conditioned role | oracle_cal | 1.0477 | 0 | 0.5476 | 1.000 | 0 |
| A/B no adapter, conditioned role | oracle_ca | 49.8159 | 6.137e+08 | 0.6534 | 1.000 | 0 |
| A/B no adapter, conditioned role | oracle_c | 1.2026 | 0.06235 | 0.5778 | 1.000 | 0 |
| A/B no adapter, conditioned role | free | 1.2985 | 0.3468 | 0.2346 | 1.000 | 0 |
| C conditioned adapter, conditioned role | oracle_cal | 1.0477 | 0 | 0.5476 | 1.000 | 0 |
| C conditioned adapter, conditioned role | oracle_ca | 1.4261 | 427 | 0.5725 | 1.000 | 0 |
| C conditioned adapter, conditioned role | oracle_c | 1.2138 | 0.2037 | 0.6170 | 1.000 | 0 |
| C conditioned adapter, conditioned role | free | 1.2815 | 0.483 | 0.1688 | 1.000 | 0 |

Key paired comparisons:

- B - A: E3 tensor conditioning is neutral on oracle_cal/oracle_c but catastrophically unstable on oracle_ca without the lattice adapter: tensor RMSE 49.8159 and volume W1 6.137e8.
- C - B: the JARVIS generated-side lattice adapter repairs most oracle_ca damage: tensor RMSE 49.8159 -> 1.4261 and volume W1 6.137e8 -> 426.98.
- C - A: the final conditioned system is still not qualified. In oracle_c/free it worsens volume-W1 relative to base, and in oracle_ca the volume tail remains far above normal generated-lattice scale.

## Lattice trajectory

| Trace | Arm/role | logV median | logV p99 | logV max | physical V median | physical V p99 | physical V max | cond p99 | cond max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no adapter | oracle_ca/base | 5.712 | 27.08 | 34.7 | 302.4 | 8.345e+11 | 1.169e+15 | 26.57 | 29.28 |
| no adapter | oracle_ca/conditioned | 5.709 | 27.43 | 35.1 | 301.5 | 1.25e+12 | 1.759e+15 | 29.27 | 32.96 |
| no adapter | oracle_c/base | 5.27 | 6.494 | 6.833 | 194.5 | 661.2 | 928.3 | 10.11 | 16.16 |
| no adapter | oracle_c/conditioned | 5.273 | 6.481 | 6.907 | 195 | 652.7 | 999 | 12.24 | 18.75 |
| no adapter | free/base | 5.534 | 6.703 | 6.945 | 253.1 | 815.4 | 1038 | 3.94 | 5.401 |
| no adapter | free/conditioned | 5.533 | 6.702 | 6.942 | 252.8 | 814.6 | 1035 | 4.197 | 6.335 |
| conditioned adapter | oracle_ca/base | 5.712 | 27.08 | 34.7 | 302.4 | 8.345e+11 | 1.169e+15 | 26.57 | 29.28 |
| conditioned adapter | oracle_ca/conditioned | 5.321 | 10.99 | 19.18 | 204.6 | 6.205e+04 | 2.134e+08 | 10.82 | 12.71 |
| conditioned adapter | oracle_c/base | 5.27 | 6.494 | 6.833 | 194.5 | 661.2 | 928.3 | 10.11 | 16.16 |
| conditioned adapter | oracle_c/conditioned | 5.222 | 6.789 | 6.866 | 185.3 | 888.1 | 959.2 | 19.99 | 30.54 |
| conditioned adapter | free/base | 5.534 | 6.703 | 6.945 | 253.1 | 815.4 | 1038 | 3.94 | 5.401 |
| conditioned adapter | free/conditioned | 5.555 | 7.108 | 7.469 | 258.6 | 1222 | 1753 | 4.083 | 5.825 |

The no-adapter oracle_ca tail remains explosive under correct VP: physical volume p99 around 1e12 and max around 1e15. The adapter reduces the conditioned oracle_ca max to 2.134e8, but this is still a tail pathology rather than a normal lattice distribution. oracle_c/free trajectories are numerically stable without adapter; adapter shifts their volume/shape tails upward.

## Stage-C correct-VP requalification

| global step | relative step | NN W1 | volume W1 | exact comp | finite lattice | failures |
|---:|---:|---:|---:|---:|---:|---:|
| 30523 | 20000 | 0.56282 | 0.06763 | 1.000 | 1.000 | 0 |
| 35523 | 25000 | 0.55812 | 0.06784 | 1.000 | 1.000 | 0 |
| 40523 | 30000 | 0.56561 | 0.06802 | 1.000 | 1.000 | 0 |
| 45523 | 35000 | 0.57058 | 0.06824 | 1.000 | 1.000 | 0 |
| 50523 | 40000 | 0.57846 | 0.07112 | 1.000 | 1.000 | 0 |
| 55523 | 45000 | 0.57870 | 0.06886 | 1.000 | 1.000 | 0 |
| 60523 | 50000 | 0.57234 | 0.06755 | 1.000 | 1.000 | 0 |

For the original four-candidate selection panel (20k/30k/40k/50k relative), the generation metrics retain the declared Pareto-minimax choice at 30k relative / 40523 global. Intermediate 25k/35k/45k diagnostics are recorded but not part of the frozen selection candidate set.

## Data contract

The lattice contract audit found consistent finite positive lattices, Angstrom^3 determinant volume, volume-per-atom definition det(L)/N, and the P1 trace-free shape chart across the checked Stage-D/JARVIS and Alex paths. `source_index` is an upstream table identifier and is not used as a leakage key; split/cache row membership is the correct audit boundary.

## Boundary

Stage-E remains blocked. Do not start F, tensor-conditioned material claims, relaxation, DFT, or DFPT from this evidence. The next implementation step should be a constrained fix of the generated-lattice handoff/exposure path, evaluated against the same A/B/C factorial protocol.
