# Geometry-complete assignment carrier v2

Decision: **PASS**.

This Gate recompiles the unchanged qualified O1 occurrences into a canonical
row-HNF carrier with one species-free coordinate per finite-action node.  The
terminal coloring is stored in a separate target object and never participates
in HNF conversion, periodic node alignment, or action conjugation.

| metric | observed | frozen requirement |
|---|---:|---:|
| source candidates | 454 | 454 |
| candidate rebuild fraction | 1.0 | 1 |
| archived identity fraction | 1.0 | 1 |
| HNF index closure | 1.0 | 1 |
| expanded node closure | 1.0 | 1 |
| action-node alignment | 1.0 | 1 |
| target reconstruction | 1.0 | 1 |
| source relabel consistency | 1.0 | 1 |
| maximum periodic alignment error (A) | 4.6117381930282355e-14 | <=1e-6 |
| processing failures | 0 | 0 |

All checks: `True`.  Archived O1 and failed Q1 artifacts were
read-only dependencies and were not overwritten.  Passing this Gate permits
only a geometry-aware zero-training assignment expressivity audit.
