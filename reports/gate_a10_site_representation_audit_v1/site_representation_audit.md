# Gate A10 species-aware site-representation audit

This read-only audit uses a periodic, species-decorated StructureMatcher. It distinguishes a harmless CIF row permutation from a real chemical-sublattice mismatch.

| Source | samples | CIF-index atom accuracy | species-aware match rate |
|---|---:|---:|---:|
| gate_a7_joint_type_set_v1 | 16 | 0.406 | 0.438 |
| gate_a8_time_conditioned_type_set_v1 | 16 | 0.594 | 0.562 |
| gate_a9_source_weighted_type_set_v1 | 16 | 0.578 | 0.188 |

The non-unit species-aware rates prove that the residual errors are not merely arbitrary CIF ordering. The code audit also finds that endpoint-ID emits no response edge field and the scalar type messages contain no periodic edge length or vector-state invariant. Therefore a future architecture repair must make scalar site decoding geometrically informative and introduce symmetry-breaking node latents without using target atom order. No further sampler/loss search is justified by this audit.
