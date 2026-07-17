# H0-E-v2 E1a maximal-t parent occurrence

## Decision

`H0-E-v2-E1a_failed_stop_before_E1b_and_H1a`.

The frozen 64-row panel contained only H0-E-v1 rows with zero parent
candidates. The setting-exact maximal translationengleiche search discovered
`0/64` new candidate materials, below the preregistered minimum `3/64`.
H0-E-v1 remains unchanged. E1b, H1a/H1b and H2--H6 are not authorized.

## Frozen results

| Metric | Result | Requirement | Pass |
|---|---:|---:|:---:|
| Selected rows | 64 | 64 | yes |
| Alex source join | 1.0 | 1.0 | yes |
| v1 zero-candidate fraction | 1.0 | 1.0 | yes |
| Processing failures / nonfinite values | 0 / 0 | 0 / 0 | yes |
| New candidate materials | 0 | >= 3 | **no** |
| New candidate-material fraction | 0.0 | >= 0.046875 | **no** |
| Forward/reverse embedding-set agreement | 1.0 | 1.0 | yes |
| Total / p95 per-row runtime | 34.62 s / 1.802 s | diagnostic | -- |

Candidate-conditional parent-reidentification and StructureMatcher fractions
are fail-closed at zero because no occurrence survived. The independent audit
rebuilt all 64 rows in reverse E0 order, reproduced zero candidates and passed
every artifact, selection, numerical and decision check.

## Failure localization

Seventeen materials have no incoming maximal-t E0 edge. The other 47 materials
evaluate 430 setting-exact edges:

- 236 fail the triangle-safe 0.4 Angstrom orbit-defect prefilter. Their minimum,
  median, p95 and maximum defects are 0.6476, 3.0807, 5.0622 and 6.5109
  Angstrom, respectively; these are not near-threshold rejections.
- 191 cannot define a species-preserving nearest assignment satisfying the
  complete parent permutation group law.
- three embeddings reach a strict SG 71 projection from the same SG 12 material
  (`alex<agm004639609>`) but fail StructureMatcher. They have one-sided source
  displacement 0.09859 Angstrom and Hencky norm 0.48977. Even if matcher
  certification were repaired, this is only one candidate material and the
  strain exceeds the frozen 0.15 bound.

The negative result therefore does not support a hidden threshold adjustment.
One-step maximal-t occurrence does not address the v1 coverage deficit on this
panel. A future successor would need a separately proposed H0-E-v3 mechanism
for multi-step and/or cell-changing paths; the failed E1a protocol does not
authorize silently continuing to its planned E1b.

## Provenance

- Protocol freeze commit: `2110f8961de7c00c5fdd9bae6d9163c0b7499778`
- WSL-safe provenance fix and run commit:
  `9f4ca8b71cf9008d5c26c42c62fff00310fd1cab`
- Results SHA-256:
  `7d693919eb0acea9a4231b3a9a08211f8979c2ac09652178515ebe3608b73cb0`
- Manifest SHA-256:
  `30cf68e46723d3020cf33674042f315f192943c985c3431ee1de024eda1df405`
- Independent audit SHA-256:
  `4b52e962c1c35e405c8f7bdb2ab8c256c6200291ca0d0131e82a1b9dd7bcb00d`
- External artifact root:
  `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v5/maximal_t_parent_occurrence_e1a_v1/`

