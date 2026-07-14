# Gate A5 quotient substrate repair

A5 is a new, fixed-budget endpoint-ID substrate test. It does not alter the negative A4 conclusion or constitute a tensor-conditioned result.

## A5.0 path/coupling invariants

All typewise OT costs were no worse than identity: `True`.
Maximum simplex unit-sum error: `1.192e-07`; maximum tangent-sum error: `4.292e-06`.
Maximum no-drift graph-mean residual: `1.490e-08`.

## Fixed endpoint-ID result

| Variant | type composition | geometry retrieval | joint retrieval | between/within | failures | qualifies |
|---|---:|---:|---:|---:|---:|---:|
| type_riemannian_simplex_endpoint_nll | 0.312 | 1.000 | 1.000 | 3.771 | 0 | False |
| geometry_typewise_ot_no_drift_normalized | 1.000 | 0.562 | 0.812 | 1.132 | 0 | False |
| joint_ot_simplex_no_drift_normalized | 0.062 | 0.688 | 0.812 | 1.163 | 0 | False |

Only an all-criteria pass would permit a separately versioned tensor-conditioned gate. A5 itself never passes Gate A.
