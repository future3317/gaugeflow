# H1a coordinate tangent audit v1

Status: **completed but numerically inconclusive for strict rank; it does show
severe tangent ill-conditioning. H1a remains failed.**

On the fixed 11-site, `t=0.005` state, the FP32 Jacobian Gram spectrum reported
28 active directions against a 30-dimensional translation quotient, target
projection residual `0.17442`, condition number `7.76e6`, and effective rank
only `2.27`.  The coordinate-edge-head gradient norm was `0.19391`, while the
coordinate-vector-head and control-gate norms were only `0.001283` and
`0.000562`.  Thus the output tangent is strongly dominated by the direct
central edge route and the transverse/deep vector route is weak.

The v1 implementation accumulated `J J^T` in FP32 before converting it to
FP64 for eigendecomposition.  Its spectrum contains impossible negative values
down to `-1.21e-6`; the frozen active threshold was `5.45e-6`.  Therefore the
reported two additional null directions may be roundoff-amplified and cannot
justify a strict architectural rank-deficiency claim.  Changing the vector
head directly from this result would be scientifically premature.

The corrected v2 audit keeps the same FP32 forward/Jacobian and fixed state but
accumulates the small output Gram matrix in FP64.  It is a new preregistered
numeric qualification, not a reinterpretation of the v1 thresholds.  No model
or optimizer change is authorized by v1.
