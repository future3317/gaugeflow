# H1a coordinate NTK trust-step audit v1

Status: **completed; the full damped Gauss--Newton step is nonlocal and the
pre-registered scale panel does not resolve the local trust radius. H1a remains
failed.**

On the same fixed 11-site, `t=0.005` state as the corrected tangent audit, the
FP32 Jacobian/FP64 output Gram predicts that the damped full step would reduce
coordinate MSE from `0.745457` to `0.0004946` (`99.9337%`).  Thus the linear
output tangent can fit this target.

That formal step has norm `524.856`, or `3.1576` times the norm of all active
parameters.  Even the smallest pre-registered scale, `1/32`, changes the
parameters by about `9.87%` of their norm.  It is therefore not a genuinely
local derivative check: its real forward MSE is `180.618`, and larger scales
diverge further while remaining finite.  The result rejects an undamped or
naively scaled full output-pseudoinverse update, but it cannot distinguish an
autograd/numeric inconsistency from ordinary nonlinear curvature because the
panel never entered a small trust region.

The only justified successor is a no-training radius audit whose steps are
defined directly as fixed fractions of the parameter norm.  It may locate the
linearization radius; it may not change the model, optimizer, probability path,
historical H1a result or later-Gate boundary.

Exact values and the frozen acceptance checks are in `result.json`.
