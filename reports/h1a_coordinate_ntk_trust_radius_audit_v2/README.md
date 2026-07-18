# H1a coordinate NTK trust-radius audit v2

Status: **completed; the strict smallest-radius check failed and no useful
nonlinear step was found. H1a remains failed.**

This no-training successor parameterized every trial by its actual update norm
relative to the active parameter norm.  The damped linear tangent still
predicts `99.9337%` loss removal, but the full step is `3.1575` parameter norms
long.

At the smallest radius (`1e-4`), actual and linear MSE are respectively
`0.745410310` and `0.745410366`, versus `0.745457283` initially.  The loss-level
agreement is excellent, but the pre-registered output-change-relative error is
`0.33845 > 0.25`; the formal local-consistency check therefore fails rather
than being relaxed after seeing the result.  This ratio is sensitive to FP32
quantization because the intended output change is tiny.

The best preregistered nonlinear result occurs at radius `0.003` and removes
only `0.1388%` of the loss, below the frozen `0.5%` useful-step threshold.  At
radius `0.01` the real loss is already worse although the linear model still
predicts improvement.  Thus the weak tangent directions require parameter
motion beyond the useful curvature radius.  A direct full pseudoinverse or
generic trust-step optimizer is not supported.

The next bounded question is whether the coordinate output is sufficiently
spanned by its affine final readout while all upstream features are fixed.  An
exact readout solve can separate an inadequate coordinate feature basis from
optimization-induced feature drift without changing or training production.

Exact values are in `result.json`.  No result authorizes H1b, tensor
conditioning or a later Gate.
