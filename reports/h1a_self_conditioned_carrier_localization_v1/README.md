# H1a self-conditioned carrier localization v1

Status: **completed; scalar topology carrier rejected**.

At the fixed `t=0.6` state, every approximate topology field was given its own
closed-form optimal ridge carrier on the 512-structure train panel and then
evaluated once on the disjoint 256-structure panel. This removes the shared
clean-oracle coefficient as a possible explanation of the earlier negative
result. Two concatenated noisy-plus-estimated fields also test whether a
linear dual-state conversion is sufficient.

Only the clean oracle passes (`+0.14203`). Tweedie-specific, probe-specific,
and noisy-specific readouts give `-0.00012`, `-0.00045`, and `-0.00041` held-out
improvement. Noisy-plus-Tweedie and noisy-plus-probe give `-0.00009` and
`-0.00134`. Every non-oracle bootstrap interval includes or lies below zero.

Therefore the failure is not caused by reusing the clean-oracle coefficient.
The scalar topology-weighted radial-direction family does not contain a
held-out correction obtainable from the current approximate fields. The next
bounded test must distinguish a nonlinear state-derived vector conversion from
irreducible probability-path conditional variance; calibrating or widening the
same linear carrier is not a correct repair.
