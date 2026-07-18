# H1a function-preserving coordinate-readout scale v1

Status: **qualified in its no-training scope. H1a remains failed.**

The active vector and edge final-readout weights are initialized at exactly
`1/1024` of the prior values, and their combined Cartesian output is multiplied
by `1024`.  Because 1024 is a power of two, the initialized function is
preserved to FP32 rounding: maximum output difference from the algebraic
baseline is `1.79e-7`.

This reparameterization directly repairs the measured parameter-distance
problem.  The exact quotient target still has rank 30 and projection residual
`7.47e-16`; its minimum-norm readout update falls from `2079.20` to `2.0302`,
while actual affine-fit MSE is `4.27e-8`.  The tangent condition number remains
`3.50e7` by design—the basis is not rotated—but the optimizer no longer needs
to traverse thousands of initial readout norms to use its weak directions.

There is no material runtime tax: RTX 4060 Ti coordinate training reaches
`452.75 graphs/s` at `1636.19 MiB`.  Translation error is `1.88e-7`, parameter
count is unchanged, and a persistent scale buffer makes incompatible failed
historical checkpoints fail strict loading rather than silently changing their
function.

Qualification authorizes only the same-seed, same-state, 1,024-step
memorization test.  It does not qualify H1a or authorize a larger panel,
resampled states, H1b, tensor conditioning or later Gates.

Exact values are in `result.json`.
