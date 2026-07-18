# H1a coordinate affine-readout audit v1

Status: **formally failed two strict numerical checks, while demonstrating a
complete but catastrophically scaled final coordinate basis. H1a remains
failed.**

The fixed upstream representation was held constant.  Only the 225 parameters
of the final vector and edge linear readouts were included, making the output
algebraically affine in the audited parameters.  Their Jacobian has all 30
translation-quotient directions active.  The edge readout alone also has rank
30; the failure is not a missing Cartesian direction.

The FP64 minimum-norm solve predicts MSE `4.68e-10`; the actual FP32 forward
after applying it reaches `4.76e-8`, far below the frozen `1e-5` MSE threshold.
However, target projection residual `2.50e-5` exceeds `1e-5`, and affine-forward
relative error `2.54e-4` exceeds `1e-4`.  Both strict checks remain failed.
They are evaluated in a 33-dimensional representation with three near-zero
translation modes and a very large FP32 parameter update; a successor must
remove those modes analytically rather than relaxing this result.

The scale diagnosis is unambiguous: the initial final-readout parameter norm is
`0.80036`, whereas the minimum-norm fitting update is `2079.13` (about `2598x`).
The affine tangent condition number is `3.50e7` and effective rank is `2.23`.
Ordinary Adam updates therefore cannot move enough along the weak coordinate
basis before upstream nonlinear features drift.

The next no-training audit uses an explicit Helmert basis for the exact
translation quotient.  It changes no model, loss, optimizer, path or Gate.
Exact values are in `result.json`.
