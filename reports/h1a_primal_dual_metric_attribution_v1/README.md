# H1a primal/dual metric attribution v1

Status: **frozen before run; read-only.**

For row fractional coordinates `r=fL`, define the fractional displacement
metric `G=L L^T`. The failed training loss measures a score-covector residual
`e` as

```text
ell_dual = e G^-1 e^T,
```

while the physical endpoint displacement induced by the same fractional error
is measured locally as

```text
ell_primal = e G e^T.
```

These metrics agree up to a scalar only for isotropic cells. This audit uses the
fixed validation panel and checkpoint without gradients through the model or
optimizer steps. It reports the exact dual and primal quadratic forms, their
output-space gradient cosine, lattice metric condition, and log correlation of
each form with the actual periodic, translation-aligned endpoint error.

The metric-mismatch attribution requires all pre-registered conditions at
`t=0.005`: median relative metric condition at least `4`, dual/primal gradient
cosine at most `0.8`, primal-to-endpoint log correlation at least `0.8`, and a
primal-minus-dual correlation gap at least `0.2`. Failure of any condition sends
the diagnosis back to the learned carrier rather than authorizing a path change.
