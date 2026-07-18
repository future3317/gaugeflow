# H1a primal/dual metric attribution v1

Status: **completed; primal/dual metric mismatch confirmed.**

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

## Result

Every frozen condition passes at `t=0.005`. The median relative metric
condition is `15.94997` (maximum `544.50970`), and the flattened dual/primal
output-gradient cosine is only `0.31570`. The dual loss has log correlation
`0.26532` with actual periodic endpoint squared error, whereas the primal form
has correlation `0.999996`; the gap is `0.73468`. The same pattern persists at
all five audited times, with gradient cosine around `0.30` and primal endpoint
correlation above `0.9979`.

The attribution is therefore `primal_dual_metric_mismatch`. More training of
the Cartesian-covector objective is not authorized. The implementation must
next distinguish the score covector from the mobility-applied reverse drift:
the sampler adds its network output to fractional coordinates and therefore
consumes a tangent vector. A Cartesian tangent vector transforms to fractional
components with `L^-1`, not the covector pullback `L^T`. A separately frozen
no-training index-raising correction may test this exact type repair; it may
not change the probability path, carrier, optimizer, steps, seeds, or sampler
equations at the same time.
