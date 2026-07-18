# H1a Krylov production gradient attribution v1

Status: **completed; fractional covector chart is the dominant amplification.**

The standalone 80-channel Cartesian carrier passed, while its first production
integration failed only the frozen absolute-gradient guardrail. This protocol
loads the exact candidate source from Git commit `e25f432` in a temporary
read-only worktree. The active production tree remains the preceding combined
head.

On the same 16 states and initialization, the audit decomposes gradient norm at
three stages:

```text
carrier probe -> Cartesian score -> fractional score covector.
```

It also recomputes the exact carrier with graphwise RMS denominators detached
only for derivative attribution, while preserving identical forward values;
splits the final field into `v`, `m`, `Qm`, and `Q^2m`; and reports squared
gradient mass for the final head, moment projection, edge encoder, control gate,
message blocks, and shared embeddings. These are counterfactual derivatives,
not candidate methods.

The frozen decision order is: RMS-denominator derivative amplification at least
`2x`; otherwise fractional-over-Cartesian amplification at least `3x`;
otherwise `Q^2m` gradient at least the full-field gradient; otherwise a
parameter group with at least 50% squared mass; otherwise distributed scale.
No coordinate target or optimizer step is permitted.

## Execution qualification

The first launch was rejected before attribution because a WSL `/tmp`
worktree disappeared across a distro restart and therefore loaded the active
source instead of `e25f432`.  The second launch loaded the correct source but
exposed CUDA `index_add_` reduction-order noise (`1.60e-4`) above the exact
carrier-reconstruction check.  Neither launch produced an attribution result.
The successful run used PyTorch deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; this made the independently reconstructed
carrier bitwise identical without changing any scientific threshold or
decision rule.

## Result

The live-RMS and detached-RMS carriers have exactly the same forward values.
Their fractional-output gradient norms are respectively `373.2660` and
`617.1612`, so the RMS-denominator derivative amplification is only `0.6048x`:
RMS differentiation is not the source.  The Cartesian-output gradient norm is
`5.1712`, while the physically transformed fractional-covector gradient is
`373.2660`, a `72.1818x` amplification.  The median lattice spectral norm is
`8.1628`, consistent with this cell-scale effect.

`Q^2m` contributes a gradient norm of `121.6384`, only `0.3259x` the complete
field, so carrier-order deletion is not authorized.  The edge encoder contains
`71.9320%` of the complete fractional-output squared gradient, but it is
downstream of the already-dominant chart transformation in the frozen causal
priority.  The decision is therefore `fractional_chart_dominant`.

The only authorized successor is a separately frozen, mathematically
equivalent Cartesian-covector loss metric: retain the exact fractional score
for the reverse process, but compare prediction and target after applying the
inverse lattice covector chart.  No RMS-epsilon tuning, carrier-order deletion,
parameter rescaling, target fitting, or later Gate is authorized by this
result.
