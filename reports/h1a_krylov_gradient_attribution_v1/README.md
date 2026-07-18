# H1a Krylov production gradient attribution v1

Status: **frozen before run; target-free and read-only.**

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
