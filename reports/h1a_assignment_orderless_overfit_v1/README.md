# Orderless assignment fixed-batch overfit v1

Status: **failed and superseded by a separately frozen pair-context screen**.

The count-exact uniform-order objective, gradients, sampling closure and five of
eight fixed carriers behaved correctly. Aggregate target-quotient probability
lower bound was `0.67945`, categorical retrieval was `0.66602`, and aligned
site accuracy was `0.86011`; all missed the frozen thresholds. Exact
composition and finite gradients were `1.0`.

Read-only attribution localized the failure to three carriers whose unary site
signatures collide. Training each carrier alone for the same 2,000 updates did
not improve over the count-only uniform law, and every reveal depth stayed at
the uniform conditional NLL. The other five carriers reached target-quotient
probabilities `0.944--0.999`. This rules out data loading, gradient-chain,
multi-carrier interference and insufficient steps as the primary cause.

A target-free two-point Cartesian audit then formed, for every pair `(i,j)`,
the endpoint-symmetrized Gram feature

```text
sum_k sym[r(d_ik) outer r(d_jk)].
```

Together with the direct pair RBF, its 152-dimensional feature separated the
three collision classes exactly down to their parent target orbits:
`6 -> 2`, `220 -> 4`, and `420 -> 2`. This authorizes only the v2 fixed-batch
screen. It does not qualify the IID assignment law, generated composition,
`p(N)`, lattice, joint generation, tensor conditioning, relaxation, DFT or
DFPT.
