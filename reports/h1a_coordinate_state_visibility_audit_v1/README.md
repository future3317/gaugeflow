# H1a coordinate state-visibility audit v1

Status: **completed; formal invariance failed, causal attribution rejected**.

The translation-only quotient DSM target is evaluated on two row-label
representatives of the same unlabeled endpoint while keeping the visible noisy
state bitwise identical.  A vectorized cyclic permutation is applied inside
every repeated-species block, so element tokens, composition, lattice,
coordinate set and all denoiser inputs are unchanged.  Any target difference
is therefore arbitrary CIF-row information unavailable to a permutation-
equivariant denoiser.

This read-only attribution does not change the failed coordinate-pretraining
result.  The preregistered rule associated a raw representative difference
with a joint translation--permutation quotient target, but the subsequent
likelihood-weighted causal check below rejects that repair: the alternative
representatives have negligible posterior mass.  Nothing here authorizes
extra steps, joint initialization, H1b, tensor conditioning or later Gates.

## Result

Of 128 fixed validation graphs, 111 (86.72%) contain a repeated-species block
and therefore admit a nonidentity type-preserving row relabeling.  Across all
seven fixed times, the visible noisy state and unlabeled endpoint set are
identical to machine precision, but the two translation-only DSM targets have
relative difference 1.4016--1.4161 and cosine -0.0484--0.0407.  The current
endpoint-conditioned target is therefore not itself a deterministic function
on the endpoint's node-permutation quotient.

That formal difference is not a causal explanation of the failed one-pass
fit.  A likelihood-weighted follow-up over the same two representatives gives
zero minority posterior mass through `t=0.2`; at `t=0.5` the maximum minority
mass is only `5.42e-14`.  The largest posterior-weighted conditional variance
relative to mixture-target energy is `2.22e-15`.  The visibly different target
belongs to an overwhelmingly impossible matching and therefore contributes no
material DSM variance on this panel.

A conditional DSM sample need not be uniquely recoverable from the noisy
state, and ordinary labeled DSM remains an unbiased estimator of the
symmetrized population score.  This audit consequently does **not** authorize
a permutation-marginal target, Hungarian relabeling, Sinkhorn surrogate or
probability-path change.  The coordinate-pretraining failure must instead be
located in optimization/representation or generalization.  The predeclared
formal invariance check remains failed in the JSON; the posterior-weighted
causal refinement is explicitly post-hoc and does not rewrite that check.
