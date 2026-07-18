# H1a coordinate state-visibility audit v1

Status: **completed; translation-only target failed representative visibility**.

The translation-only quotient DSM target is evaluated on two row-label
representatives of the same unlabeled endpoint while keeping the visible noisy
state bitwise identical.  A vectorized cyclic permutation is applied inside
every repeated-species block, so element tokens, composition, lattice,
coordinate set and all denoiser inputs are unchanged.  Any target difference
is therefore arbitrary CIF-row information unavailable to a permutation-
equivariant denoiser.

This read-only attribution does not change the failed coordinate-pretraining
result.  A measured difference requires a joint translation--permutation
quotient target before another H1a training protocol; it does not authorize
extra steps, joint initialization, H1b, tensor conditioning or later Gates.

## Result

Of 128 fixed validation graphs, 111 (86.72%) contain a repeated-species block
and therefore admit a nonidentity type-preserving row relabeling.  Across all
seven fixed times, the visible noisy state and unlabeled endpoint set are
identical to machine precision, but the two translation-only DSM targets have
relative difference 1.4016--1.4161 and cosine -0.0484--0.0407.  The current
target is therefore not a function on the endpoint's node-permutation
quotient.

This result must be interpreted precisely.  A conditional DSM sample need not
be uniquely recoverable from the noisy state, and ordinary labeled DSM can
still be an unbiased estimator of a symmetrized population score.  The audit
instead proves that arbitrary endpoint row labels contribute enormous target
variance which a permutation-equivariant denoiser cannot use.  Analytically
marginalizing or consistently approximating that nuisance is a justified
Rao--Blackwell repair; a hard Hungarian relabeling or unqualified Sinkhorn
surrogate is not.  The next operator must first pass endpoint-row invariance,
noisy-row equivariance, translation invariance, exact small-N comparison,
finite-gradient and CUDA-cost checks before any new training.
