# Proposed next E1 substrate: explicit composition and assignment states

## Why the state space, not another head, must change

The three bounded mechanisms agree on the same decomposition.  Once the target
species multiset is supplied only as an offline oracle, the terminal site
logits achieve about 0.70 site accuracy and 0.27--0.36 exact assignment.  In
contrast, every free reverse run has zero exact composition.  E1.3 proves that
an exact current-token histogram fixes low-noise counting but cannot create a
coherent formula from the nearly uniform high-noise state.

The occupational variable should therefore be factorized as

\[
  A=(C,Y),\qquad
  C\in\mathcal C_N=
  \{n\in\mathbb N^{118}:\sum_k n_k=N\},
  \qquad
  Y\in\mathcal A(C),
\]

where `C` is an unordered graph composition and `Y` assigns that multiset to
sites.  This is a refinement of the existing element modality, not a fourth
physical modality and not target-composition conditioning.

## Bounded qualification order

1. Audit the active train split for atom count, number of distinct species,
   count-partition complexity, and active vocabulary.  This determines whether
   exact sparse composition states are computationally practical.
2. Define a normalized sparse composition generator over ordered
   species--positive-count pairs.  Ordering species by atomic number is only a
   unique serialization of an unordered histogram; it does not introduce CIF
   site order.
3. On synthetic `N<=20` states, prove normalization, exact atom-count
   conservation, permutation invariance, sampling/reconstruction closure, and
   absence of duplicate likelihood for identical species.
4. Only after that kernel qualifies, run one separately frozen small E1 screen
   in which the model generates `C` and then uses the already qualified
   count-constrained assignment readout for `Y`.

A convenient exact factorization is

\[
  p_\theta(C\mid z,N)
  =p_\theta(S\mid z,N)
   \prod_{r=1}^{S}
   p_\theta(k_r\mid k_{<r},z,N)
   p_\theta(n_r\mid n_{<r},k_{\le r},z,N),
\]

with `k_1<...<k_S`, positive `n_r`, and `sum_r n_r=N`.  Masks enforce only
these mathematical constraints, never a target formula or target-specific
vocabulary.  At `N<=20`, the sequence is short and its cost is negligible
beside 100 graph-denoiser evaluations.

This factorization supplies a genuine stochastic formula prior at the
high-noise boundary while retaining exact counts and a permutation-invariant
scientific meaning.  It should not be implemented or trained until the data
audit and exact synthetic qualification freeze its finite state contract.
