# E1.2 graph composition field rationale

## Identified mechanism

E1.1 did not fail because the site logits lacked all chemical structure.  With
model-predicted counts, terminal site accuracy was 0.06175 and count overlap
was 0.08144.  Replacing only those counts by the frozen target counts in an
offline attribution raised accuracy to 0.70861 and exact assignment to
0.30859.  Thus the relative site ranking is useful, while the global species
multiset is not represented by averaging independent site posteriors.

## Graph composition posterior

Let `h_i` denote the final permutation-equivariant node state and let `g` be
the invariant graph context already used by the lattice heads.  E1.2 adds one
graph posterior

\[
  c_\theta(A_t,F_0,L_0,N,t_A)
  =\operatorname{softmax}(\operatorname{MLP}_{\rm comp}(g))
  \in\Delta^{117}.
\]

It predicts atom-mass abundance, not a target formula ID.  Its expected species
embedding

\[
  z_{\rm comp}=\sum_{k=1}^{118}c_{\theta,k}E_k
\]

is broadcast to the site head.  All sites therefore share a global chemical
context, while their local equivariant states determine the assignment.  The
posterior is permutation invariant and depends only on the current noisy
element state, observed E1 geometry/lattice, graph size, and diffusion clocks.
Target formula, counts, IDs, and metadata are not model inputs.

For target counts `n_k` and total atom count `N`, the graph loss is the proper
cross entropy

\[
  \mathcal L_{\rm comp}
  =-\sum_k\frac{n_k}{N}\log c_{\theta,k}.
\]

It remains equal-weight with the corrupted-site clean-token cross entropy, as
in E1.1; no loss weight is searched.  At terminal decoding the predicted mass
`N c_theta` is rounded by the largest-remainder rule to an integer vector with
exact sum `N`, followed by the existing at-most-20-by-20 Hungarian MAP
assignment.  This projection enforces only model-predicted counts.

## Boundary

The graph posterior is one head of the unified product-space reverse field,
not a separate composition model or a leaked conditioning variable.  The run
uses the same seed, 2,111 updates, data panel, optimizer, uniform D3PM path,
reverse NFE, and thresholds as E1.1.  It cannot authorize L1 or later work
unless every frozen E1.2 check passes.
