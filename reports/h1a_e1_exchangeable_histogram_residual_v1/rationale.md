# E1.3 exchangeable histogram residual rationale

For a graph with current element tokens `A_t`, the normalized histogram

\[
  h_k(A_t)=\frac{1}{N}\sum_i\mathbf 1[A_{t,i}=k]
\]

is invariant to node relabeling and is the exact graph-level statistic that
E1.2 attempted to reconstruct after several learned compressions.  The
information audit showed this was destructive: at `t=0.25`, the input
histogram retained 0.85913 target-count overlap while the learned graph
posterior retained only 0.68352.

E1.3 defines the composition base measure

\[
  b_t=a_t h(A_t)+(1-a_t)u,\qquad
  a_t=\cos^2(\pi t_A/2),\quad u_k=1/118,
\]

and predicts

\[
  \log c_\theta
  =\log b_t+(1-a_t)\Delta_\theta(g,h(A_t),a_t).
\]

The clean boundary is exact: at `t_A=0`, the residual gate vanishes and
largest-remainder decoding recovers the current integer counts.  At high
noise the base becomes uniform and the graph network must supply the learned
correction from current geometry, lattice, graph size, and categorical state.
The residual output is initialized small but nonzero, so all internal
parameters receive gradients immediately.

This is not target leakage or a target-specific vocabulary.  The histogram is
computed from the state already consumed by the Markov reverse field in one
`O(N+118G)` vectorized operation.  E1.3 changes neither the D3PM transition nor
the terminal count-constrained assignment, optimizer, loss weight, seed,
training exposure, reverse NFE, or acceptance thresholds.
