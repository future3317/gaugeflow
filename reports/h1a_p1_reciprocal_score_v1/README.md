# H1a projective reciprocal score v1

Status before execution: **frozen, not run**.

## Rationale

The log-uniform torus schedule moved the generated nearest-neighbour median
from 1.6031 Å to 2.19392 Å and reduced normalized distance Wasserstein from
1.97287 to 0.95972, but did not reach 0.75.  Its validation coordinate loss
plateaued near 0.50.  The next isolated mechanism therefore changes periodic
vector-field expressivity, not the schedule, loss weights, capacity, optimizer,
or training budget.

DiffCSP reports faster coordinate convergence from fractional Fourier features
than from a periodic multigraph.  Ewald message passing independently shows
that reciprocal-space structure factors provide an inexpensive nonlocal
augmentation to local molecular graph networks.  The present mechanism adapts
those observations to a translation-quotient *score covector*, with explicit
cell covariance and without a grid or pair--mode loop.

For row-style `r=fL`, define the physical projective reciprocal ball

```text
K(L) = { k in Z^3 \ {0} : |2*pi*k*L^-T| < q_max } / {k ~ -k}.
```

Let `a_ic` be learned scalar node channels and

```text
Z_kc = N^-1/2 sum_j a_jc exp(i 2*pi*k.f_j).
```

The global Cartesian score residual is

```text
s_i = sqrt(2/(K*C)) sum_[k],c w_c(|q_k|) a_ic
      Im[exp(-i 2*pi*k.f_i) Z_kc] q_k,
q_k = 2*pi*k*L^-T.
```

The `sqrt(2)` factor makes the `k~-k` quotient exactly equal to the normalized
full symmetric ball.  Expanding the imaginary term gives pair contributions
`a_ic*a_jc*sin(2*pi*k.(f_j-f_i))*q_k`; exchanging `i,j` changes their sign, so
the graph sum is zero.  A common translation cancels from every phase
difference.  Under `L'=BL, f'=fB^-1, k'=Bk` for `B in GL(3,Z)`, both the phase
and physical `q_k` are unchanged.  Under `Q in O(3)`, `L'=LQ` gives `q'=qQ`,
so the output is Cartesian covariant.  The radial cosine envelope and its first
derivative vanish at `q_max`, preventing a finite jump when the finite mode set
changes.

Structure factors reduce naive `O(N^2*K*C)` pair--mode evaluation to
`O(N*K*C)` via two batched `index_add` reductions.  On a real 64-graph RTX
4060 Ti batch, the exact projective ball contained 46/107/277 min/median/max
modes, achieved 525.26 graphs/s, and used 1,470.72 MiB peak allocated memory.
The new complete model has 4,287,073 parameters; replacing the prior direct
edge-output skip head makes it about 186k parameters smaller rather than
increasing capacity.

## Frozen screen

Seed 5501 must complete 20,000 steps with finite training, final coordinate
validation loss at most 0.47, total validation ratio at most 0.65, generated
nearest-neighbour median at least 2.3 Å, zero failures/masks, and valid
lattices.  Only a complete pass permits seeds 5502/5503.

References:

- Jiao et al., *Crystal Structure Prediction by Joint Equivariant Diffusion*,
  NeurIPS 2023, arXiv:2309.04475.
- Kosmala et al., *Ewald-based Long-Range Message Passing for Molecular
  Graphs*, ICML 2023, arXiv:2303.04791.
- Li et al., *Fourier Neural Operator for Parametric Partial Differential
  Equations*, ICLR 2021, arXiv:2010.08895.
