# H1a pairwise reciprocal-torus operator v1

Status: **frozen, not run**.

The one-pass coordinate-only run reduced validation loss monotonically but
stopped at `0.54928 > 0.35`; raw training loss was still `0.49768`, so EMA lag
and ordinary train--validation generalization are not the main cause.  A
likelihood-weighted follow-up also rejected repeated-species row permutations
as a material source of target variance.  This protocol therefore qualifies
one coordinate-field representation change before any new training.

This is not a revival of the archived `h1a_p1_reciprocal_score_v1`.  That
operator factorized every pair coefficient as products of node amplitudes,
`a_ic a_jc`, and its active teacher-forced residual did not repair joint free
sampling.  The proposed operator instead assigns an arbitrary signed,
symmetric coefficient `b_ijc=b_jic` to every unordered pair, while retaining a
small shared channel factorization over reciprocal modes.

For row-style Cartesian coordinates `r=fL`, define the physical reciprocal
covector and projective ball

```text
q_k = 2*pi*k*L^-T,
K(L) = {k in Z^3 minus {0}: |q_k| < q_max} / {k equivalent to -k}.
```

For symmetric pair channels `b_ijc` and radial mode channels `w_c(|q_k|)`,
the residual assigned to the first endpoint of pair `{i,j}` is

```text
g_ij = sqrt(2) / sqrt(N*K*C)
       sum_[k],c b_ijc w_c(|q_k|)
       sin(2*pi*k.(f_i-f_j)) q_k,
g_ji = -g_ij.
```

Using only one of `k,-k` is exact because both give the same
`sin(phase)*q` contribution; `sqrt(2)` retains the normalization of the full
symmetric ball.  Pair antisymmetry makes the graphwise sum exactly zero.
Integer coordinate shifts and common translations leave every phase
unchanged.  For a unimodular basis change `L'=BL`, `f'=fB^-1`, the relabeling
`k'=kB^T` preserves both phase and physical covector.  A physical orthogonal
rotation sends `L` and every `q_k` through the same Cartesian action.  The
finite ball uses a cosine envelope whose value and first derivative vanish at
the cutoff.

The implementation must enumerate complete unordered pairs and the complete
physical reciprocal ball with tensor operations.  It may not contain a graph,
pair, or mode Python loop and may not retain a disabled compatibility branch.
The existing local score remains unchanged, so the qualification isolates a
complementary global periodic residual rather than replacing already useful
real-space geometry.

The frozen checks cover translation, integer representatives, node
permutations, `O(3)`, unimodular cell changes, projective duplicate correction,
zero graph mean, cutoff continuity, finite gradients, an explicit pair oracle,
and a real 64-graph RTX 4060 Ti BF16 training-step capacity measurement.  Only
a complete pass permits a separately frozen one-seed, one-pass coordinate
experiment.  No result here can authorize joint initialization or H1b--H6.
