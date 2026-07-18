# H1a pairwise reciprocal-torus operator v1

Status: **completed; qualified for one separately frozen coordinate-only
experiment**.

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

## Result

Every frozen numerical and capacity check passed.  On the FP64 reference
panel, maximum errors were `2.78e-17` for common translation, `1.19e-16` for
integer coordinate representatives, `6.94e-18` for node permutations,
`3.47e-17` for physical `O(3)`, and `4.86e-17` for a nontrivial unimodular cell
change.  The projective `k~-k` calculation agreed with the explicitly doubled
ball to `1.04e-17`; an independent scalar pair/mode loop agreed with the
production tensor reductions to `1.30e-17`.  Graphwise residual sums were at
most `8.67e-19`, cutoff value and radial derivative were exactly zero, and all
forward/backward gradients were finite and nonzero where expected.

The production module has no graph, pair, or reciprocal-mode Python loop.  It
adds only 9,072 parameters to the 4.47M model, for 4,482,234 total parameters.
On the frozen first 64 train graphs (596 nodes, 3,062 unordered pairs), the
complete projective reciprocal ball contained 46/107/277 minimum/median/maximum
modes.  Ten complete coordinate-only BF16 optimizer steps on the RTX 4060 Ti
ran at `490.77 graphs/s` with `1,734.19 MiB` peak allocated memory.  The FP32
relative deviation from FP64 was `6.76e-7`; BF16 differed from FP32 by
`0.01944`, below the frozen `0.05` bound.  The tensor-free bypass constructed
zero atlas candidates.

This is an operator qualification, not evidence that the added residual
improves learned coordinates or generated structures.  It authorizes only one
versioned, single-seed, exactly one-pass coordinate-pretraining experiment
with the old local field unchanged.  Failure of that experiment requires
removing this residual from production rather than retaining it as a dormant
fallback.
