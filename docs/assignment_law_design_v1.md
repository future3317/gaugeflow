# GaugeFlow assignment law: evidence and next design

## Evidence boundary

The failed Q1 established that exact count constraints, quotient likelihood,
sampling and MAP were implemented correctly, while its action-only unary
representation did not generalize. The geometry-complete carrier v2 repairs a
real serialization defect: all 454 carriers now expose one species-free parent
coordinate per action node, a canonical row-HNF supercell, translation cosets
and the conjugated finite-site action.

The subsequent no-training audit shows that geometry is necessary but that a
static pair-energy histogram is not a complete assignment law. Geometry-aware
unary signatures resolve 47.36% of carriers. Complete target-free two-point
distance descriptors resolve 87.87% of the remaining exact collision classes,
with mean quotient ceiling 0.93933. The IID test stratum is harder: pair
resolution is 0.63636 and its mean ceiling is 0.81818. A production mechanism
must therefore see the evolving global coloring, not only sum fixed unary or
pair energies.

## Why not add an arbitrary pairwise Gibbs energy

A configurational cluster expansion provides the physically natural hierarchy

\[
E(A;X)=J_\varnothing(X)+\sum_iJ_i(A_i;X)
       +\sum_{i<j}J_{ij}(A_i,A_j;X)+\cdots .
\]

This hierarchy is well founded in alloy theory, but a dense learned pair term
destroys the mixed-radix normalization used by Q1. Exact evaluation of

\[
Z_{\mathbf n}(X)=\sum_{A:\,\operatorname{count}(A)=\mathbf n}
\exp[-E(A;X)]
\]

is generally exponential unless the interaction graph has bounded treewidth
or another special factorization. Adding such an energy and then using
Hungarian repair, pseudo-likelihood or an unreported variational partition
function would weaken the probabilistic contract.

The cluster-expansion literature is nevertheless useful as an expressivity
guide. Sánchez, Ducastelle and Gratias introduced the multicomponent cluster
basis ([DOI 10.1016/0378-4371(84)90096-7](https://doi.org/10.1016/0378-4371(84)90096-7)).
Hart and Forcade give symmetry-correct derivative-structure enumeration
([DOI 10.1103/PhysRevB.77.224115](https://doi.org/10.1103/PhysRevB.77.224115))
and fixed-concentration enumeration
([DOI 10.1016/j.commatsci.2012.02.015](https://doi.org/10.1016/j.commatsci.2012.02.015)).
These works support the group-action and collision audits, but they do not make
an arbitrary neural Gibbs partition function cheap.

## Proposed law

The next bounded mechanism is a **residual-stabilizer, remaining-count
orderless autoregressive occupation law**. Let the species-free carrier be

\[
X=(F_{\rm parent}^{\rm HNF},L_{\rm parent}^{\rm HNF},G\curvearrowright V),
\]

let the oracle composition for the first Gate be the count vector
\(\mathbf n\), and draw a reveal order \(Z=(z_1,\ldots,z_N)\) uniformly and
independently of the target coloring. At step \(r\), the model receives only
the carrier, the revealed partial coloring \(A_{S_{r-1}}\), its mask and the
remaining counts. A permutation-equivariant all-pair Cartesian encoder returns
site--species logits \(\ell_{ik}\). For the next revealed site,

\[
p_\theta(A_{z_r}=k\mid Z,A_{S_{r-1}},X,\mathbf n)
=
\frac{n^{(r)}_k\exp\ell_{z_rk}}
{\sum_{q:n^{(r)}_q>0}n^{(r)}_q\exp\ell_{z_rq}}.
\]

The factor \(n_k^{(r)}\) is not an ad-hoc reweighting. At zero logits it makes
every distinct complete assignment with the prescribed counts equiprobable.
After sampling, the selected count is decremented. Consequently terminal masks
and count repair are unnecessary and exact composition is guaranteed.

The joint path law is

\[
p_\theta(A,Z\mid X,\mathbf n)
=\frac1{N!}\prod_{r=1}^N
p_\theta(A_{z_r}\mid Z,A_{S_{r-1}},X,\mathbf n),
\]

so it is normalized by construction. Its order-marginal assignment law is

\[
p_\theta(A\mid X,\mathbf n)
=\frac1{N!}\sum_{Z\in S_N}\prod_{r=1}^N p_\theta(A_{z_r}\mid\cdots).
\]

For a fixed target assignment this marginal is exactly auditable on small
systems by subset dynamic programming:

\[
D(\varnothing)=1,\qquad
D(S)=\sum_{i\in S}D(S\setminus\{i\})
p_\theta(A_i\mid A_{S\setminus\{i\}},i,X,\mathbf n),
\]

\[
p_\theta(A\mid X,\mathbf n)=D(V)/N!.
\]

The quotient probability retains unique-orbit semantics,

\[
p_\theta([A])=
\sum_{\widetilde A\in\operatorname{UniqueOrbit}_{G}(A)}
p_\theta(\widetilde A),
\]

so duplicate crystallographic operations never add likelihood. At a partial
state the diagnostic residual group is

\[
\Gamma_r=\{g\in G:\;gS_r=S_r,\ A_{g(i)}=A_i\ \forall i\in S_r\}.
\]

Equivariance requires equal predictions on each unresolved
\(\Gamma_r\)-orbit. The first implementation will verify this equality rather
than introduce a compatibility fallback.

If \(P_\sigma\) relabels nodes, uniform reveal orders push forward uniformly
and an equivariant score network satisfies

\[
p_\theta(P_\sigma A\mid P_\sigma X,\mathbf n)
=p_\theta(A\mid X,\mathbf n).
\]

Thus stochastic reveal order supplies symmetry breaking without CIF-row
positional embeddings or an independent node latent.

## Efficient implementation

The carrier encoder uses the full periodic all-pair graph. With \(N\le20\),
this is at most 400 directed pairs and avoids hard-neighbor discontinuities.
Static distance/RBF and parent-action features are computed once and cached.
Training samples an order and one or more reveal depths in parallel; sampling
uses at most 20 categorical steps. There is no factorial enumeration in the
runtime path. Exact subset DP is restricted to the software qualification
panel, while larger-carrier calibration uses common random reveal orders with
structure-level uncertainty.

Orderless autoregressive training has precedent in NADE
([arXiv:1310.1757](https://arxiv.org/abs/1310.1757)) and autoregressive
diffusion ([arXiv:2110.02037](https://arxiv.org/abs/2110.02037)). The crystal
literature supplies useful comparison points through DiffCSP++
([arXiv:2402.03992](https://arxiv.org/abs/2402.03992)), SymmCD
([arXiv:2502.03638](https://arxiv.org/abs/2502.03638)) and Wyckoff Transformer
([arXiv:2503.02407](https://arxiv.org/abs/2503.02407)). GaugeFlow's distinct
contribution is the combination of an exact generated composition, a
species-free parent-action carrier, stochastic reveal-order marginalization,
residual-stabilizer audits and a count-exact normalized occupation path.

Tarlow et al.'s recursive-cardinality construction
([arXiv:1210.4899](https://arxiv.org/abs/1210.4899)) motivates using explicit
remaining-count state and exact small-system inference rather than hiding a
cardinality constraint in a post-processing step.

## Gate order

1. Q0 has proved normalization, exact counts, relabel equivariance,
   unique-orbit quotient probability and subset-DP agreement on exhaustive
   small systems. It also exposed and repaired an AMP reduction-buffer dtype
   defect. Its final RTX 4090 no-grad qualification is `5.07 ms / 99.05 MiB`.
2. Freeze a single-seed IID oracle-composition training Gate using the existing
   IID fit/calibration/test split.
3. Report untouched formula/prototype-disjoint validation/test as OOD stress,
   never as the IID calibration result.
4. Only after IID assignment qualifies may generated composition be connected.

This design does not authorize `p(N)`, L1/M1, tensor conditioning, relaxation,
DFT or DFPT.
