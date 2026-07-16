# Hierarchical parent--distortion--child design v1

## Decision

The exact space-group blueprint remains valuable as a parent prior, but it is
not identified with the symmetry of the final generated material. GaugeFlow is
extended by a second, explicitly symmetry-breaking stage. This addresses the
specific failure mode in which an exact blueprint confines generation to common
ideal ordered prototypes and cannot represent polar soft modes, metastable
commensurate distortions, or coupled mode condensation.

The active factorization is

\[
p(x_c\mid[e])=\sum_{b_p,x_p,d}
p_{\theta_p}(b_p,x_p\mid c_{\rm inv}([e]))
p_{\theta_d}(d\mid x_p,[e])
p_{\theta_q}(y\mid x_p,d,[e])
\delta[x_c=\Phi(x_p,d,y)],
\]

with parent blueprint \(b_p\), generated parent \(x_p\), discrete distortion
blueprint \(d=(B,\{k_l,\Gamma_l,c_l,z_l\})\), and continuous variables
\(y=(s,\eta,\delta r)\). The current exact-symmetry generator is the strict
branch \(d=\varnothing\), so the extension does not introduce a second legacy
runtime.

## What is implemented now

- `ParentBlueprint` and the leakage-free `ParentBlueprintBatch` P1 substrate;
- canonical low-index HNF validation with `det(B) <= 4`;
- commensurability checks for `B k`;
- finite OPD branches with orthonormal fixed-space bases and stabilizer indices;
- a versioned `ModeCatalog` contract;
- a `DistortionBlueprint` with at most two active modes;
- the child-operation rule
  \(H(d)=G_p^B\cap\bigcap_{l:z_l=1}H_{l,c_l}\);
- exact supercell coset expansion in the row-lattice convention;
- mass-weighted mode reconstruction, invariant strain, translation-gauge
  removal, Reynolds-projected small residuals, and a fail-closed residual RMS
  budget;
- gauge-safe phonon subspace, mode-effective-charge and generalized-mode-force
  targets;
- reachable-child compatibility marginalization. Tensor compatibility is no
  longer a hard filter on the parent group.

The production tensor-free trainer still runs only the parent P1 branch. That
is intentional: these primitives do not authorize H2--H6 training before the
real-data S1a parent generator passes.

## Conditioning placement

Before a concrete parent geometry exists, parent and distortion decisions may
use tensor-orbit invariants, the physical-zero flag and terminal-child
compatibility residuals. They must not use a Cartesian representative frame.
After \(x_p\) and \(d\) exist, the S0.4.1 Cartesian atlas may condition the
mode-amplitude, child-strain and bounded-residual heads. This avoids asking the
atlas to choose a parent frame that has not yet been generated.

For reachable paths \(d\in\mathcal C(G_p)\), the router uses

\[
p(G_p\mid[e])\propto p_0(G_p\mid c_{\rm inv})
\sum_d p_0(d\mid G_p)\exp[-\beta_d r_{H(d)}([e])^2].
\]

Thus a centrosymmetric parent is legal when an inversion-odd mode reaches a
compatible non-centrosymmetric child. The exact parent branch can have zero
probability for a nonzero piezoelectric target without deleting the parent and
all of its physically reachable children.

The code exposes both an invariant-only default and
`ReachableChildCompatibilityRouter.route_from_logits(...)`. The latter accepts
separate parent-prior logits and path-prior logits, so the production path
model can evaluate $p_0(d\mid x_p)$ after a concrete parent has been generated.
Explicit `-inf` catalogue masks are supported. A parent with no compatible
child receives zero probability; if every parent in a batch row has no
compatible reachable child, routing fails closed instead of returning NaNs.

## External data activation evidence

Large data remain outside the repository under `E:/DATA`. The data-center
manifest dated 2026-07-17 reports:

| Domain | Activated source | Evidence and present limit |
|---|---|---|
| structure | Alex-MP-20 | local train/val/test Parquet files and CC-BY-4.0 dataset card are present; GaugeFlow has not yet frozen its own formula/prototype-disjoint split or parent decompositions |
| tensor | JARVIS/GMTNet | 5,000 source rows; 4,998 full-O(3)-audited targets; reserved for later oracle/tensor gates |
| auxiliary tensor | Materials Project | 3,316 records; retained as a separate source, not a drop-in replacement for JARVIS |
| nonzero-q modes | PhononDB | 10,034 archives have complete displacement--force data and a supercell matrix; force constants, dynamical matrices, frequencies and eigenvectors are derivable but not stored and are not yet qualified here |
| Gamma modes | JARVIS-DFPT | 4,995 complete schema-4 records among the 4,998-ID cohort; three named source failures are documented externally |
| PES | MatPES-PBE 2025.2 | streaming JSONL and BSD-3-Clause card are present; this is a training dataset for a teacher, not itself a frozen teacher checkpoint |

High-confidence data conclusions:

1. `source_id` and `source_database` are mandatory. JARVIS, MP and PhononDB
   numerical labels must not share an uncalibrated absolute frequency/response
   scale.
2. PhononDB currently supports a derivation pipeline, not direct claims that
   stored q-point eigenvectors already exist: the audited stored counts for
   force constants, q-points, frequencies and eigenvectors are all zero.
3. MatPES data availability does not establish availability or qualification of
   `TensorNet-PES-MatPES-PBE-2025.2`. A frozen checkpoint and independent
   disagreement teacher must be separately activated.
4. Parent--child pairs and OPD catalogues do not yet exist. H3 cannot start
   until the split is frozen first and every candidate parent/mode scan follows
   the child's split.

## Qualification order and scientific boundary

1. **H0:** freeze source versions/hashes, the formula/prototype split, catalogue
   provenance, cross-source joins and derivation attestations.
2. **H1:** qualify the real-data tensor-free parent generator.
3. **H2:** qualify source-calibrated mode sign/magnitude, degenerate subspaces
   and mode effective charge.
4. **H3:** qualify parent--child reconstruction without a tensor condition.
5. **H4:** qualify frozen-teacher mode scans and projected energy/force/stress
   targets.
6. **H5:** qualify tensor-free hierarchical child generation.
7. **H6:** only then attach the Cartesian atlas to continuous mode variables.

The v1 scientific domain is ordered, stoichiometric, periodic inorganic
crystals with `det(B) <= 4` and at most two commensurate active modes. Partial
occupancy, substitutional disorder, vacancies, charged defects,
incommensurate phases, larger supercells and finite-temperature ensembles are
not silently represented by the residual branch. They require separately
versioned state spaces and data semantics.
