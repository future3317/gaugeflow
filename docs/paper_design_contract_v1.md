# GaugeFlow-Piezo production design contract v1

The normative method source for the next production implementation is
`E:/Downloads/GaugeFlow_PiezoGen_Revised.tex`, SHA-256
`9ad4ed018600a62b5f663255a1e0a4d59abcdc26303e523a4f151bdfaf07dd31`.
The source was supplied on 2026-07-16. Superseded Gate A--P5-C0 and Q0
artifacts do not become evidence for this architecture. Their final pre-cleanup
state is recoverable from tag `archive/pre-production-cleanup-20260716`; the
active tree keeps only a condensed research history.

## Execution boundary

The revised paper defines a new S0--S5 sequence. S0 mathematical qualification
and the tensor-free S1a implementation closure are authorized. No tensor
fine-tuning, learned tensor oracle, MLIP screening, relaxation, DFT, or DFPT is
authorized by this document.

## Versioned parent--distortion--child amendment

An exact symmetry blueprint is a parent prior, not a completeness claim for the
final piezoelectric phase. The active code now includes versioned contracts for
low-index commensurate symmetry breaking:

- `ParentBlueprint` and P1 `ParentBlueprintBatch`;
- HNF `DistortionBlueprint` with `det(B) <= 4` and at most two active modes;
- finite OPD branches selected before reduced continuous amplitudes;
- child group `G_parent^B` intersected with active OPD stabilizers;
- mass-weighted mode displacement, child-invariant strain and a fail-closed
  0.10 Angstrom projected residual;
- reachable-child compatibility marginalization, never a hard tensor router on
  the parent group;
- gauge-safe phonon subspace, mode-effective-charge and generalized-mode-force
  targets.

These are mathematical/software interfaces only. They do not authorize a mode
catalogue build, parent-pair training, PES-teacher use, hierarchical generation,
tensor conditioning, relaxation, DFT or DFPT. H0 v1 is frozen as
`H0_not_passed_stop_before_H1`; H0-A, H0-B and H0-C have subsequently qualified in
the versioned v4 repair. H0-B uses full-universe force-constant algebra plus a
frozen long-tail/stratified numerical audit rather than claiming a second
10,034-material eigendecomposition sweep. H0-C uses frozen TensorNet/QET
MatPES-PBE-2025.2 teachers under an offline-only 512/32 qualification. H0-D-v1
remains frozen failed, while H0-D-v2 qualifies the complete abstract finite
affine OPD catalogue using compact displacement actions and vectorized
fixed-space-lattice enumeration. H0-E-v1 concrete parent realization is now
frozen failed: all 125 discovered candidates qualified, but coverage was
`125/1024=0.12207<0.15`. H1a, H1b, H2,
H3, H4, H5 and H6 remain strictly ordered and unauthorized.

## Production state spaces

- Chemical tokens are `0..117 <-> Z=1..118`; token `118` is a distinct
  absorbing mask and is never decoded as an element.
- Periodic coordinates use a wrapped Gaussian density on the graphwise
  translation quotient. A hard nearest-image log map is not the production
  probability path.
- Lattices are represented by log volume and a trace-free log-metric shape;
  the shape is projected into the full-O(3) point-group-compatible subspace.
- The tensor condition is a proper-SO(3) orbit. Improper operations enter only
  the physical Reynolds compatibility router.
- The production conditioner is a Stratified Cartesian Gauge Atlas with a
  state-dependent finite prior, multiplicity correction, smooth stratum
  blending and residual descriptor-frame marginalization. Harmonic code is
  available only in the frozen Git archive and is never a runtime fallback.

## Code ownership

Production primitives live under `gaugeflow.production`. Superseded
continuous-logit/ODE implementations were removed from the active tree after
archival and must not be reintroduced as compatibility paths.

## S1a implementation amendment

The tensor-free production path uses absorbing categorical elements, a wrapped
translation-quotient coordinate score, and cosine-VP clean log-volume/log-shape
prediction. The clean lattice parameterization was introduced only after the
frozen v1--v1.2 high-noise raw-score rollouts failed. S1a-I0 v1.3 passed a
bounded single-panel CUDA software closure; it is not a real-data generation
claim and does not authorize tensor conditioning.

## S0 advancement rule

S0 is complete only when every regression named in the paper's code contract
passes in the pinned WSL CUDA environment, the model signature contains no
target-only metadata, and no fixed periodic image cube is used by the
production wrapped kernel.

## Historical S0 qualification

S0.1--S0.4.1 qualified the mathematical interfaces, scalable wrapped quotient,
space-group metric charts and current 4,032-candidate Cartesian prior. Their
frozen runners, harmonic reference and per-run reports were removed from the
active package after the conclusions were recorded in the manuscript and
`docs/research_iteration_history.md`. Exact reproduction uses Git tag
`archive/pre-runtime-cleanup-20260717`; those files must not be restored as
production dependencies.
