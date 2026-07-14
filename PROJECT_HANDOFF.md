# GaugeFlow implementation handoff

E:\CODE\T2C-Flow\gaugeflow is the active implementation. It is standalone:
it uses PyTorch, PyG, e3nn, and pymatgen, but does not import FlowMM at runtime.

The legacy E:\CODE\T2C-Flow\flowmm working tree was removed after its local
T2C changes and run metadata were preserved in
E:\CODE\T2C-Flow\legacy_backups\flowmm_local_2026-07-14. Reconstruct it from
the upstream FlowMM repository plus that patch set only when a historical
baseline must be rerun. Do not add new GaugeFlow features there.

## Active entry points

- scripts/train.py: standalone orbit-response-field flow matching.
- scripts/sample.py: tensor-orbit sampling; intentionally accepts no target
  lattice.
- src/gaugeflow/tensor.py: tensor convention, SO(3) orbit set, complete
  vector response fields, isotypic normalization.
- src/gaugeflow/model.py: graph-level latent alignment plus PyG-assisted
  geometric message passing; scalar/vector updates use only rotational
  invariants and covariant Cartesian vectors. The active ``orbit_alignment``
  condition uses a finite tensor orbit and a soft automorphism posterior inferred from the
  current generated state, so train and tensor-only sampling see the same
  information.
- src/gaugeflow/manifold.py: torus coordinates and SPD lattice-log flow
  coordinates.
- src/gaugeflow/unit_cell.py: tracked Niggli cell reduction for unit-cell
  basis equivalence.
- src/gaugeflow/stabilizer.py: proper (determinant +1) crystallographic
  stabilizer extraction and analysis utilities; improper parity operations are
  not pooled and target-CIF stabilizers are not model inputs.
- tests/: standalone unit and flow smoke tests.

## Data

GaugeFlow owns its paired CSV at
E:\CODE\T2C-Flow\gaugeflow\data\piezo\{train,val,test}. It retains the
historical TensorOrbit-JARVIS-v1 artifact at
E:\CODE\T2C-Flow\gaugeflow\data\tensororbit_jarvis_v1, containing a
4,000/499/499 ID split and Reynolds-projected condition cache. v1 is not
formula-disjoint: its audit found 165 cross-split formula groups (672 rows) and
56 structural near duplicates. Preserve it unchanged for reproducibility only;
the versioned TensorOrbit-JARVIS-v2 activation candidate is required before
future validation/test claims. GaugeFlow reads these files directly through
pymatgen and does not import, call, or wait for another project's model, graph
cache, or checkpoint.

PiezoJet is a separate prediction project. Its current weights are not a
GaugeFlow oracle and must not select GaugeFlow methods, hyperparameters, or
generated candidates. At most, a future PiezoJet checkpoint may enter the
separately frozen oracle-qualification procedure as one diagnostic ensemble
member after passing its criteria.

## Conditioning contract

The active ``orbit_alignment`` implementation uses a soft latent-automorphism
posterior derived from the current generated lattice, coordinates, and
atom-type state. Integer unimodular matrices are proposal indices only: each is
polar-projected to a proper SO(3) Cartesian rotation before it can rotate a
tensor. Its weight combines lattice-action residual and type-aware periodic
self-match. This is intentionally not exact space-group recovery for a noisy
state. Using the paired target CIF during training remains prohibited. The
legacy ``double_coset`` configuration name is kept only to load old checkpoints
and maps to ``orbit_alignment``.

``direct_irrep`` is a genuine Cartesian direct-interaction baseline: it passes
the two covariant contractions ``e_{ijk} n_j n_k`` and ``e_{ijk} n_i n_k`` to
the graph messages, with scalar invariant contractions for gating. It avoids
spherical-harmonic and Clebsch--Gordan evaluation. CFG is trained by graphwise
condition dropout; the Boolean condition-present mask keeps physical zero
tensors separate from the learned null condition.

## Runtime

Run WSL Ubuntu-22.04 as user future04 in
/mnt/e/CODE/T2C-Flow/gaugeflow. The micromamba environment is flowmm-t2c.

Read E:\CODE\T2C-Flow\03_modify.md and the paper-side REDESIGN_PLAN.md before
altering the model contract.
