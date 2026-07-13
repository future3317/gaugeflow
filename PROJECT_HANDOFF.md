# GaugeFlow implementation handoff

E:\CODE\T2C-Flow\gaugeflow is the active implementation. It is standalone:
it uses PyTorch, PyG, e3nn, and pymatgen, but does not import FlowMM at runtime.

The legacy E:\CODE\T2C-Flow\flowmm tree is retained only for reproducible
baselines and historical data conversion. Do not add new GaugeFlow features
there.

## Active entry points

- scripts/train.py: standalone orbit-response-field flow matching.
- scripts/sample.py: tensor-orbit sampling; intentionally accepts no target
  lattice.
- src/gaugeflow/tensor.py: tensor convention, SO(3) orbit set, complete
  vector response fields, isotypic normalization.
- src/gaugeflow/model.py: graph-level latent alignment plus PyG-assisted
  geometric message passing; scalar/vector updates use only rotational
  invariants and covariant Cartesian vectors.
- src/gaugeflow/manifold.py: torus coordinates and SPD lattice-log flow
  coordinates.
- src/gaugeflow/unit_cell.py: tracked Niggli cell reduction for unit-cell
  basis equivalence.
- src/gaugeflow/stabilizer.py: proper (determinant +1) crystallographic
  stabilizer extraction; improper parity operations are not pooled.
- tests/: standalone unit and flow smoke tests.

## Data

The current paired CSV is E:\CODE\T2C-Flow\flowmm\data\piezo\{train,val,test}.csv,
produced from PiezoJet/GMTNet data. GaugeFlow reads the CSV directly through
pymatgen and does not invoke FlowMM. The raw source and fixed split remain at:

- E:\CODE\PiezoJet\data\raw\gmtnet\data\jarvis_diele_piezo.pkl
- E:\CODE\PiezoJet\data\processed\splits.json

PiezoJet also has a persistent target PBC-graph cache at
E:\CODE\PiezoJet\data\processed\pbc_graph_cache. Do not pass those target
graphs to GaugeFlow's conditional generator: they are unavailable at sampling
time and would leak the target structure. They may be used as a later geometry
validation reference.

## Runtime

Run WSL Ubuntu-22.04 as user future04 in
/mnt/e/CODE/T2C-Flow/gaugeflow. The micromamba environment is flowmm-t2c.

Read E:\CODE\T2C-Flow\03_modify.md and the paper-side REDESIGN_PLAN.md before
altering the model contract.
