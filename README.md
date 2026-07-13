# GaugeFlow

GaugeFlow is the standalone implementation of stabilizer-aware,
tensor-orbit-conditioned crystal generation. It does not import FlowMM at
runtime. FlowMM remains in ../flowmm solely as a historical baseline.

The core condition is a rank-three piezoelectric tensor orbit. GaugeFlow
constructs a finite SO(3) orbit set, uses one graph-level latent alignment
distribution, and injects the complete vector response field
F_e(n) = e:(n outer n) into periodic geometric messages.

Before batching, the data path performs a tracked Niggli reduction: lattice
rows change by an integer unimodular basis transform and fractional coordinates
by its inverse, while the tensor stays in its Cartesian physical frame. The
model pools only proper (determinant +1) crystal stabilizer rotations; improper
parity operations remain distinct.

## Layout

- src/gaugeflow/tensor.py: tensor conventions, orbit samples, response fields.
- src/gaugeflow/manifold.py: standalone product crystal flow coordinates.
- src/gaugeflow/model.py: orbit-response encoder and graph vector field.
- src/gaugeflow/data.py: direct CSV/CIF reader using PyG Data/Batch, independent of FlowMM.
- src/gaugeflow/unit_cell.py: strict Niggli reduction with tracked basis changes.
- src/gaugeflow/stabilizer.py: proper crystallographic stabilizer extraction.
- src/gaugeflow/flow.py: conditional flow-matching objective and sampler.
- scripts/: training and tensor-orbit sampling entry points.

## Status contract

The new package is the active GaugeFlow path. QR canonicalization, raw
component conditioning, and FlowMM are baselines, not fallbacks. The prepared
PiezoJet/GMTNet CSV may be read as an input dataset, but no FlowMM Python
module is imported.
