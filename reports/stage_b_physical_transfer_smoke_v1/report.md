# Stage-B physical-transfer data and CUDA smoke

This is a bounded software qualification, not a scientific Stage-B training
result. It authorizes protocol preparation only.

## Data closure

- Six immutable MatPES 2025.2 artifacts were pooled: PBE and r2SCAN
  train/valid/test.
- Raw rows: 433,189 PBE and 386,544 r2SCAN.
- The shared GaugeFlow-base domain excludes 69,867 structures with more than
  20 atoms, leaving 749,866 functional rows and 387,697 unique `matpes_id`s.
- ID-grouped split: 674,709 train, 37,054 calibration, 38,103 test.
- Invalid selected rows: zero.
- IDs are used only for split grouping and are absent from model batches.
- The explicit common target is cohesive energy per atom; force and stress are
  complete. Partial formation-energy fields are not used as fallbacks.

The qualified index manifest SHA256 is
`fc2474d6fc4be356b852d8657b7b10a52b7a566268d0ecaf33324710a3b04a5e`.
The train-only covariance-preserving normalizer SHA256 is
`6935ea4f5d9963a4fa2179b44539b6fa38657f5f780b466ab7c18b38c46f5cca`.

## Implementation closure

- Generation and physical transfer use one periodic message-passing encoder.
- The clean physical interface returns scalar/vector Cartesian features before
  generation terminal heads, so it does not evaluate the coordinate carrier or
  composition/assignment/lattice heads.
- PBE/r2SCAN identity enters only through a graph-level functional embedding at
  the physical readout. The shared geometry representation never sees material
  IDs.
- Energy, vector force, symmetric Kelvin stress and optional teacher-feature
  losses use explicit masks and graph-equal reduction.
- MatPES and Alex replay losses are accumulated before one optimizer step; two
  optimizers never compete over the same backbone.
- Streaming normalization was changed from scalar per-record Torch updates to
  batched `bincount` reductions. The sufficient statistics are identical in the
  unit test, while the full scan completed in roughly six minutes.

## CUDA preflight findings

The fail-closed preflight found and corrected three execution defects before a
scientific run:

1. deterministic CUDA required `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Torch
   initialization;
2. the new clean lattice decomposition initially inherited BF16 autocast and
   attempted a BF16 matrix inverse;
3. globally enabling TF32 changed 3x3 metric reconstruction residuals from
   approximately (10^{-7}) to (10^{-4})--(10^{-3}). Stage-B therefore keeps
   graph-level lattice/log-SPD geometry in IEEE FP32 while learned matmuls use
   BF16 tensor cores.

The fixes do not relax the lattice consistency check and do not add a precision
fallback.

## CUDA result

The final smoke used the 34,284,207-parameter A1 EMA checkpoint
`7c8fb7afc3aee6d4723d700b59f2a0523da25e897a46de8e9d2c7e5db824b6da`
on two real MatPES graphs:

| check | result |
|---|---:|
| finite losses and gradients | pass |
| physical loss, step 1 -> 2 | 1.00253 -> 0.52080 |
| Alex replay loss, step 1 -> 2 | 5.61466 -> 3.97025 |
| exact-resume metric max error | 0 |
| exact-resume parameter max error | 0 |
| peak CUDA allocation | 1,557,303,296 bytes |

The two-step decrease is an overfit/gradient sanity check, not a validation
claim. Formal Stage-B thresholds, exposure, replay ratio, teacher-feature
source and two-GPU protocol remain to be frozen before training.
