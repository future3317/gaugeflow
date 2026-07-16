# GaugeFlow code redundancy audit v1

Date: 2026-07-16

Scope: active revised-paper production modules, executable entry points, and a repository-wide classification pass.

Scientific boundary: static cleanup only; no Gate threshold, frozen result, model claim, tensor experiment, or training protocol was changed.

## Result

The active production surface is clean under the versioned AST audit in
`scripts/audit_code_redundancy.py`. It contains 140 class/function definitions
and reports no:

- duplicate normalized bodies;
- unreferenced private definitions;
- stored-but-unread instance attributes;
- lexically unreachable statements;
- constant-boolean branches; or
- declared-but-unused CLI arguments.

The audit is enforced by `tests/test_code_redundancy_audit.py` so later changes
cannot silently restore an active duplicate/fallback branch.

## Remediation performed

1. Removed stored-but-unread attributes from the dataset, geometry RBF, legacy
   vector-field/flow-map, and substrate-v2 scorer implementations. Constructor
   arguments that still determine layer shapes or validation remain intact.
2. Deleted the unreachable Q0 implementation body that sat behind a `run()`
   function which always raises. The original protocol remains frozen at Git
   commit `42a34c5`; the current entry point remains explicitly fail-closed and
   points to Q0.1. This removes a compatibility fallback without rewriting any
   frozen Q0/Q0.1 evidence.
3. Removed unused imports and local assignments where this does not alter the
   frozen vNext source manifest (S0.2, substrate-v2, legacy model, and tests).
4. Consolidated graphwise mean projection into
   `gaugeflow.production.state_projection.graph_mean`; the production denoiser
   now uses that single implementation.
5. Removed audit-only candidate/score convenience methods from the production
   Cartesian atlas. Tests and audit runners now exercise the actual discrete
   measure and contraction rather than a parallel production API.
6. Removed runtime method-identity introspection that existed only to support a
   monkeypatched test. Duplicate-expansion tests now alter the candidate measure
   explicitly.
7. Replaced a theoretically impossible empty directional partition with an
   explicit `RuntimeError`; a regression test verifies that it cannot become a
   silent zero-condition fallback.

## Repository-wide classification

The all-source scan finds no unreachable statements, constant-boolean branches,
or unused CLI arguments. Its remaining duplicate bodies are intentionally
concentrated in frozen, standalone research protocols: file hashing, path
resolution, CUDA seeding/synchronization, and small protocol-local diagnostics.
It also identifies two unused A4 private helpers (`_endpoint_ids` and
`_categorical_sample`). They were verified as dead but deliberately retained:
removing them changes the protected vNext legacy manifest. Git history alone is
not substituted for an executable frozen-source integrity contract. These
archived findings cannot enter the active runtime and are excluded from the
active-clean assertion.

The same rule applies to unused imports/local variables in manifest-covered
historical runners. They remain byte-for-byte unchanged because centralizing or
formatting them would couple archived protocols to mutable shared code and
invalidate the source hashes used to reproduce historical evidence.

Two apparent stored-but-unread attributes are not dead code:

- `PiezoCrystalDataset.preprocessed_manifest` is public run provenance read by
  `scripts/train.py` and `scripts/benchmark_performance.py`.
- `_GeometryMessageBlock.vector_channels` is read by its owning scorer through
  `self.blocks[0].vector_channels` to allocate equivariant vector state.

The specialized rank-three `einsum` paths in the Cartesian atlas are also
retained. The grouped-batch kernel and scalar/reference contractions implement
the same mathematics at different execution granularities; the former is the
qualified performance path, not an accidental duplicate operator.

## Protocol preservation

- S0.3-v1 remains frozen as failed.
- S0.4-v1 remains `failed_no_advance` because its archived CUDA latency was
  41.89 ms against the frozen 20 ms limit.
- S0.4.1 remains a separately versioned runtime qualification; cleanup does not
  overwrite S0.4-v1 or authorize S1a.
- No S1a/S2 training, tensor conditioning, oracle, relaxation, DFT, or DFPT was
  started.

## Validation commands

The authoritative validation environment is WSL Ubuntu-22.04 with
`/home/future04/micromamba/envs/flowmm-t2c/bin/python`:

- runtime: torch 2.5.1+cu124, CUDA 12.4, NVIDIA GeForce RTX 4060 Ti;
- full suite: 204 passed, 102 pre-existing third-party/TorchScript warnings;
- configured Ruff: passed;
- production mypy: no issues in 14 source files;
- active redundancy audit: all six finding categories empty;
- frozen vNext legacy manifest verification: unchanged;
- no-write CUDA smoke: 4,032 raw/unique candidates, aligned relative error
  `2.17e-15`, sorted-posterior L1 `4.80e-16`, prior L1 `0`, atlas latency
  `13.09 ms/forward`, and peak memory `15.19 MB`.

The smoke measurement is a regression check, not a replacement or rewrite of
the official S0.4.1 report. The pushed commit SHA is recorded in the handoff.
