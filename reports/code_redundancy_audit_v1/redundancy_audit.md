# GaugeFlow code redundancy audit v1

Date: 2026-07-16

Scope: the retained production architecture, current S0/data entry points and
a full scan of every remaining Python source file.

Scientific boundary: repository cleanup only. No model equation, physical
definition, frozen threshold, completed result or advancement decision was
changed.

## Result

Both static scans are clean:

- active production surface: 164 class/function definitions;
- all retained source and scripts: 310 class/function definitions;
- duplicate normalized bodies: none;
- unreferenced private definitions: none;
- stored-but-unread instance attributes: none;
- lexically unreachable statements: none;
- constant-boolean branches: none;
- declared-but-unused CLI arguments: none.

The regression in `tests/test_code_redundancy_audit.py` prevents later changes
from silently restoring duplicate or fallback branches on the production
surface.

## Cleanup performed

1. Tagged and pushed the complete pre-cleanup state as
   `archive/pre-production-cleanup-20260716` at commit
   `0dbcfbabd997b3e32a18ed391e28adb1fe4f3ffc`.
2. Removed superseded Gate A--A11, P5-D0/C0, substrate-v2 and vNext Q0/Q1
   code, configs, tests, reports and run payloads from the active tree.
3. Removed the legacy trainer/sampler and continuous-logit/ODE runtime modules.
   No compatibility dispatch to those APIs remains.
4. Removed old profiling traces, frozen exploratory caches, review dossiers and
   protocol-local utilities that no retained production path consumes.
5. Consolidated file and canonical-JSON hashing in `gaugeflow.file_utils`, and
   response-stratum classification in `gaugeflow.data`.
6. Retained the complete direct-CG implementation as the final matched
   baseline and retained the harmonic implementation only under
   `production/archive_harmonic` for paper diagnostics.
7. Expanded Ruff from a selected subset to every remaining source, script and
   test, then corrected import order, dead references and formatting findings.

The scientific lessons from retired protocols are condensed in
`docs/research_iteration_history.md`. Exact historical source and evidence are
available through the archive tag rather than through active runtime branches.

## Deliberately retained distinctions

- The optimized generic Cartesian-atlas contraction and scalar/reference
  contraction implement the same mathematics at different execution
  granularities. The former is the qualified 4,032-candidate runtime path; the
  latter is a numerical oracle, not a fallback.
- Proper-SO(3) tensor-orbit operations and full-O(3) crystal compatibility
  remain separate physical operations.
- A physical zero tensor remains distinct from a missing condition.

## Protocol preservation

- S0.3-v1 remains frozen as failed.
- S0.4-v1 remains `failed_no_advance` because its archived CUDA latency was
  41.89 ms against the frozen 20 ms limit.
- S0.4.1 remains a separately versioned runtime qualification; cleanup neither
  overwrites that report nor authorizes S1a.
- No training, tensor fine-tuning, oracle promotion, relaxation, DFT or DFPT
  was started.

## Validation

Authoritative environment: WSL Ubuntu-22.04,
`/home/future04/micromamba/envs/flowmm-t2c/bin/python`, torch 2.5.1+cu124,
CUDA 12.4, NVIDIA GeForce RTX 4060 Ti.

- tests: 77 passed (60 existing third-party/TorchScript warnings);
- configured Ruff: passed over all retained source, scripts and tests;
- production mypy: no issues in 14 source files;
- active and full redundancy scans: all finding categories empty;
- no-write CUDA smoke: 4,032 raw/unique candidates, aligned relative error
  `2.17e-15`, sorted-posterior L1 `4.80e-16`, prior L1 `0`, atlas latency
  `9.60 ms/forward`, peak memory `15.19 MB`, all outputs finite.

The smoke is a current regression check, not a rewrite of frozen S0 evidence.
