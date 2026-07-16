# Independent verification of the Claude Code audit

Date: 2026-07-16

Repository: `E:/CODE/T2C-Flow/gaugeflow_perf_audit`
Scope: verify the five Claude Code reports against the current source, the
revised-paper stage contract, the frozen S0.4 record, and the required WSL CUDA
environment. This review does not reopen S0.3/S0.4 or authorize training.

## Overall verdict

The high-level conclusion **NOT READY to execute S1a training** is correct, but the supporting
classification is only partially correct.

The immutable S0.4-v1 decision remains `failed_no_advance`, but the separately
versioned S0.4.1 performance successor has now passed the unchanged runtime
threshold. The remaining direct S1a blocker is that the repository has no
revised-paper production training loop or hybrid reverse sampler.

The claim that a complete symmetry/Wyckoff blueprint sampler is itself an S1a
blocker is not consistent with the repository's current versioned stage
contract. `reports/paper_architecture_compliance_v1.md` assigns the complete
Wyckoff-autoregressive blueprint to S2, while README and the manuscript define
S1a as a tensor-free reverse-sampler qualification. A complete blueprint is a
real missing component of the eventual hierarchical generator and an S2
blocker, not a reason to conflate S1a with S2. An S1a implementation may use an
unconstrained identity shape projector or protocol-supplied structural charts;
it must not use paired target metadata as a hidden condition.

## Finding-by-finding decision

| Claude finding | Verification | Action |
|---|---|---|
| BLOCKER-1: no production training/reverse sampler | **Confirmed.** `scripts/train.py` and `scripts/sample.py` instantiate the historical `GaugeFlowVectorField` and `RiemannianCrystalFlowMatcher`; no executable revised-paper hybrid trainer/sampler exists. | The legacy entry points now fail closed unless `--acknowledge-legacy-prototype` is supplied. Production S1a training/sampling remains unimplemented. |
| BLOCKER-2: no blueprint sampler blocks S1a | **Partly true, wrong stage.** The complete blueprint is missing, but the project contract records it as S2-locked. `shape_projector` and `fractional_to_cartesian` inputs do not logically require a learned blueprint sampler: they can be obtained from an unconstrained chart or a protocol-supplied chart for S1a. | Reclassify as **future S2/final-architecture blocker**. |
| CRITICAL-1: S0.4 latency failure | **Confirmed and closed by a successor.** Official immutable evidence records `41.89 ms > 20 ms` and `failed_no_advance`; S0.4.1 separately passes at `14.62 ms` without changing S0.4. | Preserve both versioned outcomes. Raising the frozen threshold in place remains forbidden. |
| MAJOR-1: metadata fields on `Data` are leakage | **No actual leakage demonstrated.** IDs, Niggli transforms, strata, and zero labels are required for provenance/audits but are absent from `HybridCrystalDenoiser.forward`. Merely carrying labels in a dataset object is not model leakage. | Add their exact names to the executable forbidden-signature contract and regression test. Preserve them in the dataset; future production code must construct model kwargs from an explicit allowlist. |
| MAJOR-2: legacy and production paths coexist silently | **Confirmed.** A user could previously run the historical scripts believing they were revised-paper runtime. | Closed for accidental use by explicit fail-closed acknowledgement guards. Historical reproduction remains possible and is visibly named. |
| MINOR-1: ruff/mypy unavailable | **Not a repository defect.** They were missing only from Claude's non-authoritative Windows environment and installed there ad hoc. | Rechecked in the pinned WSL environment; both pass. |
| MINOR-2: TorchScript warnings | **Real but non-blocking.** The audit hid all warnings with `-W ignore`; most observed warnings originate in e3nn/TorchScript, spglib, monty, or pandas. | Do not globally suppress them. Record them honestly; migration can be separately scoped. |
| INFO-1: coordinate condition gradient | **Plausible diagnostic, not formal gate evidence.** It was an ad hoc Windows-environment calculation without a committed protocol. | Retain only as supporting diagnostic; do not promote it to qualification evidence. |
| INFO-2: FP32/BF16 atlas precision | **Directionally consistent with official S0.4 evidence, but Claude's manual calculation is not the official run.** | Use the immutable S0.4 metrics as the scientific record. |
| INFO-3: wrapped quotient agreement | **Confirmed by committed tests and prior S0.2 evidence.** | No code change required. |

## Environment re-verification

Claude's commands report the Windows environment `D:/Anaconda/envs/EGNN` and
use `python -W ignore`. This violates `AGENTS.md`, which requires WSL 2,
Ubuntu-22.04, and
`/home/future04/micromamba/envs/flowmm-t2c/bin/python`. Consequently those
command outputs are not formal project evidence, even when their conclusions
match later verification.

Independent commands were run without warning suppression in the required
environment:

```text
Python: /home/future04/micromamba/envs/flowmm-t2c/bin/python
Torch: 2.5.1+cu124
CUDA: 12.4, available=True
GPU: NVIDIA GeForce RTX 4060 Ti
pytest: 198 passed, 102 warnings in 63.71 s (before the two new boundary tests)
ruff: All checks passed
mypy src/gaugeflow/production: Success, no issues in 14 files
targeted post-fix tests: 3 passed
```

The 102 visible warnings comprise third-party pandas/pyarrow notices,
TorchScript annotation warnings, and spglib deprecations. They did not change
test outcomes, but they must not be erased from audit logs.

## Changes executed by this verification

1. `scripts/train.py` and `scripts/sample.py` are explicitly documented as
   archived continuous-flow prototype entry points and refuse to run without
   `--acknowledge-legacy-prototype`.
2. `material_id`, `niggli_transform`, `response_stratum`, and `zero_response`
   are included in the production forbidden-signature contract.
3. Regression tests verify both fail-closed entry points and the expanded
   metadata quarantine.

No production training loop, reverse sampler, Wyckoff blueprint, tensor
training, oracle, relaxation, DFT, or DFPT was started. No frozen S0.4 threshold
or result was changed.

## Correct next gate after S0.4.1

S0.4.1 passed the performance-only remediation while preserving the weighted
4,032-candidate Cartesian prior and every S0.4 scientific semantic. The next
permitted work is to pre-register and implement an S1a production hybrid
reverse-sampler/training qualification; it has not started. The complete
learned symmetry blueprint remains a later S2 deliverable under the current
stage contract.
