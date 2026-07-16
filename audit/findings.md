# Findings

> Independent review: this is the original Claude Code artifact. Its severity
> and execution-environment claims are qualified by `CODEX_VERIFICATION.md`.

Audit date: 2026-07-16
Code root: `E:\CODE\T2C-Flow\gaugeflow_perf_audit`

Severity levels: BLOCKER (prevents S1a training), CRITICAL (violates paper contract or risks invalid science), MAJOR (incomplete implementation or potential leakage), MINOR (documentation or hygiene issue), INFO (verified behavior worth recording).

---

## BLOCKER-1: No production training or reverse sampler exists for `HybridCrystalDenoiser`

**Status:** Open
**Severity:** BLOCKER

`HybridCrystalDenoiser` is implemented and tested, but the only training and sampling entry points are legacy.

- `scripts/train.py:199-254` instantiates `GaugeFlowVectorField` and trains with `RiemannianCrystalFlowMatcher().loss()`.
- `scripts/sample.py:47-68` loads `GaugeFlowVectorField` and calls `RiemannianCrystalFlowMatcher().sample()`.
- No script constructs `HybridCrystalDenoiser`, generates `shape_projector`/`fractional_to_cartesian`, or integrates the hybrid reverse process using `project_hybrid_reverse_state` (`src/gaugeflow/production/state_projection.py:35-53`).
- `src/gaugeflow/production/__init__.py:1-36` exports production primitives, but no caller script outside tests/audit scripts instantiates them for training or sampling.

**Impact:** S1a tensor-free training cannot start. The production architecture is mathematically qualified but has no executable training path.

**Required for closure:** a versioned production training script that (1) samples space-group blueprints, (2) builds `PointGroupMetricChart`, (3) runs hybrid diffusion loss using `AbsorbingMaskDiffusion`, `ScalableWrappedQuotient`, and `LatticeVolumeShape`, and (4) uses the same probability model in a production reverse sampler.

---

## BLOCKER-2: Symmetry-blueprint sampler is not implemented

**Status:** Open
**Severity:** BLOCKER

The paper requires a space-group blueprint sampling procedure that produces `shape_projector` and `fractional_to_cartesian` for the denoiser.

- `HybridCrystalDenoiser.forward` docstring states these are "determined by the sampled space-group blueprint" (`src/gaugeflow/production/equivariant_denoiser.py:188`).
- `PointGroupMetricChart.from_fractional_operations` and `SpaceGroupCompatibilityRouter.compatibility_record` provide the primitives (`src/gaugeflow/production/lattice_volume_shape.py:160-188`; `src/gaugeflow/production/space_group_router.py:98-116`), but no module samples a space group, builds the chart, and feeds the result to the denoiser.
- `reports/paper_architecture_compliance_v1.md:19` explicitly records "Complete Wyckoff autoregressive blueprint | Not implemented in this S0 change | S2 locked".

**Impact:** Without a blueprint sampler, training batches cannot be constructed with the correct `shape_projector` and `fractional_to_cartesian`. S1a cannot start.

**Required for closure:** a versioned blueprint sampler that selects space groups (possibly Wyckoff-autoregressive), emits `shape_projector` and `fractional_to_cartesian`, and is covered by contract tests.

---

## CRITICAL-1: S0.4 Cartesian atlas failed the frozen CUDA latency limit

**Status:** Closed (documented)
**Severity:** CRITICAL

The production atlas passed all scientific checks but failed the pre-registered performance threshold.

- `README.md:41-66` states S0.4-v1 failed the frozen RTX 4060 Ti latency limit `41.89 ms > 20 ms`; decision is `failed_no_advance`.
- `configs/paper_s0_4_cartesian_atlas_prior_v1.json:3-9` records `status: completed_failed_no_advance` and `failed_checks: ["cuda_latency"]`.
- Manual measurement on this RTX 4060 Ti (2026-07-16): generic-stratum manual pool `0.60 ms/forward`, axial-stratum `1.10 ms/forward` in FP64 on CUDA for a single graph. The official `41.89 ms` figure reflects the full production forward path under the frozen protocol.

**Impact:** The atlas is scientifically correct but too slow for the declared production latency budget. Training would amplify this cost.

**Required for closure:** either a latency repair that passes the frozen threshold or an explicit protocol amendment raising the threshold.

---

## MAJOR-1: Potential data-leakage fields persist on the batch object

**Status:** Open
**Severity:** MAJOR

The dataset emits fields that are not model inputs but remain accessible to any caller.

- `PiezoCrystalDataset.__getitem__` returns `Data` objects containing `material_id`, `niggli_transform`, `response_stratum`, and `zero_response` (`src/gaugeflow/data.py:217-228`, `239-250`).
- `HybridCrystalDenoiser.forward` does not accept these fields, and `s0_audit.py:20-28` forbids target metadata in the signature.
- However, `material_id` is a source identifier and `niggli_transform` is target-derived. They are not listed in `FORBIDDEN_SIGNATURE_FIELDS` because the model never sees them, but a future training loop could accidentally use them.

**Impact:** Low immediate risk because the model contract is enforced, but a latent leakage vector if the training script is extended.

**Recommended for closure:** strip or explicitly quarantine `material_id` and `niggli_transform` from training batches, or document why their presence on the batch object is safe.

---

## MAJOR-2: Production and legacy probability paths coexist without a deprecation boundary

**Status:** Open
**Severity:** MAJOR

- `src/gaugeflow/production/__init__.py:1-36` exports the new production primitives.
- `src/gaugeflow/model.py:788-1063` still contains the legacy `GaugeFlowVectorField` and `QuotientRolloutFlowMap`.
- `scripts/train.py` and `scripts/sample.py` remain wired to the legacy path.
- `s0_audit.py:127-131` verifies that production modules do not import legacy probability-path symbols, but the reverse is not enforced: legacy scripts do not import production modules.

**Impact:** A user running `train.py` will silently train the old architecture, not the paper-declared production architecture.

**Recommended for closure:** add a runtime guard or explicit deprecation warning to `train.py`/`sample.py`, or replace them with production entry points.

---

## MINOR-1: ruff and mypy were not installed in the active environment

**Status:** Closed (resolved during audit)
**Severity:** MINOR

- Initial audit found `ruff` and `mypy` missing from `D:\Anaconda\envs\EGNN`.
- After `pip install ruff mypy` (2026-07-16), `ruff check src tests scripts` and `mypy src/gaugeflow/production` both pass.

**Impact:** None after installation; static checks are now runnable.

---

## MINOR-2: TorchScript deprecation and instance-annotation warnings pollute logs

**Status:** Open
**Severity:** MINOR

- Pytest emits 66 `torch.jit.script` deprecation warnings and 43 `TorchScript type system doesn't support instance-level annotations` warnings (`tests` collection, 2026-07-16).
- These do not affect correctness but obscure test output and audit evidence.

**Recommended for closure:** add a pytest warning filter or migrate off `torch.jit.script`.

---

## INFO-1: Coordinate-head gradient connectivity verified

**Status:** Closed
**Severity:** INFO

An earlier audit concern asked whether the coordinate head receives condition signal.

- Finite-difference check (2026-07-16): `d(score^2.sum)/d(tensor_condition[0,0]) = -7.45e-05` (finite difference) vs `-4.69e-05` (analytic). Both non-zero, same sign, same order of magnitude.
- Condition dropout sensitivity: `||score(cond_present=1) - score(cond_present=0)|| = 0.539`.
- Gradient norm on `tensor_condition` is `0.364`.

**Conclusion:** the coordinate head is connected to the condition pathway.

---

## INFO-2: Atlas precision under FP32/BF16 autocast

**Status:** Closed
**Severity:** INFO

- FP32 vs FP64 pooled relative error: `1.38e-07`.
- BF16 autocast vs FP64 pooled relative error: `4.63e-03`.
- FP32 posterior max absolute difference: `5.89e-08`.
- BF16 posterior max absolute difference: `1.41e-03`.
- Unique candidate count is stable at 4032 across FP64/FP32/BF16.

**Conclusion:** BF16 introduces measurable but bounded error; FP32 is essentially exact.

---

## INFO-3: Wrapped-quotient exact vs scalable agreement

**Status:** Closed
**Severity:** INFO

Existing tests already enforce tight agreement.

- `tests/test_paper_s0_2_scalability_symmetry.py:111-129` asserts `log_error <= 1e-6` and relative score error `<= 1e-4` for `M=2,3,4`.
- `tests/test_paper_s0_2_scalability_symmetry.py:132-156` passes the 20-site CUDA stress test (35.14 s on this RTX 4060 Ti).

**Conclusion:** the scalable QMC path is numerically faithful to the exact oracle within declared tolerances.

---

## Summary by severity

- **BLOCKER (2):** missing production training/reverse sampler; missing blueprint sampler.
- **CRITICAL (1):** S0.4 latency failure (documented, no advance).
- **MAJOR (2):** latent leakage fields on batch object; legacy/production coexistence.
- **MINOR (2):** missing ruff/mypy (resolved); warning pollution.
- **INFO (3):** coordinate-head connectivity, atlas precision, wrapped-quotient agreement verified.
