# Test Coverage and Gaps

> Provenance warning: the original execution used a non-authoritative Windows
> environment with warnings hidden. WSL/CUDA results and corrected stage
> attribution are recorded in `CODEX_VERIFICATION.md`.

Audit date: 2026-07-16
Code root: `E:\CODE\T2C-Flow\gaugeflow_perf_audit`

## Test execution summary

- **Full suite:** `198 passed in 59.96s` (`python -W ignore -m pytest tests -q --tb=short`)
- **Targeted S0 tests:** `48 passed in 46.52s` (`tests/test_paper_s0_production.py`, `tests/test_paper_s0_2_scalability_symmetry.py`, `tests/test_paper_s0_4_cartesian_atlas_prior.py`, `tests/test_cartesian_gauge_atlas.py`)
- **BF16 CUDA production path:** `1 passed in 4.94s` (`test_bf16_autocast_production_path_is_finite_and_uses_4032_candidates`)
- **20-site CUDA wrapped quotient:** `1 passed in 35.14s` (`test_scalable_wrapped_quotient_handles_twenty_sites_and_triclinic_metrics`)
- **ruff:** `All checks passed!` (after `pip install ruff` on 2026-07-16)
- **mypy:** `Success: no issues found in 14 source files` (after `pip install mypy` on 2026-07-16)

## Coverage by audit section

### A. Element categorical diffusion — COVERED

- `tests/test_paper_s0_production.py:57-61` verifies decode of MASK token 118 raises `ValueError`.
- `tests/test_paper_s0_production.py` covers corruption, reverse probabilities, and vocabulary bounds.
- `src/gaugeflow/production/categorical_mask.py:20-112` is fully exercised by these tests.

**Gap:** no dedicated test for `reverse_probabilities` under exact time endpoints (`time_to == time_from`) or under `torch.float64` vs `torch.float32` schedule drift.

### B. Coordinate / translation quotient — COVERED

- `tests/test_paper_s0_2_scalability_symmetry.py:111-129` compares exact small-site oracle vs scalable QMC for `M=2,3,4` with `log_error <= 1e-6` and relative score error `<= 1e-4`.
- `tests/test_paper_s0_2_scalability_symmetry.py:132-156` covers 20-site CUDA stress with two triclinic metrics and two sigmas.
- `tests/test_quotient_paths.py`, `test_coordinate_geometry_v1.py`, and `test_d07_multiscale_semigroup.py` cover related quotient geometry.

**Gap:** no test for `AdaptiveWrappedQuotient` resource exhaustion (`max_images` fail-closed path) or for `ScalableWrappedQuotient` under BF16 autocast.

### C. Lattice chart — COVERED

- `tests/test_paper_s0_2_scalability_symmetry.py` covers all 230 space-group charts, Reynolds ranks, and full-denoiser translation/basis equivariance.
- `tests/test_paper_s0_production.py` covers log-volume/log-shape round-trip and symmetry projection.
- `src/gaugeflow/production/lattice_volume_shape.py:12-227` is exercised.

**Gap:** no explicit test for determinant normalization edge cases (near-singular `C^T exp(A) C`) or for `PointGroupMetricChart` under degenerate point groups (e.g. triclinic with identity only).

### D. SO(3)/O(3) gauge and router — COVERED

- `tests/test_paper_s0_2_scalability_symmetry.py:75-108` checks all 230 space groups for closure and correct piezoelectric ranks.
- `tests/test_parity.py` and `test_stabilizer.py` cover parity and stabilizer behavior.
- `src/gaugeflow/production/space_group_router.py:33-58` is exercised.

**Gap:** no test for `reynolds_irrep_matrix` under float32 e3nn builds (the SVD correction is exercised under float64; float32 behavior is implicitly covered but not explicitly asserted).

### E. Cartesian gauge atlas — COVERED

- `tests/test_cartesian_gauge_atlas.py:216-224` asserts generic raw count `24*7*24 = 4032`.
- `tests/test_cartesian_gauge_atlas.py:253-274` verifies soft stratum partition continuity and finite gradients.
- `tests/test_paper_s0_4_cartesian_atlas_prior.py:24-33` verifies duplicate-expansion alignment, descriptor-frame ambiguity, and K=8/16/32/64 axial refinement.
- `tests/test_paper_s0_4_cartesian_atlas_prior.py:35-41` verifies axial refinement monotonicity and synthetic coverage `maximum_nearest_geodesic <= 0.40`.
- `tests/test_paper_s0_4_cartesian_atlas_prior.py:44-52` verifies BF16 autocast finiteness and 4032-candidate stability.

**Gap:** no test for the descriptor-isotropic fallback path with exactly isotropic covariance under BF16; no test for `gate` behavior at `time=0` or `time=1` boundaries.

### F. Equivariant denoiser — COVERED

- `tests/test_paper_s0_production.py:311-317` enforces no target metadata in `HybridCrystalDenoiser.forward` signature.
- `tests/test_cartesian_gauge_atlas.py` exercises the denoiser with identity `shape_projector` and `fractional_to_cartesian`.
- `tests/test_paper_s0_2_scalability_symmetry.py` checks translation and unimodular-basis equivariance.
- Manual finite-difference check (2026-07-16) confirmed coordinate-head gradient connectivity.

**Gap:** no test for `HybridCrystalDenoiser` under a non-identity `shape_projector` (all current tests use `torch.eye(6)`); no test for `project_lattice_state` inside the denoiser under a symmetry-reduced subspace.

### G. Blueprint sampler — NOT COVERED

No tests exist because the sampler does not exist. `reports/paper_architecture_compliance_v1.md:19` explicitly states it is not implemented.

**Required tests when implemented:**
- Space-group sampling distribution is correct and reproducible.
- `shape_projector` is idempotent and commutes with point-group action.
- `fractional_to_cartesian` is consistent with the sampled space group and reconstructs the correct metric.
- End-to-end denoiser call with blueprint-generated inputs matches contract.

### H. Training and reverse sampler — NOT COVERED

`scripts/train.py` and `scripts/sample.py` use the legacy `GaugeFlowVectorField`/`RiemannianCrystalFlowMatcher` path (`scripts/train.py:199-254`; `scripts/sample.py:47-68`). No production training or sampling tests exist.

**Required tests when implemented:**
- Production training step is finite and decreases loss on a fixed tiny panel.
- Production reverse sampler produces states that satisfy `project_hybrid_reverse_state` invariants.
- Training and sampling use the same probability model (no train/sample mismatch).
- Categorical decode never emits MASK.

### I. Data leakage — PARTIALLY COVERED

- `tests/test_paper_s0_production.py:311-317` enforces no target metadata in the model signature.
- `src/gaugeflow/data.py:89-96` documents that target-CIF stabilizers are not emitted.

**Gap:** no test asserts that `material_id`, `niggli_transform`, `response_stratum`, or `zero_response` are absent from the model input batch when using the production path. The current model contract makes this moot for `HybridCrystalDenoiser`, but a future production training script should add an explicit batch-filter test.

### J. Numerical / performance — PARTIALLY COVERED

- `tests/test_paper_s0_4_cartesian_atlas_prior.py:44-52` covers BF16 finiteness and candidate-count stability.
- `tests/test_paper_s0_2_scalability_symmetry.py:132-156` covers 20-site CUDA scalability.
- `README.md:41-66` documents the S0.4 latency failure (`41.89 ms > 20 ms`).

**Gap:** no pre-registered performance regression test that fails CI if atlas latency exceeds threshold; no memory-footprint test for the 4032-candidate atlas under large batch; no BF16 numerical-stability test for the full denoiser forward/backward.

## Static checks

- `ruff check src tests scripts` → all checks passed (2026-07-16).
- `mypy src/gaugeflow/production` → no issues in 14 source files (2026-07-16).

## Warning hygiene

Pytest emits:
- 66 `torch.jit.script` deprecation warnings
- 43 TorchScript instance-annotation warnings

These do not affect test outcomes but reduce signal-to-noise in audit logs.

## Overall coverage verdict

The production mathematical primitives are well covered for S0/S0.2/S0.4. The critical gaps are the missing blueprint sampler and the missing production training/reverse sampler; both are entirely untested because they are unimplemented.
