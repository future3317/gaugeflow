# Training Readiness

> Independent review: the overall no-advance verdict is retained, but the
> blueprint finding is an S2/final-architecture gap rather than an S1a blocker.
> See `CODEX_VERIFICATION.md`.

Audit date: 2026-07-16
Code root: `E:\CODE\T2C-Flow\gaugeflow_perf_audit`

Question: is the codebase ready to begin S1a tensor-free training?

## Four-way verdict

| Criterion | Verdict | Evidence |
|---|---|---|
| 1. Production architecture implemented and mathematically qualified | PASS | `HybridCrystalDenoiser` and all production primitives are implemented, tested, and pass S0/S0.2/S0.4 scientific checks (`src/gaugeflow/production/equivariant_denoiser.py:135-277`; `README.md:18-39`) |
| 2. Production training path exists and uses the same probability model | FAIL | `scripts/train.py:199-254` uses legacy `GaugeFlowVectorField`/`RiemannianCrystalFlowMatcher`; no production training script exists |
| 3. Symmetry-blueprint sampler exists to generate `shape_projector` and `fractional_to_cartesian` | FAIL | No sampler found; `reports/paper_architecture_compliance_v1.md:19` states it is not implemented |
| 4. Performance/latency budget satisfied for the atlas | FAIL | S0.4-v1 failed the frozen RTX 4060 Ti latency limit `41.89 ms > 20 ms` (`README.md:41-66`; `configs/paper_s0_4_cartesian_atlas_prior_v1.json:3-9`) |

**Overall readiness: NOT READY for S1a tensor-free training.**

## Detailed assessment

### What is ready

- **Element categorical diffusion** (`src/gaugeflow/production/categorical_mask.py:20-112`) is complete and tested.
- **Translation-quotient wrapped kernel** (`src/gaugeflow/production/wrapped_coordinates.py:41-184`, `362-475`) is complete, with exact oracle and scalable QMC paths verified.
- **Lattice chart** (`src/gaugeflow/production/lattice_volume_shape.py:12-227`) is complete, including point-group projection and metric reconstruction.
- **SO(3)/O(3) router** (`src/gaugeflow/production/space_group_router.py:33-116`) is complete for all 230 space groups.
- **Cartesian gauge atlas** (`src/gaugeflow/production/cartesian_gauge_atlas.py:320-639`) is complete and scientifically correct.
- **Equivariant denoiser** (`src/gaugeflow/production/equivariant_denoiser.py:135-277`) is complete, with time/condition FiLM in every block and no target-metadata inputs.
- **Static checks** pass: `ruff check src tests scripts` and `mypy src/gaugeflow/production` (2026-07-16).
- **Full test suite** passes: `198 passed in 59.96s`.

### What blocks training

1. **No production training script.** The existing `train.py` trains a legacy model with a legacy objective. It does not construct `HybridCrystalDenoiser`, does not use `AbsorbingMaskDiffusion`, `ScalableWrappedQuotient`, or `LatticeVolumeShape`, and does not project reverse states through `project_hybrid_reverse_state`.

2. **No blueprint sampler.** `HybridCrystalDenoiser.forward` requires `shape_projector` and `fractional_to_cartesian` as inputs (`src/gaugeflow/production/equivariant_denoiser.py:183-185`). These must come from a space-group blueprint. The primitives exist (`PointGroupMetricChart`, `SpaceGroupCompatibilityRouter`), but no sampler wires them together.

3. **S0.4 latency failure.** The Cartesian atlas is too slow for the declared production budget. Training would run this atlas in every forward pass, so the latency problem compounds across steps.

4. **Potential leakage vector on batch object.** `PiezoCrystalDataset` emits `material_id`, `niggli_transform`, `response_stratum`, and `zero_response` on the `Data` object (`src/gaugeflow/data.py:217-228`, `239-250`). These are not model inputs today, but a production training script must explicitly exclude them.

### What must be built before S1a

1. **Blueprint sampler module** (new): samples space groups, builds `PointGroupMetricChart`, emits `shape_projector` and `fractional_to_cartesian`, and is covered by contract tests.
2. **Production training script** (new): constructs `HybridCrystalDenoiser`, generates hybrid diffusion targets using `AbsorbingMaskDiffusion`, `ScalableWrappedQuotient`, and `LatticeVolumeShape`, and optimizes the model.
3. **Production reverse sampler** (new): integrates the hybrid reverse process using the same probability model, projecting each step through `project_hybrid_reverse_state`.
4. **Latency repair or protocol amendment** for the Cartesian atlas.
5. **Batch hygiene guard** that strips `material_id` and `niggli_transform` from training inputs or documents why they are safe.

### Recommended S1a entry criteria

S1a should not start until all of the following hold:

- A versioned production training script passes a fixed tiny-panel overfit test.
- A versioned production reverse sampler produces finite states satisfying projection invariants.
- The blueprint sampler passes contract tests for projector idempotence and metric consistency.
- The atlas latency either meets the frozen `20 ms` threshold or the protocol is explicitly amended.
- A leakage audit of the production training batch confirms no target metadata is present.

## Final readiness statement

The codebase contains a complete, tested, and statically clean implementation of the paper's mathematical architecture. However, it is **not ready to begin S1a tensor-free training** because the executable training path, the symmetry-blueprint sampler, and the production reverse sampler are all missing, and the Cartesian atlas currently violates the frozen latency budget.
