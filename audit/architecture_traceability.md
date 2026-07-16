# Architecture Traceability

> Independent review: static traceability is largely valid, but a missing
> complete blueprint is assigned to S2 by the current versioned contract, not
> to the S1a reverse-sampler qualification. See `CODEX_VERIFICATION.md`.

Audit date: 2026-07-16
Code root: `E:\CODE\T2C-Flow\gaugeflow_perf_audit`
Paper: `E:\PAPER\GaugeFlow Stabilizer-Aware Response-Field Flow for Tensor-Orbit Conditioned Crystal Generation\GaugeFlow.tex`
Design contract: `docs/paper_design_contract_v1.md` (SHA-256 pinned in `src/gaugeflow/production/s0_audit.py:19`)

Scope: verify that the declared production architecture is faithfully implemented. Every row cites exact `file:line` evidence.

## 1. Element categorical diffusion (118 + MASK)

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| 118 chemical tokens `0..117` map to atomic numbers `Z=1..118` | `AbsorbingMaskDiffusion` fixes `element_count = CHEMICAL_ELEMENT_COUNT` and decodes via `tokens + 1` | VERIFIED | `src/gaugeflow/production/categorical_mask.py:32-38`, `110-112`; `src/gaugeflow/vocabulary.py` defines `CHEMICAL_ELEMENT_COUNT = 118` |
| MASK token is `118`, absorbing, never decoded | `mask_index = element_count`; `validate_clean` rejects `tokens >= element_count` | VERIFIED | `src/gaugeflow/production/categorical_mask.py:37`, `40-44` |
| Corruption uses continuous-time absorbing mask with `alpha(t)^2` survival | `corrupt` computes `keep_probability = self.schedule.alpha(time).square()[batch]` | VERIFIED | `src/gaugeflow/production/categorical_mask.py:59-68` |
| Reverse kernel copies revealed tokens, unmasks masked ones analytically | `reverse_probabilities` returns 119-column probabilities; masked tokens get `reveal` mass on chemical softmax and residual mass on MASK; revealed tokens are copied exactly | VERIFIED | `src/gaugeflow/production/categorical_mask.py:70-108` |
| MASK excluded from clean posterior and decoded samples | `validate_clean` enforces `0..117`; tests assert decode of `118` raises `ValueError` | VERIFIED | `tests/test_paper_s0_production.py:57-61` |

## 2. Coordinate / translation quotient

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| Wrapped Gaussian density on graphwise translation quotient | `AdaptiveWrappedQuotient.evaluate` fixes last-site image to zero, expands `3(M-1)` integer lattice adaptively, projects displacements to zero-mean before exponentiation | VERIFIED | `src/gaugeflow/production/wrapped_coordinates.py:41-50`, `96-184` |
| Exact small-site oracle with certified tail stop | Expansion stops only after Gaussian-tail bound below tolerance; `max_images` is fail-closed resource guard, not fixed-shell approximation | VERIFIED | `src/gaugeflow/production/wrapped_coordinates.py:52-67`, `77-81` |
| Scalable common-translation QMC | `ScalableWrappedQuotient.evaluate` replaces exponential lattice with 3-D common-translation integral, chooses image vs Fourier dual kernels by truncation cost, uses dyadic nested rank-3 torus lattice rule | VERIFIED | `src/gaugeflow/production/wrapped_coordinates.py:362-475` |
| Zero-mean projector `P_M` | `translation_horizontal` subtracts per-graph mean; `project_translation_state` uses `scatter(..., reduce="mean")` | VERIFIED | `src/gaugeflow/production/wrapped_coordinates.py:35-38`; `src/gaugeflow/production/state_projection.py:19-32` |
| Fractional coordinates deliberately not wrapped during reverse trajectory | `project_hybrid_reverse_state` docstring: wrapping is terminal decoding only | VERIFIED | `src/gaugeflow/production/state_projection.py:41-53` |

## 3. Lattice chart: log-volume + trace-free log-shape

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| Lattice represented by `log_volume` and 6-dim trace-free `log_shape` | `LatticeVolumeShape` stores both; `from_lattice` removes trace drift via `log_shape_matrix - trace/3 * I` | VERIFIED | `src/gaugeflow/production/lattice_volume_shape.py:12-60` |
| Fractional metric reconstructed as `G_raw = C^T exp(A) C` | `metric` computes `chart.T @ spd_exp(matrix) @ chart`, then determinant normalizes and rescales by `exp((2/3) log_volume)` | VERIFIED | `src/gaugeflow/production/lattice_volume_shape.py:62-78` |
| Lattice recovered by Cholesky of metric | `lattice` returns `torch.linalg.cholesky(self.metric(...))` | VERIFIED | `src/gaugeflow/production/lattice_volume_shape.py:80-81` |
| Point-group-compatible trace-free subspace | `SymmetryShapeBasis.from_operations` builds invariant basis via SVD of `(action - identity)` plus trace constraint | VERIFIED | `src/gaugeflow/production/lattice_volume_shape.py:84-143` |
| Input and future reverse-step states projected, not only score head | `PointGroupMetricChart.project_log_shape` projects batched states; `project_lattice_state` used by `project_hybrid_reverse_state`; `HybridCrystalDenoiser.forward` projects input `log_shape` | VERIFIED | `src/gaugeflow/production/lattice_volume_shape.py:197-227`; `src/gaugeflow/production/state_projection.py:35-53`; `src/gaugeflow/production/equivariant_denoiser.py:210` |

## 4. SO(3)/O(3) gauge and compatibility router

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| Tensor orbit is proper SO(3); improper operations only enter Reynolds router | `orbit_compatibility_residual` validates `det(rotations) == 1` | VERIFIED | `src/gaugeflow/production/space_group_router.py:119-133` |
| Full-O(3) Reynolds compatibility for odd-rank polar tensor | `reynolds_project` averages `rotate_rank3` over all operations, including improper ones; `cartesian_point_group_operations` explicitly preserves improper operations | VERIFIED | `src/gaugeflow/production/space_group_router.py:33-40`, `61-95` |
| Reynolds projector is idempotent | `reynolds_irrep_matrix` uses SVD to recover exact idempotent projector because e3nn basis conversion introduces `O(1e-8)` drift | VERIFIED | `src/gaugeflow/production/space_group_router.py:43-58` |
| All 230 space-group charts supported | `cartesian_point_group_operations` is `lru_cache(maxsize=230)`; test checks all 230 for closure and correct piezoelectric ranks | VERIFIED | `src/gaugeflow/production/space_group_router.py:61-62`; `tests/test_paper_s0_2_scalability_symmetry.py:75-108` |

## 5. Cartesian stratified gauge atlas

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| 4032 generic candidates = 24 proper frames × 7 chart nodes × 24 proper frames | `_base_cubature` builds `proper_frames[:, None, None] @ local[None, :, None] @ proper[None, None]` | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:420-423`; `tests/test_cartesian_gauge_atlas.py:216-224` |
| Deduplication with multiplicity-corrected prior | `_deduplicate_measure` rounds rotation matrices to integer keys, sums raw prior masses, returns unique rotations with aggregated priors | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:469-500` |
| Smooth partition-of-unity stratum weights | `_frame_data` uses cubic smoothstep over `[eta/2, 2 eta]` on relative eigen gaps; returns four weights (generic, lower axial, upper axial, isotropic) | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:369-402`; `tests/test_cartesian_gauge_atlas.py:253-274` |
| Axial SO(2) circle rule | `_axial_rotations` constructs `residual_circle_samples` SO(2) nodes; `_residual_nodes` dispatches `AXIAL` to it | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:84-97`, `404-408` |
| Descriptor-isotropic fallback | `_frame_data(..., directional=False)` zeroes weights; `_raw_candidate_measure` returns empty measure; `forward` sends zero-signal tensors to invariant-only conditioning with `gate = 0` | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:400-401`, `447-449`, `547-560`, `605-607` |
| Condition schedule gate `lambda_max * snr/(1+snr) * confidence * available` | `gate = self.lambda_max * (snr / (1.0 + snr)) * confidence * available.to(confidence)` | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:605-607` |
| Physical-zero tensor distinct from CFG null token | `graph_condition = torch.where(present, graph_condition, null_condition)`; zero tensor with `present=True` retains invariant and present_bias | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:612-616`; `src/gaugeflow/conditioning.py:16-20` |

## 6. Equivariant denoiser

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| Time and condition injected in every block via FiLM | `EquivariantDenoisingBlock` has `time_film` and `condition_film`, each producing scale/shift applied to scalar update | VERIFIED | `src/gaugeflow/production/equivariant_denoiser.py:61-62`, `115-118` |
| O(3)-typed scalar/vector message block | `scalar_message`, `vector_coefficients`, `scalar_update`, `vector_gate` implement typed messages | VERIFIED | `src/gaugeflow/production/equivariant_denoiser.py:52-65`, `99-122` |
| Coordinate score is Cartesian covector converted to fractional via `L^T` | `fractional_score = cartesian_score @ L^T`, then graphwise mean re-centred | VERIFIED | `src/gaugeflow/production/equivariant_denoiser.py:260-266` |
| No target-structure inputs or endpoint tokens | `HybridCrystalDenoiser.forward` signature contains only `element_tokens, frac_coords, log_volume, log_shape, batch, time, tensor_condition, condition_present, shape_projector, fractional_to_cartesian`; docstring explicitly forbids target CIF/lattice/space group/stabilizer/source/endpoint tokens | VERIFIED | `src/gaugeflow/production/equivariant_denoiser.py:173-190`; `tests/test_paper_s0_production.py:311-317` |
| Shape score projected onto symmetry-allowed subspace | `shape_score = torch.einsum("bij,bj->bi", shape_projector, raw_shape_score)` | VERIFIED | `src/gaugeflow/production/equivariant_denoiser.py:268-269` |

## 7. Blueprint sampler

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| Sample space-group blueprint, produce `shape_projector` and `fractional_to_cartesian` | No end-to-end sampler found. `PointGroupMetricChart` and `SpaceGroupCompatibilityRouter.compatibility_record` can produce these quantities, but no script or module wires them into a sampling procedure that feeds `HybridCrystalDenoiser` | MISSING | `src/gaugeflow/production/lattice_volume_shape.py:146-227`; `src/gaugeflow/production/space_group_router.py:98-116`; `reports/paper_architecture_compliance_v1.md:19` states "Complete Wyckoff autoregressive blueprint | Not implemented in this S0 change | S2 locked" |

## 8. Training and reverse sampler

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| Production training uses `HybridCrystalDenoiser` with absorbing categorical, wrapped quotient, and lattice chart | `scripts/train.py` trains `GaugeFlowVectorField` with `RiemannianCrystalFlowMatcher().loss()` | NOT IMPLEMENTED | `scripts/train.py:199-254` |
| Production sampling uses hybrid reverse integrator with `project_hybrid_reverse_state` | `scripts/sample.py` loads `GaugeFlowVectorField` and calls `RiemannianCrystalFlowMatcher().sample()` | NOT IMPLEMENTED | `scripts/sample.py:47-68` |
| Same probability model in training and reverse sampler | Legacy `flow.py` uses `GaugeFlowVectorField` and `RiemannianCrystalFlowMatcher`; production modules are isolated under `gaugeflow.production` with no legacy imports | PARTIAL | `src/gaugeflow/production/s0_audit.py:127-131` verifies no legacy imports in production; no production training/sampling script exists |

## 9. Data leakage / target metadata

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| No target CIF, lattice, space group, stabilizer, source ID, or endpoint token in model inputs | `HybridCrystalDenoiser.forward` signature contains none of these; `s0_audit.py` forbids them explicitly | VERIFIED | `src/gaugeflow/production/equivariant_denoiser.py:173-190`; `src/gaugeflow/production/s0_audit.py:20-28` |
| Target-CIF stabilizers deliberately not emitted as model inputs | `PiezoCrystalDataset.__getitem__` docstring states this; emitted fields are `atom_types, frac_coords, lattice, piezo_irreps, condition_present, niggli_transform, response_stratum, zero_response, material_id, num_nodes` | VERIFIED | `src/gaugeflow/data.py:89-96`, `213-250` |
| Potential leakage via `material_id`, `niggli_transform`, `response_stratum`, `zero_response` carried in `Data` object | These fields are not passed to `HybridCrystalDenoiser.forward`, but they remain on the batch object and are accessible to any caller | CONCERN | `src/gaugeflow/data.py:217-228`, `239-250` |

## 10. Numerical / performance

| Paper/spec claim | Implementation | Verdict | Evidence |
|---|---|---|---|
| No TODO, NotImplementedError, or placeholder | Grep across `src/gaugeflow` and `scripts` returns no matches | VERIFIED | `grep -r "TODO|FIXME|NotImplementedError|XXX" src scripts` (2026-07-16) |
| BF16 autocast safe: invariant algebra and graph reductions in FP32 | `cartesian_gauge_atlas.py` disables autocast for invariant/frame covariance and deduplication; `equivariant_denoiser.py` accumulates messages in residual FP32 state | VERIFIED | `src/gaugeflow/production/cartesian_gauge_atlas.py:276-281`, `380-381`, `502-508`; `src/gaugeflow/production/equivariant_denoiser.py:106-110` |
| S0.4 latency failure documented | README states S0.4-v1 failed frozen RTX 4060 Ti latency limit `41.89 ms > 20 ms`, decision `failed_no_advance` | VERIFIED | `README.md:41-66`; `reports/paper_s0_4_cartesian_atlas_prior_v1/official_summary.md` |

## Summary

- **Implemented and verified:** element diffusion, translation-quotient wrapped kernel, lattice chart, SO(3)/O(3) router, Cartesian atlas, equivariant denoiser, no-target-metadata contract, BF16 safety.
- **Missing:** symmetry-blueprint sampler, production training script, production reverse sampler.
- **Concern:** legacy training/sampling entry points remain active; potential leakage fields (`material_id`, `niggli_transform`) persist on the `Data` batch object.
