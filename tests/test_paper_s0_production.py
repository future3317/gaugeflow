import inspect
from pathlib import Path

import torch

from gaugeflow.checkpoints import load_safe_checkpoint, save_safe_checkpoint
from gaugeflow.manifold import vector_to_symmetric
from gaugeflow.production.archive_harmonic.harmonic_gaugeflow import (
    GeometryHarmonicQueries,
    HarmonicGaugeFlowConditioner,
    nested_hopf_so3_grid,
    weighted_geometric_harmonic_queries,
    weighted_harmonic_alignment_scores,
)
from gaugeflow.production.categorical_mask import AbsorbingMaskDiffusion
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape, SymmetryShapeBasis
from gaugeflow.production.space_group_router import (
    TerminalGroupCompatibilityRouter,
    compatibility_record,
    orbit_compatibility_residual,
)
from gaugeflow.production.symmetry_expand import expand_asymmetric_unit
from gaugeflow.production.wrapped_coordinates import AdaptiveWrappedQuotient
from gaugeflow.tensor import piezo_from_irreps, piezo_to_irreps, response_field, rotate_rank3
from gaugeflow.vocabulary import atomic_numbers_to_tokens, tokens_to_atomic_numbers


def _trace_free_projector(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    identity = torch.eye(6, dtype=dtype)
    trace = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return identity - torch.outer(trace, trace) / 3.0


def _small_hybrid_input():
    tokens = torch.tensor([4, 6, 12, 15], dtype=torch.long)
    frac = torch.tensor(
        [[0.05, 0.10, 0.15], [0.35, 0.25, 0.70], [0.15, 0.75, 0.45], [0.72, 0.55, 0.20]]
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    log_volume = torch.log(torch.tensor([64.0, 91.125]))
    log_shape = torch.zeros((2, 6))
    condition = torch.randn((2, 18), generator=torch.Generator().manual_seed(7))
    present = torch.ones((2, 1), dtype=torch.bool)
    projectors = _trace_free_projector().expand(2, -1, -1).clone()
    charts = torch.eye(3).expand(2, -1, -1).clone()
    return tokens, frac, log_volume, log_shape, batch, condition, present, projectors, charts


def test_element_vocabulary_roundtrip():
    atomic_numbers = torch.arange(1, 119)
    tokens = atomic_numbers_to_tokens(atomic_numbers)
    diffusion = AbsorbingMaskDiffusion()
    assert torch.equal(diffusion.decode(tokens), atomic_numbers)
    assert torch.equal(tokens_to_atomic_numbers(tokens), atomic_numbers)
    try:
        diffusion.decode(torch.tensor([118]))
    except ValueError as error:
        assert "outside 0..117" in str(error)
    else:
        raise AssertionError("MASK must never decode as a chemical element")


def test_absorbing_reverse_kernel_is_normalized_and_copies_revealed_tokens():
    process = AbsorbingMaskDiffusion()
    current = torch.tensor([3, process.mask_index, process.mask_index])
    logits = torch.randn((3, 118), generator=torch.Generator().manual_seed(8))
    batch = torch.tensor([0, 0, 0])
    probability = process.reverse_probabilities(
        current, logits, torch.tensor([0.8]), torch.tensor([0.4]), batch
    )
    assert torch.allclose(probability.sum(-1), torch.ones(3), atol=1e-6)
    assert probability[0, 3] == 1 and torch.count_nonzero(probability[0]) == 1
    assert torch.all(probability[1:, :118] >= 0)


def test_wrapped_score_matches_autograd():
    kernel = AdaptiveWrappedQuotient(
        absolute_tail_tolerance=1e-12, relative_tail_tolerance=1e-10, max_images=100_000
    )
    current = torch.tensor(
        [[0.13, 0.27, 0.41], [0.61, 0.52, 0.19]], dtype=torch.float64, requires_grad=True
    )
    clean = torch.tensor([[0.07, 0.11, 0.37], [0.49, 0.66, 0.22]], dtype=torch.float64)
    lattice = torch.tensor([[3.0, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 4.1]], dtype=torch.float64)
    result = kernel.evaluate(current, clean, lattice, 0.45)
    gradient = torch.autograd.grad(result.log_unnormalized_density, current)[0]
    assert torch.allclose(gradient, result.fractional_score, atol=2e-10, rtol=2e-10)


def test_single_site_translation_quotient_has_no_coordinate_degree_of_freedom():
    kernel = AdaptiveWrappedQuotient()
    current = torch.tensor([[0.3, 0.4, 0.5]], requires_grad=True)
    result = kernel.evaluate(current, torch.tensor([[0.8, 0.1, 0.2]]), torch.eye(3), 0.5)
    assert result.log_unnormalized_density == 0
    assert torch.equal(result.fractional_score, torch.zeros_like(current))
    assert torch.equal(
        torch.autograd.grad(result.log_unnormalized_density, current)[0], torch.zeros_like(current)
    )


def test_wrapped_kernel_translation_invariance():
    kernel = AdaptiveWrappedQuotient(absolute_tail_tolerance=1e-12, max_images=100_000)
    current = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.4, 0.8]], dtype=torch.float64)
    clean = torch.tensor([[0.2, 0.1, 0.4], [0.6, 0.5, 0.7]], dtype=torch.float64)
    lattice = torch.diag(torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64))
    shift = torch.tensor([0.31, -0.27, 1.19], dtype=torch.float64)
    first = kernel.evaluate(current, clean, lattice, 0.5)
    second = kernel.evaluate(current + shift, clean + shift, lattice, 0.5)
    assert torch.allclose(first.log_unnormalized_density, second.log_unnormalized_density, atol=1e-12)
    assert torch.allclose(first.fractional_score, second.fractional_score, atol=1e-11)


def test_wrapped_kernel_unimodular_basis_invariance():
    kernel = AdaptiveWrappedQuotient(absolute_tail_tolerance=1e-13, max_images=1_000_000)
    current = torch.tensor([[0.17, 0.31, 0.44], [0.72, 0.28, 0.81]], dtype=torch.float64)
    clean = torch.tensor([[0.05, 0.12, 0.38], [0.64, 0.39, 0.75]], dtype=torch.float64)
    lattice = torch.tensor([[3.0, 0.0, 0.0], [0.4, 3.6, 0.0], [0.2, 0.5, 4.2]], dtype=torch.float64)
    basis = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    inverse = torch.linalg.inv(basis)
    original = kernel.evaluate(current, clean, lattice, 0.42)
    transformed = kernel.evaluate(current @ inverse, clean @ inverse, basis @ lattice, 0.42)
    assert torch.allclose(
        original.log_unnormalized_density, transformed.log_unnormalized_density, atol=2e-10, rtol=2e-10
    )
    # Scores are covectors under f' = f B^-1: grad_f' = grad_f B^T.
    assert torch.allclose(transformed.fractional_score, original.fractional_score @ basis.T, atol=2e-9)


def test_adaptive_image_sum_tail_bound():
    loose = AdaptiveWrappedQuotient(
        absolute_tail_tolerance=1e-4, relative_tail_tolerance=1e-4, max_images=100_000
    )
    strict = AdaptiveWrappedQuotient(
        absolute_tail_tolerance=1e-15, relative_tail_tolerance=1e-15, max_images=1_000_000
    )
    current = torch.tensor([[0.11, 0.21, 0.31], [0.66, 0.47, 0.82]], dtype=torch.float64)
    clean = torch.tensor([[0.03, 0.17, 0.29], [0.58, 0.53, 0.71]], dtype=torch.float64)
    lattice = 2.5 * torch.eye(3, dtype=torch.float64)
    coarse = loose.evaluate(current, clean, lattice, 0.65)
    reference = strict.evaluate(current, clean, lattice, 0.65)
    omitted = float(torch.exp(reference.log_unnormalized_density) - torch.exp(coarse.log_unnormalized_density))
    assert omitted >= -1e-12
    assert omitted <= coarse.omitted_weight_upper_bound + 1e-10
    assert reference.radius >= coarse.radius


def test_lattice_volume_shape_roundtrip_and_symmetry_projection():
    lattice = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.2, 4.0, 0.0], [0.1, 0.3, 5.0]]], dtype=torch.float64
    )
    chart = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    state = LatticeVolumeShape.from_lattice(lattice, chart)
    assert torch.allclose(
        state.metric(chart), lattice @ lattice.transpose(-1, -2), atol=2e-12, rtol=2e-12
    )
    record = compatibility_record(75)
    basis = SymmetryShapeBasis.from_operations(record.operations)
    assert basis.dimension == 1
    projected = basis.project(torch.randn((4, 6), dtype=torch.float64))
    matrices = vector_to_symmetric(projected)
    transformed = torch.einsum("oip,npq,ojq->noij", record.operations, matrices, record.operations)
    assert torch.allclose(transformed, matrices.unsqueeze(1), atol=2e-10, rtol=2e-10)


def test_space_group_expansion_exact():
    frac = torch.tensor([[0.13, 0.21, 0.34]], dtype=torch.float64)
    species = torch.tensor([13], dtype=torch.long)
    rotations = torch.stack((torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)))
    translations = torch.zeros((2, 3), dtype=torch.float64)
    lattice = 4.0 * torch.eye(3, dtype=torch.float64)
    expanded = expand_asymmetric_unit(frac, species, rotations, translations, lattice)
    assert expanded.frac_coords.shape == (2, 3)
    assert torch.equal(expanded.species, torch.tensor([13, 13]))
    assert any(torch.allclose(value, torch.remainder(-frac[0], 1.0)) for value in expanded.frac_coords)


def test_router_is_tensor_representative_invariant():
    torch.manual_seed(12)
    condition = torch.randn((2, 18), dtype=torch.float64)
    tensor = piezo_from_irreps(condition)
    frames = nested_hopf_so3_grid(128, dtype=torch.float64)
    representative_rotation = frames[19]
    representative = piezo_to_irreps(rotate_rank3(tensor, representative_rotation))
    operations = compatibility_record(99).operations
    left = orbit_compatibility_residual(representative, operations, frames)
    right = orbit_compatibility_residual(condition, operations, frames @ representative_rotation)
    # Cartesian/e3nn basis conversion contributes about 1e-8 in float64;
    # the residual itself is otherwise identical under the right-shifted rule.
    assert torch.allclose(left, right, atol=2e-8, rtol=2e-8)
    router = TerminalGroupCompatibilityRouter([1, 2], hidden_dim=16, rotation_count=12)
    nonzero_logits, _ = router(condition.float())
    zero_logits, zero_residual = router(torch.zeros_like(condition.float()))
    assert torch.isneginf(nonzero_logits[:, 1]).all()
    assert torch.isfinite(zero_logits).all() and torch.equal(zero_residual, torch.zeros_like(zero_residual))


def test_response_field_is_lossless():
    tensor = piezo_from_irreps(torch.randn((3, 18), dtype=torch.float64))
    u = torch.randn((3, 3), dtype=torch.float64)
    v = torch.randn((3, 3), dtype=torch.float64)
    bilinear = torch.einsum("bijk,bj,bk->bi", tensor, u, v)
    polarized = 0.5 * (
        response_field(tensor, u + v) - response_field(tensor, u) - response_field(tensor, v)
    )
    assert torch.allclose(bilinear, polarized, atol=2e-12, rtol=2e-12)


def test_continuous_harmonic_score_covariance():
    torch.manual_seed(13)
    condition = torch.randn((1, 18), dtype=torch.float64)
    directions = torch.nn.functional.normalize(torch.randn((8, 3), dtype=torch.float64), dim=-1)
    edge_graph = torch.zeros(8, dtype=torch.long)
    query_weights = torch.randn((8, 2), dtype=torch.float64)
    rotations = nested_hopf_so3_grid(31, dtype=torch.float64)
    g, h = rotations[7], rotations[17]
    transformed_condition = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), h))
    weights = dict(
        coupling_l1=torch.tensor([[0.6, -0.4], [0.2, 0.7]], dtype=torch.float64),
        coupling_l2=torch.tensor([[0.3], [-0.1]], dtype=torch.float64),
        coupling_l3=torch.tensor([[-0.8], [0.4]], dtype=torch.float64),
    )
    left = weighted_harmonic_alignment_scores(
        transformed_condition, directions @ g.T, edge_graph, query_weights, rotations, **weights
    )
    right = weighted_harmonic_alignment_scores(
        condition, directions, edge_graph, query_weights,
        g.T.unsqueeze(0) @ rotations @ h.unsqueeze(0), **weights
    )
    assert torch.allclose(left, right, atol=4e-5, rtol=4e-5)


def test_state_weighted_queries_do_not_artificially_cancel_polar_odd_degrees():
    directions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    edge_graph = torch.zeros(2, dtype=torch.long)
    asymmetric_weights = torch.tensor([[1.0, 0.2], [0.1, -0.4]])
    first, _, third = weighted_geometric_harmonic_queries(
        directions, edge_graph, asymmetric_weights, graph_count=1
    )
    assert torch.linalg.vector_norm(first) > 0
    assert torch.linalg.vector_norm(third) > 0


def test_full_o3_router_operations_form_a_group_and_keep_improper_compatibility():
    record = compatibility_record(99)
    determinants = torch.linalg.det(record.operations)
    assert bool((determinants < 0).any()) and bool((determinants > 0).any())
    for left in record.operations:
        for right in record.operations:
            residual = torch.linalg.matrix_norm(left @ right - record.operations, dim=(-2, -1))
            assert float(residual.min()) < 2e-10
    assert compatibility_record(2).compatible_rank == 0


def test_double_coset_stabilizer_identity_and_physical_zero_is_not_cfg_null():
    conditioner = HarmonicGaugeFlowConditioner(hidden_dim=24, grid_size=32).eval()
    condition = torch.zeros((2, 18))
    directions = torch.nn.functional.normalize(torch.randn((6, 3)), dim=-1)
    edge_graph = torch.tensor([0, 0, 0, 1, 1, 1])
    queries = GeometryHarmonicQueries(
        first=torch.randn((2, 2, 3)),
        second=torch.randn((2, 2, 5)),
        third=torch.randn((2, 2, 7)),
    )
    output = conditioner(
        condition, torch.tensor([[True], [False]]), directions, edge_graph,
        queries, torch.tensor([0.4, 0.4])
    )
    assert torch.allclose(output.posterior, torch.full_like(output.posterior, 1.0 / 32), atol=1e-6)
    assert not torch.allclose(output.graph_condition[0], output.graph_condition[1])
    assert torch.equal(output.edge_response, torch.zeros_like(output.edge_response))


def test_finite_grid_error_decreases_with_k():
    coarse = nested_hopf_so3_grid(32, dtype=torch.float64)
    fine = nested_hopf_so3_grid(512, dtype=torch.float64)
    assert torch.equal(coarse, fine[:32])
    coarse_error = torch.linalg.matrix_norm(coarse.mean(dim=0))
    fine_error = torch.linalg.matrix_norm(fine.mean(dim=0))
    assert fine_error < coarse_error / 4.0


def test_time_reaches_every_block_and_head_and_coordinate_score_has_zero_graph_mean():
    torch.manual_seed(14)
    model = HybridCrystalDenoiser(
        hidden_dim=32, vector_dim=8, layers=2, radial_dim=6, atlas_residual_circle_samples=8
    ).eval()
    values = _small_hybrid_input()
    first = model(*values[:5], torch.tensor([0.2, 0.2]), *values[5:])
    second = model(*values[:5], torch.tensor([0.8, 0.8]), *values[5:])
    for block in model.blocks:
        assert hasattr(block, "time_film") and hasattr(block, "condition_film")
    assert not torch.allclose(first.clean_element_logits, second.clean_element_logits)
    assert not torch.allclose(first.coordinate_fractional_score, second.coordinate_fractional_score)
    assert not torch.allclose(first.clean_log_volume, second.clean_log_volume)
    assert not torch.allclose(first.clean_log_shape, second.clean_log_shape)
    batch = values[4]
    for graph in range(2):
        selected = first.coordinate_fractional_score[batch == graph]
        assert torch.allclose(selected.mean(dim=0), torch.zeros(3), atol=2e-6)
    lattice = LatticeVolumeShape(values[2], values[3]).lattice(values[8])
    expected_fractional = torch.einsum(
        "ni,nij->nj", first.coordinate_cartesian_score, lattice[batch].transpose(-1, -2)
    )
    expected_fractional = expected_fractional - torch.stack(
        [expected_fractional[batch == graph].mean(0) for graph in range(2)]
    )[batch]
    assert torch.allclose(first.coordinate_fractional_score, expected_fractional, atol=2e-6)


def test_no_target_metadata_in_model_signature():
    parameters = set(inspect.signature(HybridCrystalDenoiser.forward).parameters)
    forbidden = {
        "material_id", "niggli_transform", "response_stratum", "zero_response",
        "target_cif", "target_lattice", "target_space_group", "target_stabilizer",
        "source_id", "endpoint_id", "target_metadata",
    }
    assert parameters.isdisjoint(forbidden)


def test_checkpoint_manifest_hashes(tmp_path: Path):
    model = HybridCrystalDenoiser(
        hidden_dim=16, vector_dim=4, layers=1, radial_dim=4, atlas_residual_circle_samples=8
    )
    path = tmp_path / "s0_weights.pt"
    sidecar = save_safe_checkpoint(
        path,
        model_state=model.state_dict(),
        isotypic_scales=torch.ones(3),
        training_step=0,
        metadata={"design_sha256": "9ad4ed018600a62b5f663255a1e0a4d59abcdc26303e523a4f151bdfaf07dd31"},
    )
    payload, metadata = load_safe_checkpoint(path, map_location="cpu")
    assert payload["training_step"] == 0
    assert metadata["design_sha256"].startswith("9ad4ed")
    assert sidecar.is_file()
