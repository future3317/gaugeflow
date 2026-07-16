import inspect
import math

import pytest
import torch

from gaugeflow.geometry import GaussianRadialBasis
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.harmonic_gaugeflow import (
    ConditionFreeGeometryQueryEncoder,
    GeometryHarmonicQueries,
    HarmonicGaugeFlowConditioner,
)
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.space_group_router import compatibility_record
from gaugeflow.production.state_projection import project_hybrid_reverse_state
from gaugeflow.production.wrapped_coordinates import (
    AdaptiveWrappedQuotient,
    ScalableWrappedQuotient,
)


def _complete_geometry(
    positions: torch.Tensor,
    *,
    hidden_dim: int = 16,
    radial_dim: int = 5,
) -> tuple[torch.Tensor, ...]:
    sites = positions.shape[0]
    source = torch.tensor(
        [left for left in range(sites) for right in range(sites) if left != right]
    )
    target = torch.tensor(
        [right for left in range(sites) for right in range(sites) if left != right]
    )
    displacement = positions[target] - positions[source]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / distance.unsqueeze(-1)
    radial = GaussianRadialBasis(radial_dim, 8.0)(distance)
    token = torch.randn((1, hidden_dim), generator=torch.Generator().manual_seed(81))
    initial_nodes = token.expand(sites, -1).clone()
    node_time = torch.zeros_like(initial_nodes)
    batch = torch.zeros(sites, dtype=torch.long)
    return initial_nodes, node_time, source, target, direction, radial, batch


def _query_encoder() -> ConditionFreeGeometryQueryEncoder:
    torch.manual_seed(82)
    return ConditionFreeGeometryQueryEncoder(16, 5, query_channels=2, layers=3).eval()


def test_point_group_metric_chart_has_all_crystal_system_dimensions_and_hexagonal_invariance():
    representatives = {
        1: 5,    # triclinic
        3: 3,    # monoclinic
        16: 2,   # orthorhombic
        75: 1,   # tetragonal
        143: 1,  # trigonal
        168: 1,  # hexagonal
        195: 0,  # cubic
    }
    for number, dimension in representatives.items():
        assert compatibility_record(number).metric_chart.shape_dimension == dimension
    generator = torch.Generator().manual_seed(83)
    for number in (143, 168):
        chart = compatibility_record(number).metric_chart
        coordinates = torch.randn(
            chart.shape_dimension, generator=generator, dtype=torch.float64
        )
        log_shape = coordinates @ chart.invariant_log_shape_basis.T
        metric = chart.metric(torch.tensor(math.log(80.0), dtype=torch.float64), log_shape)
        assert float(chart.invariance_residual(metric).max()) < 1e-8


def test_all_230_space_group_charts_close_and_have_correct_piezoelectric_ranks():
    zero_rank_point_groups = {
        "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-31m", "-3m1", "-3m",
        "6/m", "6/mmm", "m-3", "432", "m-3m",
    }
    for number in range(1, 231):
        record = compatibility_record(number)
        fractional = record.fractional_operations
        cartesian = record.operations
        fractional_products = fractional[:, None] @ fractional[None, :]
        fractional_closure = torch.linalg.matrix_norm(
            fractional_products[:, :, None] - fractional[None, None], dim=(-2, -1)
        ).amin(dim=-1)
        cartesian_products = cartesian[:, None] @ cartesian[None, :]
        cartesian_closure = torch.linalg.matrix_norm(
            cartesian_products[:, :, None] - cartesian[None, None], dim=(-2, -1)
        ).amin(dim=-1)
        assert float(fractional_closure.max()) < 1e-10
        assert float(cartesian_closure.max()) < 1e-9
        assert torch.allclose(
            torch.linalg.det(fractional).abs(), torch.ones(fractional.shape[0], dtype=torch.float64)
        )
        assert torch.allclose(
            cartesian.transpose(-1, -2) @ cartesian,
            torch.eye(3, dtype=torch.float64).expand_as(cartesian),
            atol=1e-9,
            rtol=1e-9,
        )
        reynolds = record.reynolds_irrep
        assert torch.allclose(reynolds @ reynolds, reynolds, atol=2e-9, rtol=2e-9)
        if record.point_group in zero_rank_point_groups:
            assert record.compatible_rank == 0
        else:
            assert record.compatible_rank > 0


@pytest.mark.parametrize("sites,sigma", [(2, 0.35), (3, 0.35), (4, 0.20)])
def test_scalable_wrapped_quotient_matches_small_site_exact_oracle(sites: int, sigma: float):
    generator = torch.Generator().manual_seed(90 + sites)
    current = torch.rand((sites, 3), generator=generator, dtype=torch.float64)
    clean = torch.rand((sites, 3), generator=generator, dtype=torch.float64)
    lattice = torch.tensor(
        [[2.2, 0.0, 0.0], [0.3, 2.5, 0.0], [0.2, 0.4, 2.8]], dtype=torch.float64
    )
    exact = AdaptiveWrappedQuotient(
        absolute_tail_tolerance=1e-10,
        relative_tail_tolerance=1e-10,
        max_images=2_000_000,
    ).evaluate(current, clean, lattice, sigma)
    scalable = ScalableWrappedQuotient().evaluate(current, clean, lattice, sigma)
    log_error = (exact.log_unnormalized_density - scalable.log_unnormalized_density).abs()
    score_error = torch.linalg.vector_norm(exact.fractional_score - scalable.fractional_score)
    score_scale = torch.linalg.vector_norm(exact.fractional_score) + 1e-8
    assert float(log_error) <= 1e-6
    assert float(score_error / score_scale) <= 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="S0.2 scalability qualification uses pinned CUDA")
def test_scalable_wrapped_quotient_handles_twenty_sites_and_triclinic_metrics():
    generator = torch.Generator(device="cpu").manual_seed(96)
    current = torch.rand((20, 3), generator=generator, dtype=torch.float32).cuda()
    clean = torch.rand((20, 3), generator=generator, dtype=torch.float32).cuda()
    lattices = (
        2.5 * torch.eye(3, dtype=torch.float32, device="cuda"),
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.35, 1.8, 0.0], [0.25, 0.2, 5.0]],
            dtype=torch.float32,
            device="cuda",
        ),
    )
    kernel = ScalableWrappedQuotient(
        kernel_tail_tolerance=1e-8,
        qmc_log_tolerance=2e-3,
        qmc_relative_score_tolerance=2e-3,
        chunk_size=1024,
    )
    for lattice in lattices:
        for sigma in (0.25, 0.50):
            result = kernel.evaluate(current, clean, lattice, sigma)
            assert torch.isfinite(result.log_unnormalized_density)
            assert torch.isfinite(result.fractional_score).all()
            assert result.qmc_samples <= 128**3


def test_all_mask_noncentrosymmetric_geometry_has_allowed_queries():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.3, 0.4], [0.3, 0.2, 1.7]]
    )
    queries = _query_encoder()(*_complete_geometry(positions), graph_count=1)
    assert torch.linalg.vector_norm(queries.first) > 1e-5
    assert torch.linalg.vector_norm(queries.third) > 1e-5


def test_single_species_chiral_geometry_retains_l3_query():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3], [0.35, 0.25, 1.4]]
    )
    queries = _query_encoder()(*_complete_geometry(positions), graph_count=1)
    assert torch.linalg.vector_norm(queries.third) > 1e-5


def test_inversion_symmetric_geometry_has_zero_odd_query():
    positions = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    queries = _query_encoder()(*_complete_geometry(positions), graph_count=1)
    assert torch.linalg.vector_norm(queries.first) < 2e-7
    assert torch.linalg.vector_norm(queries.third) < 2e-7


def test_query_uses_no_tensor_condition():
    parameters = set(inspect.signature(ConditionFreeGeometryQueryEncoder.forward).parameters)
    assert parameters.isdisjoint({"tensor", "tensor_condition", "condition", "piezo_irreps"})


def _denoiser_input() -> tuple[torch.Tensor, ...]:
    tokens = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    fractional = torch.tensor(
        [[0.05, 0.10, 0.15], [0.35, 0.25, 0.68], [0.15, 0.73, 0.45], [0.72, 0.55, 0.20]]
    )
    log_volume = torch.tensor([math.log(27.0)])
    log_shape = torch.tensor([[0.2, -0.1, 0.3, 0.07, -0.05, 0.04]])
    batch = torch.zeros(4, dtype=torch.long)
    time = torch.tensor([0.45])
    condition = torch.zeros((1, 18))
    present = torch.zeros((1, 1), dtype=torch.bool)
    trace = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    projector = (torch.eye(6) - torch.outer(trace, trace) / 3.0).unsqueeze(0)
    chart = torch.eye(3).unsqueeze(0)
    return (
        tokens, fractional, log_volume, log_shape, batch, time,
        condition, present, projector, chart,
    )


def test_full_denoiser_projects_input_shape_and_is_translation_equivariant():
    torch.manual_seed(101)
    model = HybridCrystalDenoiser(
        hidden_dim=24, vector_dim=6, layers=2, radial_dim=5, harmonic_grid=16
    ).eval()
    values = _denoiser_input()
    projected_shape = torch.einsum("bij,bj->bi", values[8], values[3])
    projected_values = (*values[:3], projected_shape, *values[4:])
    first = model(*values)
    projected = model(*projected_values)
    shifted_values = list(projected_values)
    shifted_values[1] = shifted_values[1] + torch.tensor([0.31, -0.27, 1.19])
    shifted = model(*shifted_values)
    for name in (
        "clean_element_logits", "coordinate_cartesian_score", "coordinate_fractional_score",
        "log_volume_score", "log_shape_score",
    ):
        assert torch.allclose(getattr(first, name), getattr(projected, name), atol=3e-6, rtol=3e-6)
        assert torch.allclose(getattr(projected, name), getattr(shifted, name), atol=3e-6, rtol=3e-6)


def test_full_unconditional_denoiser_is_unimodular_basis_equivariant():
    torch.manual_seed(102)
    model = HybridCrystalDenoiser(
        hidden_dim=24, vector_dim=6, layers=2, radial_dim=5, harmonic_grid=16
    ).eval()
    values = list(_denoiser_input())
    values[3] = torch.zeros_like(values[3])
    original = model(*values)
    basis = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transformed_values = list(values)
    transformed_values[1] = values[1] @ torch.linalg.inv(basis)
    transformed_values[9] = (values[9] @ basis.T).contiguous()
    transformed = model(*transformed_values)
    lattice_original = LatticeVolumeShape(values[2], values[3]).lattice(values[9])
    lattice_transformed = LatticeVolumeShape(
        transformed_values[2], transformed_values[3]
    ).lattice(transformed_values[9])
    rotation = (
        torch.linalg.inv(lattice_original[0])
        @ torch.linalg.inv(basis)
        @ lattice_transformed[0]
    )
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        transformed.clean_element_logits, original.clean_element_logits, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        transformed.coordinate_cartesian_score,
        original.coordinate_cartesian_score @ rotation,
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(
        transformed.coordinate_fractional_score,
        original.coordinate_fractional_score @ basis.T,
        atol=4e-5,
        rtol=4e-5,
    )
    assert torch.allclose(transformed.log_volume_score, original.log_volume_score, atol=2e-5)
    assert torch.allclose(transformed.log_shape_score, original.log_shape_score, atol=2e-5)


def test_reverse_step_projection_is_idempotent_for_translation_and_shape():
    values = _denoiser_input()
    first = project_hybrid_reverse_state(values[1], values[3], values[4], values[8])
    second = project_hybrid_reverse_state(
        first.fractional_coordinates, first.log_shape, values[4], values[8]
    )
    assert torch.allclose(first.fractional_coordinates.mean(dim=0), torch.zeros(3), atol=1e-7)
    assert torch.allclose(first.fractional_coordinates, second.fractional_coordinates, atol=1e-7)
    assert torch.allclose(first.log_shape, second.log_shape, atol=1e-7)


def _finite_qmc_object(grid_size: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(107)
    conditioner = HarmonicGaugeFlowConditioner(24, grid_size=grid_size, query_channels=2).double()
    condition = torch.randn((1, 18), dtype=torch.float64)
    first = torch.randn((1, 2, 3), dtype=torch.float64, requires_grad=True)
    second = torch.randn((1, 2, 5), dtype=torch.float64, requires_grad=True)
    third = torch.randn((1, 2, 7), dtype=torch.float64, requires_grad=True)
    queries = GeometryHarmonicQueries(first, second, third)
    directions = torch.nn.functional.normalize(torch.randn((6, 3), dtype=torch.float64), dim=-1)
    edge_graph = torch.zeros(6, dtype=torch.long)
    output = conditioner(
        condition, torch.ones((1, 1), dtype=torch.bool), directions, edge_graph,
        queries, torch.tensor([0.2], dtype=torch.float64),
    )
    gradient = torch.autograd.grad(output.graph_condition.square().sum(), (first, second, third))
    posterior_rotation_moment = torch.einsum(
        "bf,fij->bij", output.posterior, conditioner.rotations
    )
    return (
        output.aligned_irreps.detach(),
        posterior_rotation_moment.detach(),
        output.graph_condition.detach(),
        torch.cat([value.flatten() for value in gradient]).detach(),
    )


def test_finite_qmc_model_objects_converge_under_nested_refinement():
    coarse = _finite_qmc_object(240)
    medium = _finite_qmc_object(960)
    fine = _finite_qmc_object(3840)
    for coarse_value, medium_value, fine_value in zip(coarse, medium, fine):
        coarse_error = torch.linalg.vector_norm(coarse_value - fine_value)
        medium_error = torch.linalg.vector_norm(medium_value - fine_value)
        assert medium_error < coarse_error
