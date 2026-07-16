import math

import pytest
import torch

from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.space_group_router import compatibility_record
from gaugeflow.production.state_projection import project_hybrid_reverse_state
from gaugeflow.production.wrapped_coordinates import (
    AdaptiveWrappedQuotient,
    ScalableWrappedQuotient,
)


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
        hidden_dim=24, vector_dim=6, layers=2, radial_dim=5, atlas_residual_circle_samples=8
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
        "clean_log_volume", "clean_log_shape",
    ):
        assert torch.allclose(getattr(first, name), getattr(projected, name), atol=3e-6, rtol=3e-6)
        assert torch.allclose(getattr(projected, name), getattr(shifted, name), atol=3e-6, rtol=3e-6)


def test_full_unconditional_denoiser_is_unimodular_basis_equivariant():
    torch.manual_seed(102)
    model = HybridCrystalDenoiser(
        hidden_dim=24, vector_dim=6, layers=2, radial_dim=5, atlas_residual_circle_samples=8
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
    assert torch.allclose(transformed.clean_log_volume, original.clean_log_volume, atol=2e-5)
    assert torch.allclose(transformed.clean_log_shape, original.clean_log_shape, atol=2e-5)


def test_reverse_step_projection_is_idempotent_for_translation_and_shape():
    values = _denoiser_input()
    first = project_hybrid_reverse_state(values[1], values[3], values[4], values[8])
    second = project_hybrid_reverse_state(
        first.fractional_coordinates, first.log_shape, values[4], values[8]
    )
    assert torch.allclose(first.fractional_coordinates.mean(dim=0), torch.zeros(3), atol=1e-7)
    assert torch.allclose(first.fractional_coordinates, second.fractional_coordinates, atol=1e-7)
    assert torch.allclose(first.log_shape, second.log_shape, atol=1e-7)

