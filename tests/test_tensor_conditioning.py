import torch

from gaugeflow.production.cartesian_gauge_atlas import CartesianGaugeAtlasOutput
from gaugeflow.production.equivariant_denoiser import HybridDenoiserOutput
from gaugeflow.production.tensor_conditioning import (
    null_condition_retention_distance,
    orbit_mimic_distance,
    rotate_orbit_representative,
    sample_proper_rotations,
)
from gaugeflow.tensor import piezo_from_irreps


def _output(seed: int, *, nodes: int = 5, graphs: int = 2) -> HybridDenoiserOutput:
    generator = torch.Generator().manual_seed(seed)
    aligned = torch.randn((graphs, 3, 3, 3), generator=generator)
    aligned = 0.5 * (aligned + aligned.transpose(-1, -2))
    atlas = CartesianGaugeAtlasOutput(
        graph_condition=torch.randn((graphs, 8), generator=generator),
        edge_response=torch.randn((4, 3), generator=generator),
        posterior=torch.ones((graphs, 1)),
        candidate_prior=torch.ones((graphs, 1)),
        candidate_mask=torch.ones((graphs, 1), dtype=torch.bool),
        aligned_tensor=aligned,
        gate=torch.ones(graphs),
        entropy=torch.zeros(graphs),
        effective_frame_count=torch.ones(graphs, dtype=torch.long),
        raw_candidate_count=torch.ones(graphs, dtype=torch.long),
        residual_kind=torch.zeros(graphs, dtype=torch.long),
    )
    return HybridDenoiserOutput(
        clean_element_logits=torch.randn((nodes, 118), generator=generator),
        clean_composition_logits=torch.randn((graphs, 118), generator=generator),
        coordinate_cartesian_scaled_score=torch.randn((nodes, 3), generator=generator),
        coordinate_fractional_scaled_score=torch.randn((nodes, 3), generator=generator),
        clean_volume_latent=torch.randn(graphs, generator=generator),
        clean_shape_latent=torch.randn((graphs, 5), generator=generator),
        gauge_atlas=atlas,
    )


def test_sampled_orbit_representative_is_proper_and_preserves_norm() -> None:
    condition = torch.randn((7, 18), generator=torch.Generator().manual_seed(1))
    rotation = sample_proper_rotations(
        7, condition, generator=torch.Generator().manual_seed(2)
    )
    rotated = rotate_orbit_representative(condition, rotation)
    assert torch.allclose(
        torch.linalg.det(rotation), torch.ones(7), atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(
        torch.linalg.vector_norm(piezo_from_irreps(rotated).flatten(1), dim=-1),
        torch.linalg.vector_norm(piezo_from_irreps(condition).flatten(1), dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )


def test_orbit_mimic_is_zero_for_identical_marginalized_outputs() -> None:
    output = _output(3)
    distance = orbit_mimic_distance(
        output,
        output,
        torch.tensor([0, 0, 1, 1, 1]),
        2,
    )
    assert distance.loss.item() < 1e-9
    assert distance.response.item() < 1e-9


def test_orbit_mimic_reaches_all_typed_fields_and_gradients() -> None:
    first = _output(4)
    second = _output(5)
    second.clean_element_logits.requires_grad_()
    second.coordinate_cartesian_scaled_score.requires_grad_()
    second.clean_volume_latent.requires_grad_()
    second.clean_shape_latent.requires_grad_()
    second.gauge_atlas.aligned_tensor.requires_grad_()
    distance = orbit_mimic_distance(
        first,
        second,
        torch.tensor([0, 0, 1, 1, 1]),
        2,
    )
    distance.loss.backward()
    for value in (
        second.clean_element_logits,
        second.coordinate_cartesian_scaled_score,
        second.clean_volume_latent,
        second.clean_shape_latent,
        second.gauge_atlas.aligned_tensor,
    ):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.linalg.vector_norm(value.grad) > 0


def test_null_retention_excludes_atlas_response() -> None:
    teacher = _output(6)
    student = _output(6)
    replacement = _output(7).gauge_atlas
    student = HybridDenoiserOutput(
        clean_element_logits=student.clean_element_logits,
        clean_composition_logits=student.clean_composition_logits,
        coordinate_cartesian_scaled_score=student.coordinate_cartesian_scaled_score,
        coordinate_fractional_scaled_score=student.coordinate_fractional_scaled_score,
        clean_volume_latent=student.clean_volume_latent,
        clean_shape_latent=student.clean_shape_latent,
        gauge_atlas=replacement,
    )
    distance = null_condition_retention_distance(
        student,
        teacher,
        torch.tensor([0, 0, 1, 1, 1]),
        2,
    )
    assert distance.loss.item() < 1e-9
    assert distance.response.item() == 0.0
