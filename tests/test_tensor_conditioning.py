import copy

import torch

from gaugeflow.production.cartesian_gauge_atlas import CartesianGaugeAtlasOutput
from gaugeflow.production.equivariant_denoiser import (
    CenteredResidualAdapter,
    HybridCrystalDenoiser,
    HybridDenoiserOutput,
)
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


def test_centered_residual_is_zero_but_active_path_has_immediate_gradient() -> None:
    adapter = CenteredResidualAdapter(8)
    value = torch.randn((3, 8), generator=torch.Generator().manual_seed(21))
    delta = adapter(value)
    assert torch.equal(delta, torch.zeros_like(delta))
    probe = torch.randn_like(delta)
    (delta * probe).sum().backward()
    active_gradients = [parameter.grad for parameter in adapter.active.parameters()]
    reference_gradients = [parameter.grad for parameter in adapter.reference.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in active_gradients)
    assert sum(float(gradient.square().sum()) for gradient in active_gradients if gradient is not None) > 0.0
    assert all(gradient is None for gradient in reference_gradients)


def test_centered_residual_preserves_null_branch_after_active_update() -> None:
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
    ).eval()
    reference_model = copy.deepcopy(model).eval()
    model.attach_tensor_residual_adapter()
    with torch.no_grad():
        model.tensor_residual_adapter.active[-1].bias.add_(0.25)
    tokens = torch.tensor([4, 6, 12, 15], dtype=torch.long)
    coordinates = torch.tensor(
        [[0.05, 0.10, 0.15], [0.35, 0.25, 0.70], [0.15, 0.75, 0.45], [0.72, 0.55, 0.20]]
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    log_volume = torch.log(torch.tensor([64.0, 91.125]))
    log_shape = torch.zeros((2, 6))
    time = torch.full((2,), 0.4)
    condition = torch.randn((2, 18), generator=torch.Generator().manual_seed(22))
    projector = torch.eye(6).expand(2, -1, -1).clone()
    chart = torch.eye(3).expand(2, -1, -1).clone()
    null = torch.zeros((2, 1), dtype=torch.bool)
    present = torch.ones((2, 1), dtype=torch.bool)
    null_base = reference_model(
        tokens, coordinates, log_volume, log_shape, batch, time, condition, null, projector, chart
    )
    null_adapted = model(
        tokens, coordinates, log_volume, log_shape, batch, time, condition, null, projector, chart
    )
    torch.testing.assert_close(
        null_adapted.coordinate_fractional_scaled_score,
        null_base.coordinate_fractional_scaled_score,
        atol=0.0,
        rtol=0.0,
    )
    present_adapted = model(
        tokens, coordinates, log_volume, log_shape, batch, time, condition, present, projector, chart
    )
    present_base = reference_model(
        tokens, coordinates, log_volume, log_shape, batch, time, condition, present, projector, chart
    )
    assert not torch.equal(
        present_adapted.coordinate_fractional_scaled_score,
        present_base.coordinate_fractional_scaled_score,
    )


def test_attached_adapter_present_branch_matches_stage_c_null_at_initialization() -> None:
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        modality_time_conditioning="separate",
    ).eval()
    reference_model = copy.deepcopy(model).eval()
    model.attach_tensor_residual_adapter()
    tokens = torch.tensor([4, 6, 12, 15], dtype=torch.long)
    coordinates = torch.tensor(
        [[0.05, 0.10, 0.15], [0.35, 0.25, 0.70], [0.15, 0.75, 0.45], [0.72, 0.55, 0.20]]
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    log_volume = torch.log(torch.tensor([64.0, 91.125]))
    log_shape = torch.zeros((2, 6))
    time = torch.full((2,), 0.4)
    condition = torch.randn((2, 18), generator=torch.Generator().manual_seed(23))
    projector = torch.eye(6).expand(2, -1, -1).clone()
    chart = torch.eye(3).expand(2, -1, -1).clone()
    null = torch.zeros((2, 1), dtype=torch.bool)
    present = torch.ones((2, 1), dtype=torch.bool)
    expected = reference_model(
        tokens,
        coordinates,
        log_volume,
        log_shape,
        batch,
        time,
        condition,
        null,
        projector,
        chart,
        element_time=time,
        lattice_time=time,
    )
    actual = model(
        tokens,
        coordinates,
        log_volume,
        log_shape,
        batch,
        time,
        condition,
        present,
        projector,
        chart,
        element_time=time,
        lattice_time=time,
    )
    torch.testing.assert_close(
        actual.coordinate_fractional_scaled_score,
        expected.coordinate_fractional_scaled_score,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(actual.clean_volume_latent, expected.clean_volume_latent, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual.clean_shape_latent, expected.clean_shape_latent, atol=0.0, rtol=0.0)
    lattice_expected = reference_model.forward_lattice(
        tokens,
        log_volume,
        log_shape,
        batch,
        time,
        projector,
        tensor_condition=condition,
        condition_present=null,
    )
    lattice_actual = model.forward_lattice(
        tokens,
        log_volume,
        log_shape,
        batch,
        time,
        projector,
        tensor_condition=condition,
        condition_present=present,
    )
    torch.testing.assert_close(
        lattice_actual.clean_volume_latent,
        lattice_expected.clean_volume_latent,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        lattice_actual.clean_shape_latent,
        lattice_expected.clean_shape_latent,
        atol=0.0,
        rtol=0.0,
    )
    actual.clean_volume_latent.sum().backward()
    active_gradients = [parameter.grad for parameter in model.tensor_residual_adapter.active.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in active_gradients)
    assert sum(float(gradient.square().sum()) for gradient in active_gradients if gradient is not None) > 0.0
