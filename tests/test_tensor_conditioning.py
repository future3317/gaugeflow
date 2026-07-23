import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.cartesian_gauge_atlas import CartesianGaugeAtlasOutput
from gaugeflow.production.equivariant_denoiser import (
    CenteredResidualAdapter,
    HybridCrystalDenoiser,
    HybridDenoiserOutput,
    LatticeResidualAdapter,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.reverse_sampler import TensorFreeReverseSampler
from gaugeflow.production.tensor_conditioning import (
    null_condition_retention_distance,
    orbit_mimic_distance,
    rotate_orbit_representative,
    sample_proper_rotations,
)
from gaugeflow.tensor import piezo_from_irreps
from scripts.train_stage_e_lattice_generated_exposure import _exact_composition_counts, _generated_exposure_loss


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


def _standardizer() -> P1LatticeStandardizer:
    return P1LatticeStandardizer.from_json(
        Path(__file__).parents[1] / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def _small_stage_e_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ParentBlueprintBatch]:
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2, 3]))
    elements = torch.tensor([4, 6, 12, 15, 6], dtype=torch.long)
    coordinates = torch.tensor(
        [
            [0.05, 0.10, 0.15],
            [0.35, 0.25, 0.70],
            [0.15, 0.75, 0.45],
            [0.72, 0.55, 0.20],
            [0.42, 0.31, 0.82],
        ]
    )
    lattice = torch.stack((3.0 * torch.eye(3), torch.diag(torch.tensor([3.5, 4.0, 4.5]))))
    return elements, coordinates, lattice, blueprint


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


def test_lattice_residual_adapter_is_exact_zero_and_has_gradients() -> None:
    adapter = LatticeResidualAdapter(hidden_dim=8)
    context = torch.randn((3, 8), generator=torch.Generator().manual_seed(29))
    volume = torch.randn(3, generator=torch.Generator().manual_seed(30))
    shape = torch.randn((3, 5), generator=torch.Generator().manual_seed(31))
    actual_volume, actual_shape = adapter(context, volume, shape)
    torch.testing.assert_close(actual_volume, volume, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_shape, shape, atol=0.0, rtol=0.0)
    loss = actual_volume.square().mean() + actual_shape.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in adapter.active.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.square().sum()) for gradient in gradients if gradient is not None) > 0.0
    assert all(parameter.grad is None for parameter in adapter.reference.parameters())


def test_generated_lattice_exposure_uses_exact_composition_context() -> None:
    elements, _, lattice, blueprint = _small_stage_e_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        modality_time_conditioning="separate",
    )
    model.attach_lattice_residual_adapter()
    diffusion = TensorFreeHybridDiffusion(
        model,
        _standardizer(),
    )
    clean = SimpleNamespace(
        element_tokens=elements,
        lattice=lattice,
        batch=blueprint.batch,
        node_counts=blueprint.node_counts,
    )
    captured: list[torch.Tensor | None] = []
    original = model.forward_lattice

    def capture(*args, **kwargs):
        counts = kwargs.get("composition_counts")
        captured.append(None if counts is None else counts.detach().clone())
        return original(*args, **kwargs)

    model.forward_lattice = capture  # type: ignore[method-assign]
    try:
        loss, _ = _generated_exposure_loss(
            diffusion,
            clean,
            exposure_time=0.5,
            exposure_delta=0.1,
            generator=torch.Generator().manual_seed(34),
            precision="fp32",
        )
    finally:
        model.forward_lattice = original  # type: ignore[method-assign]
    assert torch.isfinite(loss)
    expected = _exact_composition_counts(
        elements,
        blueprint.batch,
        int(blueprint.node_counts.numel()),
        diffusion.categorical.element_count,
    )
    assert len(captured) == 3
    assert all(counts is not None and torch.equal(counts, expected) for counts in captured)
    assert torch.equal(expected.sum(dim=1), blueprint.node_counts)
    assert expected[0, 12] == 0
    assert expected[1, 4] == 0

    relabeled = elements.clone()
    relabeled[elements == 4] = 7
    relabeled_counts = _exact_composition_counts(
        relabeled,
        blueprint.batch,
        int(blueprint.node_counts.numel()),
        diffusion.categorical.element_count,
    )
    assert relabeled_counts[0, 4] == 0
    assert relabeled_counts[0, 7] == expected[0, 4]
    assert torch.equal(relabeled_counts.sum(dim=1), blueprint.node_counts)


def test_lattice_residual_shape_scale_preserves_volume_channel() -> None:
    generator = torch.Generator().manual_seed(91)
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        modality_time_conditioning="separate",
    )
    model.attach_lattice_residual_adapter()
    assert model.lattice_residual_adapter is not None
    with torch.no_grad():
        model.lattice_residual_adapter.active[0].bias.add_(0.25)
        model.lattice_residual_adapter.volume_head.weight.fill_(0.5)
        model.lattice_residual_adapter.shape_head.weight.fill_(0.25)

    context = torch.randn((3, 4 * model.hidden_dim), generator=generator)
    volume = torch.randn((3,), generator=generator)
    shape = torch.randn((3, 5), generator=generator)

    full_volume, full_shape = model._apply_lattice_residual_adapter(context, volume, shape)
    assert not torch.allclose(full_volume, volume)
    assert not torch.allclose(full_shape, shape)

    model.set_lattice_residual_shape_scale(0.0)
    volume_only, shape_zero = model._apply_lattice_residual_adapter(context, volume, shape)
    torch.testing.assert_close(volume_only, full_volume)
    torch.testing.assert_close(shape_zero, shape)

    model.set_lattice_residual_shape_scale(0.25)
    scaled_volume, scaled_shape = model._apply_lattice_residual_adapter(context, volume, shape)
    torch.testing.assert_close(scaled_volume, full_volume)
    torch.testing.assert_close(scaled_shape - shape, 0.25 * (full_shape - shape))

    with pytest.raises(ValueError, match="finite and nonnegative"):
        model.set_lattice_residual_shape_scale(-0.1)


def test_free_generated_side_uses_sampled_counts_not_clean_target_counts() -> None:
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2, 3]))
    sampled_counts = torch.zeros((2, 118), dtype=torch.long)
    sampled_counts[0, 5] = 1
    sampled_counts[0, 8] = 1
    sampled_counts[1, 5] = 2
    sampled_counts[1, 9] = 1
    clean_target_counts = torch.zeros_like(sampled_counts)
    clean_target_counts[0, 4] = 1
    clean_target_counts[0, 6] = 1
    clean_target_counts[1, 6] = 1
    clean_target_counts[1, 12] = 1
    clean_target_counts[1, 15] = 1

    class DenseState:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def to_dense(self, vocabulary_size: int) -> torch.Tensor:
            if vocabulary_size != self.value.shape[1]:
                raise ValueError("unexpected vocabulary size")
            return self.value.clone()

    class FixedCompositionModel:
        context_dim = 1
        maximum_atoms = 20
        vocabulary_size = 118

        def sample(
            self,
            context: torch.Tensor,
            node_counts: torch.Tensor,
            *,
            generator: torch.Generator | None = None,
        ) -> Any:
            del generator
            assert context.shape == (2, 1)
            assert torch.equal(node_counts, blueprint.node_counts)
            return SimpleNamespace(state=DenseState(sampled_counts.to(context.device)))

    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        modality_time_conditioning="separate",
    )
    seen: list[torch.Tensor] = []
    original = model.forward

    def capture(*args, **kwargs):
        counts = kwargs.get("composition_counts")
        assert isinstance(counts, torch.Tensor)
        seen.append(counts.detach().cpu().clone())
        return original(*args, **kwargs)

    model.forward = capture  # type: ignore[method-assign]
    sampler = TensorFreeReverseSampler(
        model,
        _standardizer(),
        maximum_time=0.8,
        categorical_path="orderless_reveal",
        composition_model=cast(Any, FixedCompositionModel()),
    )
    try:
        generated = sampler.sample(
            blueprint,
            steps=2,
            categorical_generator=torch.Generator().manual_seed(35),
            continuous_mode="probability_flow",
        )
    finally:
        model.forward = original  # type: ignore[method-assign]
    assert seen
    assert all(torch.equal(counts, sampled_counts) for counts in seen)
    assert all(not torch.equal(counts, clean_target_counts) for counts in seen)
    assert torch.equal(generated.composition_counts.cpu(), sampled_counts)
