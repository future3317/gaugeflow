from __future__ import annotations

import torch

from gaugeflow.production.cartesian_coordinate_carrier import (
    CompactCartesianKrylovCarrier,
    StateAdaptiveCartesianCarrierMixer,
)


def _inputs() -> tuple[
    CompactCartesianKrylovCarrier,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(151)
    module = CompactCartesianKrylovCarrier(12, 5, moment_channels=4)
    nodes, edges = 7, 18
    vectors = torch.randn((nodes, 5, 3), generator=generator)
    hidden = torch.randn((edges, 12), generator=generator)
    target = torch.randint(nodes, (edges,), generator=generator)
    directions = torch.randn((edges, 3), generator=generator)
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    )
    envelope = torch.rand((edges, 1), generator=generator)
    edge_order = torch.argsort(target, stable=True)
    hidden = hidden[edge_order]
    target = target[edge_order]
    directions = directions[edge_order]
    envelope = envelope[edge_order]
    batch = torch.zeros(nodes, dtype=torch.long)
    return module, vectors, hidden, target, directions, envelope, batch


def test_compact_carrier_is_o3_covariant_including_reflection() -> None:
    module, vectors, hidden, target, directions, envelope, batch = _inputs()
    reference = module(vectors, hidden, target, directions, envelope, batch, 1)
    rotation, _ = torch.linalg.qr(
        torch.randn((3, 3), generator=torch.Generator().manual_seed(157))
    )
    rotation[:, 0] *= -1.0
    rotated = module(
        vectors @ rotation,
        hidden,
        target,
        directions @ rotation,
        envelope,
        batch,
        1,
    )
    torch.testing.assert_close(
        rotated, reference @ rotation, atol=2e-5, rtol=2e-5
    )


def test_compact_carrier_is_node_permutation_equivariant_and_horizontal() -> None:
    module, vectors, hidden, target, directions, envelope, batch = _inputs()
    reference = module(vectors, hidden, target, directions, envelope, batch, 1)
    order = torch.randperm(
        vectors.shape[0], generator=torch.Generator().manual_seed(163)
    )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    permuted_target = inverse[target]
    edge_order = torch.argsort(permuted_target, stable=True)
    permuted = module(
        vectors[order],
        hidden[edge_order],
        permuted_target[edge_order],
        directions[edge_order],
        envelope[edge_order],
        batch,
        1,
    )
    torch.testing.assert_close(
        permuted, reference[order], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        permuted.mean(0),
        torch.zeros_like(permuted.mean(0)),
        atol=2e-6,
        rtol=0,
    )


def test_compact_carrier_is_finite_and_differentiable_at_zero() -> None:
    module, vectors, hidden, target, directions, envelope, batch = _inputs()
    vectors = torch.zeros_like(vectors, requires_grad=True)
    hidden = torch.zeros_like(hidden, requires_grad=True)
    output = module(vectors, hidden, target, directions, envelope, batch, 1)
    output.square().sum().backward()
    assert output.shape == (vectors.shape[0], module.output_channels, 3)
    assert torch.isfinite(output).all()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_state_adaptive_mixer_has_small_orthogonal_initial_residual() -> None:
    generator = torch.Generator().manual_seed(167)
    mixer = StateAdaptiveCartesianCarrierMixer(9, 12, rank=3)
    carrier = torch.randn((7, 9, 3), generator=generator)
    state = torch.randn((7, 12), generator=generator)
    observed = mixer(carrier, state)
    expected = torch.einsum("c,ncd->nd", mixer.base_weight, carrier)
    weight = mixer.carrier_projection.weight.detach()
    torch.testing.assert_close(
        weight.T @ weight,
        torch.eye(3) * 1.0e-4,
        atol=2e-11,
        rtol=2e-6,
    )
    relative_change = torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(
        expected
    )
    assert 0.0 < float(relative_change) < 0.05


def test_state_adaptive_mixer_preserves_o3_and_node_permutation_covariance() -> None:
    generator = torch.Generator().manual_seed(173)
    mixer = StateAdaptiveCartesianCarrierMixer(9, 12, rank=3)
    with torch.no_grad():
        mixer.carrier_projection.weight.normal_(generator=generator)
    carrier = torch.randn((7, 9, 3), generator=generator)
    state = torch.randn((7, 12), generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), generator=generator))
    rotation[:, 0] *= -1.0
    order = torch.randperm(7, generator=generator)
    reference = mixer(carrier, state)
    transformed = mixer((carrier @ rotation)[order], state[order])
    torch.testing.assert_close(
        transformed, (reference @ rotation)[order], atol=2e-6, rtol=2e-6
    )


def test_state_adaptive_mixer_has_finite_trainable_low_rank_path() -> None:
    generator = torch.Generator().manual_seed(179)
    mixer = StateAdaptiveCartesianCarrierMixer(9, 12, rank=3)
    carrier = torch.randn((7, 9, 3), generator=generator, requires_grad=True)
    state = torch.randn((7, 12), generator=generator, requires_grad=True)
    mixer(carrier, state).square().mean().backward()
    assert mixer.carrier_projection.weight.grad is not None
    assert torch.isfinite(mixer.carrier_projection.weight.grad).all()
    assert float(mixer.carrier_projection.weight.grad.norm()) > 0.0
    assert mixer.state_projection.weight.grad is not None
    assert torch.isfinite(mixer.state_projection.weight.grad).all()
    assert float(mixer.state_projection.weight.grad.norm()) > 0.0


def test_production_model_contains_only_the_adaptive_compact_coordinate_readout() -> None:
    from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser

    model = HybridCrystalDenoiser()
    names = tuple(name for name, _ in model.named_parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_264_431
    assert not any("coordinate_vector_head" in name for name in names)
    assert not any("coordinate_edge_head" in name for name in names)
    assert not any("coordinate_carrier_head" in name for name in names)
    assert model.coordinate_carrier.output_channels == 80
    assert model.coordinate_carrier_mixer.rank == 8
    assert model.edge_dim == 64
    assert model.angular_channels == 8
    assert model.edge_refresh_rank == 16
    for block in model.blocks:
        assert block.angular_moments.channels == 8
        assert block.edge_refresh_rank == 16
        assert torch.count_nonzero(block.angular_scalar_residual[-1].weight) > 0
        assert torch.count_nonzero(block.angular_vector_residual[-1].weight) > 0
        assert hasattr(block, "edge_source_refresh")
        assert hasattr(block, "edge_target_refresh")
        assert hasattr(block, "edge_context_refresh")
        assert hasattr(block, "edge_vector_refresh")
    assert torch.count_nonzero(model.coordinate_edge_residual[-1].weight) > 0
