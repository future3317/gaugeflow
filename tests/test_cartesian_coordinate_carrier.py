from __future__ import annotations

import torch

from gaugeflow.production.cartesian_coordinate_carrier import (
    CompactCartesianKrylovCarrier,
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
    permuted = module(
        vectors[order],
        hidden,
        inverse[target],
        directions,
        envelope,
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


def test_production_model_contains_only_the_compact_coordinate_readout() -> None:
    from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser

    model = HybridCrystalDenoiser()
    names = tuple(name for name, _ in model.named_parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_479_161
    assert not any("coordinate_vector_head" in name for name in names)
    assert not any("coordinate_edge_head" in name for name in names)
    assert model.coordinate_carrier.output_channels == 80
