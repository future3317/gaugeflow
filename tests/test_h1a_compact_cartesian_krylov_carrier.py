from __future__ import annotations

import torch

from gaugeflow.geometry import PeriodicEdges
from scripts.audit_h1a_compact_cartesian_krylov_carrier import (
    compact_cartesian_krylov_carrier,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, PeriodicEdges, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(131)
    nodes, edges, hidden = 7, 18, 12
    vector = torch.randn((nodes, 5, 3), generator=generator)
    edge_hidden = torch.randn((edges, hidden), generator=generator)
    source = torch.randint(nodes, (edges,), generator=generator)
    target = torch.randint(nodes, (edges,), generator=generator)
    direction = torch.randn((edges, 3), generator=generator)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    periodic = PeriodicEdges(
        source=source,
        target=target,
        displacement=direction,
        direction=direction,
        distance=torch.ones(edges),
        image_shift=torch.zeros((edges, 3), dtype=torch.long),
    )
    envelope = torch.rand((edges, 1), generator=generator)
    projection, _ = torch.linalg.qr(torch.randn((hidden, 8), generator=generator))
    return vector, edge_hidden, periodic, envelope, projection


def test_carrier_is_o3_covariant_including_reflection() -> None:
    vector, edge_hidden, edges, envelope, projection = _inputs()
    batch = torch.zeros(vector.shape[0], dtype=torch.long)
    reference = compact_cartesian_krylov_carrier(
        vector, edge_hidden, edges, envelope, batch, 1, projection, 1e-4
    )
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), generator=torch.Generator().manual_seed(137)))
    rotation[:, 0] *= -1.0
    rotated_edges = PeriodicEdges(
        source=edges.source,
        target=edges.target,
        displacement=edges.displacement @ rotation,
        direction=edges.direction @ rotation,
        distance=edges.distance,
        image_shift=edges.image_shift,
    )
    rotated = compact_cartesian_krylov_carrier(
        vector @ rotation,
        edge_hidden,
        rotated_edges,
        envelope,
        batch,
        1,
        projection,
        1e-4,
    )
    torch.testing.assert_close(rotated, reference @ rotation, atol=2e-5, rtol=2e-5)


def test_carrier_is_node_permutation_equivariant_and_horizontal() -> None:
    vector, edge_hidden, edges, envelope, projection = _inputs()
    nodes = vector.shape[0]
    batch = torch.zeros(nodes, dtype=torch.long)
    reference = compact_cartesian_krylov_carrier(
        vector, edge_hidden, edges, envelope, batch, 1, projection, 1e-4
    )
    order = torch.randperm(nodes, generator=torch.Generator().manual_seed(139))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(nodes)
    permuted_edges = PeriodicEdges(
        source=inverse[edges.source],
        target=inverse[edges.target],
        displacement=edges.displacement,
        direction=edges.direction,
        distance=edges.distance,
        image_shift=edges.image_shift,
    )
    permuted = compact_cartesian_krylov_carrier(
        vector[order], edge_hidden, permuted_edges, envelope, batch, 1, projection, 1e-4
    )
    torch.testing.assert_close(permuted, reference[order], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(permuted.mean(0), torch.zeros_like(permuted.mean(0)), atol=2e-6, rtol=0)


def test_carrier_is_finite_and_differentiable_at_zero() -> None:
    vector, edge_hidden, edges, envelope, projection = _inputs()
    vector = torch.zeros_like(vector, requires_grad=True)
    edge_hidden = torch.zeros_like(edge_hidden, requires_grad=True)
    batch = torch.zeros(vector.shape[0], dtype=torch.long)
    carrier = compact_cartesian_krylov_carrier(
        vector, edge_hidden, edges, envelope, batch, 1, projection, 1e-4
    )
    carrier.square().sum().backward()
    assert torch.isfinite(carrier).all()
    assert vector.grad is not None and torch.isfinite(vector.grad).all()
    assert edge_hidden.grad is not None and torch.isfinite(edge_hidden.grad).all()
