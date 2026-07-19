from __future__ import annotations

import torch

from gaugeflow.production.edge_query_angular_kernel import (
    InducedEdgeQueryAngularKernel,
    ShellCompleteTopKTripletKernel,
    shell_complete_nearest_neighbors,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(211)
    target = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    distance = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 0.8, 1.1, 1.4, 1.8])
    state = torch.randn((target.numel(), 7), generator=generator, dtype=torch.float64)
    direction = torch.randn((target.numel(), 3), generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    envelope = torch.rand((target.numel(), 1), generator=generator, dtype=torch.float64)
    return state, target, direction, envelope, distance


def test_shell_complete_topk_never_cuts_an_equal_distance_shell() -> None:
    _, target, _, _, distance = _inputs()
    neighbors = shell_complete_nearest_neighbors(distance, target, 2, k=3)
    assert neighbors.selected_count.tolist() == [4, 3]
    assert neighbors.valid.sum(dim=-1).tolist() == [4, 3]


def test_explicit_triplet_kernel_is_o3_and_edge_permutation_invariant() -> None:
    state, target, direction, envelope, distance = _inputs()
    module = ShellCompleteTopKTripletKernel(7, 3, k=3).double()
    neighbors = shell_complete_nearest_neighbors(distance, target, 2, k=3)
    reference = module(state, target, direction, envelope, neighbors)

    generator = torch.Generator().manual_seed(223)
    orthogonal, _ = torch.linalg.qr(
        torch.randn((3, 3), generator=generator, dtype=torch.float64)
    )
    orthogonal[:, 0] *= -1.0
    rotated = module(state, target, direction @ orthogonal, envelope, neighbors)
    torch.testing.assert_close(rotated, reference, atol=2e-12, rtol=2e-12)

    edge_order = torch.tensor([3, 0, 4, 1, 2, 8, 5, 7, 6])
    permuted_neighbors = shell_complete_nearest_neighbors(
        distance[edge_order], target[edge_order], 2, k=3
    )
    permuted = module(
        state[edge_order],
        target[edge_order],
        direction[edge_order],
        envelope[edge_order],
        permuted_neighbors,
    )
    torch.testing.assert_close(permuted, reference[edge_order], atol=2e-12, rtol=2e-12)


def test_induced_slots_are_o3_and_edge_permutation_invariant() -> None:
    state, target, direction, envelope, _ = _inputs()
    module = InducedEdgeQueryAngularKernel(7, 3, slots=8).double()
    reference = module(state, target, direction, envelope, 2)
    generator = torch.Generator().manual_seed(227)
    orthogonal, _ = torch.linalg.qr(
        torch.randn((3, 3), generator=generator, dtype=torch.float64)
    )
    orthogonal[:, 0] *= -1.0
    rotated = module(state, target, direction @ orthogonal, envelope, 2)
    torch.testing.assert_close(rotated, reference, atol=2e-12, rtol=2e-12)

    edge_order = torch.tensor([2, 4, 0, 3, 1, 7, 5, 8, 6])
    permuted = module(
        state[edge_order],
        target[edge_order],
        direction[edge_order],
        envelope[edge_order],
        2,
    )
    torch.testing.assert_close(permuted, reference[edge_order], atol=2e-12, rtol=2e-12)


def test_induced_slots_have_normalized_noncollapsed_assignments_and_gradients() -> None:
    state, target, direction, envelope, _ = _inputs()
    state.requires_grad_(True)
    direction.requires_grad_(True)
    module = InducedEdgeQueryAngularKernel(7, 3, slots=8).double()
    probability = module.assignment_probabilities(state)
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.ones(state.shape[0], dtype=torch.float64)
    )
    assert float(probability.std()) > 0.0
    output = module(state, target, direction, envelope, 2)
    output.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert direction.grad is not None and torch.isfinite(direction.grad).all()
    assert module.assignment.weight.grad is not None
    assert float(module.assignment.weight.grad.norm()) > 0.0
    statistics = module.slot_statistics(state, target, direction, envelope, 2)
    assert statistics.mass.shape == (2, 8)
    assert statistics.scalar.shape == (2, 8, 3)
    assert statistics.vector.shape == (2, 8, 3, 3)
    assert statistics.stf2.shape == (2, 8, 3, 6)
    assert torch.isfinite(statistics.mass).all()


def test_induced_assignment_reads_current_layer_context_on_first_backward() -> None:
    from gaugeflow.production.equivariant_denoiser import EquivariantDenoisingBlock

    generator = torch.Generator().manual_seed(229)
    block = EquivariantDenoisingBlock(
        hidden_dim=12,
        vector_dim=4,
        radial_dim=5,
        edge_dim=7,
        angular_channels=3,
        edge_refresh_rank=3,
        angular_operator="induced_slots",
        angular_slots=8,
        triplet_k=3,
    ).double()
    source = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    target = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    direction = torch.randn((6, 3), generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    inputs = {
        "nodes": torch.randn((3, 12), generator=generator, dtype=torch.float64),
        "vectors": torch.randn((3, 4, 3), generator=generator, dtype=torch.float64),
        "edge_response": torch.randn((6, 3), generator=generator, dtype=torch.float64),
        "radial": torch.randn((6, 5), generator=generator, dtype=torch.float64),
        "edge_envelope": torch.rand((6, 1), generator=generator, dtype=torch.float64),
        "node_time": torch.randn((3, 12), generator=generator, dtype=torch.float64),
        "node_condition": torch.randn((3, 12), generator=generator, dtype=torch.float64),
        "node_state": torch.randn((3, 12), generator=generator, dtype=torch.float64),
        "edge_state": torch.randn((6, 7), generator=generator, dtype=torch.float64),
    }
    nodes, vectors, edge_state = block(
        inputs["nodes"],
        inputs["vectors"],
        source,
        target,
        direction,
        inputs["edge_response"],
        inputs["radial"],
        inputs["edge_envelope"],
        inputs["node_time"],
        inputs["node_condition"],
        inputs["node_state"],
        inputs["edge_state"],
        None,
    )
    (nodes.square().mean() + vectors.square().mean() + edge_state.square().mean()).backward()
    assert block.induced_assignment_refresh is not None
    context_gradient = block.induced_assignment_refresh[-1].weight.grad
    assert context_gradient is not None and float(context_gradient.norm()) > 0.0
    assert isinstance(block.angular_moments, InducedEdgeQueryAngularKernel)
    for parameter in (
        block.angular_moments.assignment.weight,
        block.angular_moments.coefficients.weight,
        block.angular_moments.query_gate.weight,
    ):
        assert parameter.grad is not None and float(parameter.grad.norm()) > 0.0
