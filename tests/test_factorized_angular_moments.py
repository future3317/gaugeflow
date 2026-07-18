from __future__ import annotations

import torch

from gaugeflow.production.factorized_angular_moments import (
    FactorizedCartesianAngularMoments,
)


def _inputs() -> tuple[
    FactorizedCartesianAngularMoments,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    generator = torch.Generator().manual_seed(181)
    module = FactorizedCartesianAngularMoments(7, 3).double()
    node_count = 5
    target = torch.tensor([0, 0, 0, 1, 1, 3, 3, 3, 3], dtype=torch.long)
    state = torch.randn((target.numel(), 7), generator=generator, dtype=torch.float64)
    direction = torch.randn((target.numel(), 3), generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    envelope = torch.rand((target.numel(), 1), generator=generator, dtype=torch.float64)
    return module, state, target, direction, envelope, node_count


def _explicit_triplet_reference(
    module: FactorizedCartesianAngularMoments,
    state: torch.Tensor,
    target: torch.Tensor,
    direction: torch.Tensor,
    envelope: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    coefficients = torch.tanh(module.coefficient_projection(state))
    first, second = coefficients.split(module.channels, dim=-1)
    output = state.new_zeros((state.shape[0], 2 * module.channels))
    for edge in range(state.shape[0]):
        center = int(target[edge])
        selected = torch.nonzero(target == center, as_tuple=False).squeeze(-1)
        scale = float(selected.numel()) ** -0.5
        cosines = direction[selected] @ direction[edge]
        output[edge, : module.channels] = scale * (
            first[selected]
            * envelope[selected]
            * cosines[:, None]
        ).sum(dim=0)
        output[edge, module.channels :] = scale * (
            second[selected]
            * envelope[selected]
            * (cosines.square() - 1.0 / 3.0)[:, None]
        ).sum(dim=0)
    return output


def test_factorized_moments_match_explicit_low_order_triplet_sum() -> None:
    module, state, target, direction, envelope, node_count = _inputs()
    observed = module(state, target, direction, envelope, node_count)
    expected = _explicit_triplet_reference(
        module, state, target, direction, envelope, node_count
    )
    torch.testing.assert_close(observed, expected, atol=2e-12, rtol=2e-12)


def test_factorized_moments_are_o3_invariant_including_reflection() -> None:
    module, state, target, direction, envelope, node_count = _inputs()
    generator = torch.Generator().manual_seed(191)
    orthogonal, _ = torch.linalg.qr(
        torch.randn((3, 3), generator=generator, dtype=torch.float64)
    )
    orthogonal[:, 0] *= -1.0
    reference = module(state, target, direction, envelope, node_count)
    transformed = module(
        state, target, direction @ orthogonal, envelope, node_count
    )
    torch.testing.assert_close(transformed, reference, atol=2e-12, rtol=2e-12)


def test_factorized_moments_are_node_and_edge_permutation_equivariant() -> None:
    module, state, target, direction, envelope, node_count = _inputs()
    generator = torch.Generator().manual_seed(193)
    node_order = torch.randperm(node_count, generator=generator)
    inverse = torch.empty_like(node_order)
    inverse[node_order] = torch.arange(node_count)
    permuted_target = inverse[target]
    edge_order = torch.argsort(permuted_target, stable=True)
    reference = module(state, target, direction, envelope, node_count)
    observed = module(
        state[edge_order],
        permuted_target[edge_order],
        direction[edge_order],
        envelope[edge_order],
        node_count,
    )
    torch.testing.assert_close(
        observed, reference[edge_order], atol=2e-12, rtol=2e-12
    )


def test_factorized_moments_have_finite_state_and_direction_gradients() -> None:
    module, state, target, direction, envelope, node_count = _inputs()
    state.requires_grad_(True)
    direction.requires_grad_(True)
    output = module(state, target, direction, envelope, node_count)
    output.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert direction.grad is not None and torch.isfinite(direction.grad).all()
    assert module.coefficient_projection.weight.grad is not None
    assert torch.isfinite(module.coefficient_projection.weight.grad).all()


def test_factorized_moments_use_edge_linear_output_storage() -> None:
    module, state, target, direction, envelope, node_count = _inputs()
    doubled_target = torch.cat((target, target))
    edge_order = torch.argsort(doubled_target, stable=True)
    doubled = module(
        torch.cat((state, state))[edge_order],
        doubled_target[edge_order],
        torch.cat((direction, direction))[edge_order],
        torch.cat((envelope, envelope))[edge_order],
        node_count,
    )
    assert doubled.shape == (2 * state.shape[0], module.output_dim)
    assert not hasattr(module, "triplet_index")
