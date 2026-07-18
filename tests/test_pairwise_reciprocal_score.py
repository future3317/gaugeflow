import ast
import inspect
import math

import torch

from gaugeflow.production.pairwise_reciprocal_score import (
    PairwiseReciprocalScore,
    complete_unordered_node_pairs,
    projective_reciprocal_ball,
)


def _input(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(8701)
    nodes = torch.randn((7, 12), dtype=dtype, generator=generator)
    coordinates = torch.rand((7, 3), dtype=dtype, generator=generator)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    lattice = torch.tensor(
        [
            [[4.1, 0.2, 0.1], [0.4, 3.8, 0.3], [0.2, 0.5, 4.5]],
            [[3.7, 0.1, 0.2], [0.3, 4.4, 0.1], [0.4, 0.2, 3.9]],
        ],
        dtype=dtype,
    )
    return nodes, coordinates, lattice, batch


def _active_head(dtype: torch.dtype = torch.float64) -> PairwiseReciprocalScore:
    torch.manual_seed(8702)
    head = PairwiseReciprocalScore(
        12, pair_width=8, channels=4, radial_dim=6, cutoff=4.0
    ).to(dtype=dtype)
    torch.nn.init.normal_(head.mode_channels[-1].weight, std=0.15)
    torch.nn.init.normal_(head.mode_channels[-1].bias, std=0.05)
    return head


def _full_symmetric_ball_reference(
    head: PairwiseReciprocalScore,
    nodes: torch.Tensor,
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    graphs = lattice.shape[0]
    pairs = complete_unordered_node_pairs(batch, graphs)
    ball = projective_reciprocal_ball(lattice, head.radial.cutoff)
    projected = head.node_projection(nodes)
    first, second = projected[pairs.first], projected[pairs.second]
    pair_channels = head.pair_channels(
        torch.cat((first + second, first * second), dim=-1)
    )
    radial = head.radial(ball.norms.reshape(-1)).reshape(
        graphs, ball.norms.shape[1], -1
    )
    mode_channels = head.mode_channels(radial)
    integer_modes = torch.cat((ball.integer_modes, -ball.integer_modes), dim=1)
    cartesian_modes = torch.cat(
        (ball.cartesian_covectors, -ball.cartesian_covectors), dim=1
    )
    mode_channels = torch.cat((mode_channels, mode_channels), dim=1)
    mask = torch.cat((ball.mask, ball.mask), dim=1)
    phase = 2.0 * math.pi * torch.einsum(
        "pi,pki->pk",
        coordinates[pairs.first] - coordinates[pairs.second],
        integer_modes[pairs.graph].to(coordinates),
    )
    coefficient = torch.einsum(
        "pc,pkc->pk", pair_channels, mode_channels[pairs.graph]
    ) / math.sqrt(head.channels)
    coefficient = coefficient * phase.sin() * mask[pairs.graph].to(coefficient)
    pair_score = torch.einsum(
        "pk,pki->pi", coefficient, cartesian_modes[pairs.graph]
    )
    counts = torch.bincount(batch, minlength=graphs).to(pair_score)
    full_mode_counts = mask.sum(-1).clamp_min(1).to(pair_score)
    pair_score = pair_score / (
        counts[pairs.graph] * full_mode_counts[pairs.graph]
    ).sqrt().unsqueeze(-1)
    result = coordinates.new_zeros(coordinates.shape)
    result.index_add_(0, pairs.first, pair_score)
    result.index_add_(0, pairs.second, -pair_score)
    return result


def test_pair_builder_is_complete_and_source_has_no_python_loops():
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    pairs = complete_unordered_node_pairs(batch, 2)
    observed = set(zip(pairs.first.tolist(), pairs.second.tolist(), strict=True))
    assert observed == {(0, 1), (0, 2), (1, 2), (3, 4)}
    tree = ast.parse(inspect.getsource(__import__(
        "gaugeflow.production.pairwise_reciprocal_score", fromlist=["*"]
    )))
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))


def test_pairwise_reciprocal_residual_is_exactly_zero_at_initialization():
    values = _input()
    head = PairwiseReciprocalScore(12, pair_width=8, channels=4, radial_dim=6).double()
    assert torch.equal(head(*values), torch.zeros_like(values[1]))


def test_pairwise_reciprocal_symmetries_and_horizontal_projection():
    nodes, coordinates, lattice, batch = _input()
    head = _active_head()
    reference = head(nodes, coordinates, lattice, batch)

    translated = head(
        nodes,
        coordinates + torch.tensor([0.271, -0.193, 0.417], dtype=coordinates.dtype),
        lattice,
        batch,
    )
    integer_shift = torch.tensor(
        [[1, -2, 3], [0, 1, -1], [-3, 2, 0], [2, 0, 1],
         [-1, 1, 2], [3, -2, -1], [0, 4, -3]],
        dtype=coordinates.dtype,
    )
    represented = head(nodes, coordinates + integer_shift, lattice, batch)
    assert torch.allclose(translated, reference, atol=1e-12, rtol=1e-12)
    assert torch.allclose(represented, reference, atol=2e-12, rtol=2e-12)

    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    permuted = head(
        nodes[permutation], coordinates[permutation], lattice, batch
    )
    assert torch.allclose(permuted, reference[permutation], atol=2e-12, rtol=2e-12)
    for graph in range(2):
        assert torch.allclose(
            reference[batch == graph].sum(0), torch.zeros(3, dtype=reference.dtype),
            atol=2e-12, rtol=0.0,
        )


def test_pairwise_reciprocal_o3_and_unimodular_cell_covariance():
    nodes, coordinates, lattice, batch = _input()
    head = _active_head()
    reference = head(nodes, coordinates, lattice, batch)

    matrix = torch.tensor(
        [[0.3, -0.8, 0.5], [0.7, 0.5, 0.4], [-0.6, 0.2, 0.9]],
        dtype=lattice.dtype,
    )
    rotation, _ = torch.linalg.qr(matrix)
    rotation[:, 0] *= -1.0  # include an improper O(3) action
    rotated = head(nodes, coordinates, lattice @ rotation.T, batch)
    assert torch.allclose(rotated, reference @ rotation.T, atol=2e-12, rtol=2e-12)

    basis = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=lattice.dtype,
    )
    inverse = torch.linalg.inv(basis)
    transformed = head(
        nodes,
        coordinates @ inverse,
        basis.unsqueeze(0) @ lattice,
        batch,
    )
    assert torch.allclose(transformed, reference, atol=3e-12, rtol=3e-12)


def test_projective_duplicate_correction_matches_explicit_full_ball():
    values = _input()
    head = _active_head()
    observed = head(*values)
    expected = _full_symmetric_ball_reference(head, *values)
    assert torch.allclose(observed, expected, atol=2e-12, rtol=2e-12)


def test_pairwise_reciprocal_backward_and_cutoff_are_finite_and_smooth():
    nodes, coordinates, lattice, batch = _input()
    nodes.requires_grad_(True)
    coordinates.requires_grad_(True)
    lattice.requires_grad_(True)
    head = _active_head()
    score = head(nodes, coordinates, lattice, batch)
    loss = score.square().mean()
    gradients = torch.autograd.grad(
        loss, (nodes, coordinates, lattice, *tuple(head.parameters()))
    )
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(float(value.abs().max()) > 0.0 for value in gradients[:3])

    radius = torch.tensor(
        [head.radial.cutoff], dtype=torch.float64, requires_grad=True
    )
    envelope = head.radial.envelope(radius).sum()
    derivative = torch.autograd.grad(envelope, radius)[0]
    assert abs(float(envelope)) <= 1e-12
    assert abs(float(derivative)) <= 1e-10
