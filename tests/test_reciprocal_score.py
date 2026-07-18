import torch

from gaugeflow.production.reciprocal_score import (
    ReciprocalStructureFactorScore,
    reciprocal_ball,
)


def _active_head(dtype: torch.dtype = torch.float64) -> ReciprocalStructureFactorScore:
    module = ReciprocalStructureFactorScore(
        12, channels=4, radial_dim=5, cutoff=4.0
    ).to(dtype=dtype)
    generator = torch.Generator().manual_seed(8101)
    with torch.no_grad():
        module.mode_channels[-1].weight.copy_(
            0.2
            * torch.randn(
                module.mode_channels[-1].weight.shape,
                generator=generator,
                dtype=dtype,
            )
        )
        module.mode_channels[-1].bias.copy_(
            0.2
            * torch.randn(
                module.mode_channels[-1].bias.shape,
                generator=generator,
                dtype=dtype,
            )
        )
    return module.eval()


def test_reciprocal_ball_is_complete_under_unimodular_cell_change():
    lattice = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.4, 3.4, 0.0], [0.2, 0.5, 4.1]]],
        dtype=torch.float64,
    )
    basis = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    first = reciprocal_ball(lattice, 4.0)
    second = reciprocal_ball((basis @ lattice[0]).unsqueeze(0), 4.0)
    first_modes = first.cartesian_modes[0, first.mask[0]]
    second_modes = second.cartesian_modes[0, second.mask[0]]
    distance = torch.minimum(
        torch.linalg.vector_norm(
            first_modes[:, None, :] - second_modes[None, :, :], dim=-1
        ),
        torch.linalg.vector_norm(
            first_modes[:, None, :] + second_modes[None, :, :], dim=-1
        ),
    )
    assert first_modes.shape == second_modes.shape
    assert float(distance.amin(dim=1).max()) < 2e-12
    assert float(distance.amin(dim=0).max()) < 2e-12


def test_reciprocal_score_is_translation_permutation_and_cell_covariant():
    torch.manual_seed(8102)
    module = _active_head()
    nodes = torch.randn((5, 12), dtype=torch.float64)
    coordinates = torch.tensor(
        [
            [0.05, 0.10, 0.15],
            [0.35, 0.25, 0.68],
            [0.15, 0.73, 0.45],
            [0.72, 0.55, 0.20],
            [0.44, 0.81, 0.32],
        ],
        dtype=torch.float64,
    )
    lattice = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.4, 3.4, 0.0], [0.2, 0.5, 4.1]]],
        dtype=torch.float64,
    )
    batch = torch.zeros(5, dtype=torch.long)
    observed = module(nodes, coordinates, lattice, batch)
    shifted = module(
        nodes,
        coordinates + torch.tensor([0.31, -0.27, 1.19], dtype=torch.float64),
        lattice,
        batch,
    )
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = module(nodes[order], coordinates[order], lattice, batch)
    basis = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    cell_changed = module(
        nodes,
        coordinates @ torch.linalg.inv(basis),
        (basis @ lattice[0]).unsqueeze(0),
        batch,
    )
    assert torch.allclose(shifted, observed, atol=2e-12, rtol=2e-12)
    assert torch.allclose(permuted, observed[order], atol=2e-12, rtol=2e-12)
    assert torch.allclose(cell_changed, observed, atol=2e-12, rtol=2e-12)
    assert torch.allclose(observed.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=2e-12)


def test_reciprocal_score_is_o3_covariant_and_has_finite_backward_gradient():
    module = _active_head()
    nodes = torch.randn((4, 12), dtype=torch.float64, requires_grad=True)
    coordinates = torch.rand((4, 3), dtype=torch.float64, requires_grad=True)
    lattice = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.3, 3.5, 0.0], [0.2, 0.4, 4.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.zeros(4, dtype=torch.long)
    rotation, _ = torch.linalg.qr(
        torch.tensor(
            [[0.3, 0.7, -0.2], [-0.6, 0.1, 0.8], [0.5, -0.4, 0.2]],
            dtype=torch.float64,
        )
    )
    observed = module(nodes, coordinates, lattice, batch)
    rotated = module(nodes, coordinates, lattice @ rotation, batch)
    assert torch.allclose(rotated, observed @ rotation, atol=3e-12, rtol=3e-12)
    observed.square().sum().backward()
    for value in (nodes.grad, coordinates.grad, lattice.grad):
        assert value is not None and torch.isfinite(value).all()
