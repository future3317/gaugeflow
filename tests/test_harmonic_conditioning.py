import torch

from gaugeflow.harmonic import (
    HarmonicDoubleCosetConditionEncoder,
    HarmonicRelativeAlignment,
    deterministic_so3_grid,
    low_order_orbit_invariants,
    normalized_low_order_orbit_invariants,
    rotate_piezo_irreps_on_grid,
)
from gaugeflow.tensor import piezo_to_irreps, rotate_rank3
from gaugeflow.tensor import tensor_orbit_shape_magnitude


def _piezo(batch: int = 2) -> torch.Tensor:
    tensor = torch.randn(batch, 3, 3, 3, dtype=torch.float64)
    return piezo_to_irreps(0.5 * (tensor + tensor.transpose(-1, -2)))


def test_deterministic_so3_grid_is_proper_and_reproducible():
    first = deterministic_so3_grid(31, dtype=torch.float64)
    second = deterministic_so3_grid(31, dtype=torch.float64)
    assert torch.equal(first, second)
    assert torch.allclose(first[0], torch.eye(3, dtype=torch.float64))
    assert torch.allclose(first @ first.transpose(-1, -2), torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(torch.linalg.det(first), torch.ones(31, dtype=torch.float64), atol=1e-12)


def test_grid_irrep_rotation_matches_cartesian_rank_three_rotation():
    condition = _piezo(2)
    grid = deterministic_so3_grid(7, dtype=torch.float64)
    from_grid = rotate_piezo_irreps_on_grid(condition, grid)
    cartesian = condition.new_zeros((2, 3, 3, 3))
    # Construct the Cartesian input independently from the irrep-grid routine.
    from gaugeflow.tensor import piezo_from_irreps

    cartesian.copy_(piezo_from_irreps(condition))
    expected = piezo_to_irreps(rotate_rank3(cartesian.unsqueeze(1), grid.unsqueeze(0)))
    # e3nn's matrix-to-Wigner conversion uses an angle parameterization whose
    # numerical error is O(1e-6) even for float64 matrices.
    assert torch.allclose(from_grid, expected, atol=5e-6, rtol=5e-6)


def test_low_order_invariant_channel_is_proper_rotation_invariant():
    condition = _piezo(3)
    rotation = deterministic_so3_grid(9, dtype=torch.float64)[4]
    from gaugeflow.tensor import piezo_from_irreps

    rotated = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), rotation))
    original = low_order_orbit_invariants(condition)
    transformed = low_order_orbit_invariants(rotated)
    assert torch.allclose(original, transformed, atol=2e-6, rtol=2e-6)


def test_shape_magnitude_conditioning_keeps_physical_zero_distinct_and_shape_scale_free():
    condition = _piezo(1).float()
    decomposition = tensor_orbit_shape_magnitude(torch.cat((condition, torch.zeros_like(condition)), dim=0))
    assert not decomposition.physical_zero[0]
    assert decomposition.physical_zero[1]
    assert torch.allclose(torch.linalg.vector_norm(decomposition.shape[0]), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(decomposition.shape[1], torch.zeros_like(decomposition.shape[1]))
    scaled = normalized_low_order_orbit_invariants(3.7 * condition)
    original = normalized_low_order_orbit_invariants(condition)
    assert torch.allclose(original[:, :7], scaled[:, :7], atol=2e-5, rtol=2e-5)
    assert scaled[:, 7].item() > original[:, 7].item()
    assert normalized_low_order_orbit_invariants(torch.zeros_like(condition))[0, -1] == 1


def test_harmonic_alignment_is_uniform_for_an_isotropic_state_query():
    condition = _piezo(2).float()
    aligner = HarmonicRelativeAlignment(grid_size=12)
    aligned, posterior, entropy, diagnostics = aligner(
        condition, torch.empty((0, 3)), torch.empty((0,), dtype=torch.long)
    )
    assert aligned.shape == condition.shape
    assert torch.allclose(posterior, torch.full_like(posterior, 1.0 / 12.0), atol=1e-6)
    assert torch.allclose(entropy, torch.full_like(entropy, torch.log(torch.tensor(12.0))), atol=1e-6)
    assert torch.allclose(diagnostics["posterior"].sum(dim=-1), torch.ones(2), atol=1e-6)


def test_harmonic_alignment_returns_a_finite_state_dependent_posterior():
    condition = _piezo(2).float()
    directions = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    edge_graph = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    aligner = HarmonicRelativeAlignment(grid_size=24)
    aligned, posterior, entropy, diagnostics = aligner(condition, directions, edge_graph)
    assert torch.isfinite(aligned).all()
    assert torch.isfinite(entropy).all()
    assert posterior.shape == (2, 24)
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(diagnostics["top_mode_mass"] >= 1.0 / 24.0)


def test_harmonic_encoder_keeps_null_and_physical_zero_conditions_distinct():
    condition = torch.zeros((2, 18))
    directions = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    edge_graph = torch.tensor([0, 0, 0, 1, 1, 1])
    encoder = HarmonicDoubleCosetConditionEncoder(hidden_dim=16, grid_size=12)
    graph, response, auxiliary, posterior, diagnostics = encoder(
        condition,
        torch.tensor([[True], [False]]),
        directions,
        edge_graph,
        torch.tensor([0.0, 0.7]),
        return_diagnostics=True,
    )
    assert graph.shape == (2, 16)
    assert response.shape == auxiliary.shape == (6, 3)
    assert torch.equal(response[edge_graph == 1], torch.zeros_like(response[edge_graph == 1]))
    assert posterior.shape == (2, 12)
    assert torch.all((diagnostics["alignment_gate"] >= 0) & (diagnostics["alignment_gate"] <= 1))
