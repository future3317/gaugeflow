import torch

from gaugeflow.harmonic import (
    HarmonicDoubleCosetConditionEncoder,
    HarmonicRelativeAlignment,
    deterministic_so3_grid,
    finite_grid_shift_residual,
    harmonic_alignment_scores,
    low_order_orbit_invariants,
    normalized_low_order_orbit_invariants,
    rotate_piezo_irreps_on_grid,
)
from gaugeflow.tensor import piezo_to_irreps, piezo_voigt_to_cartesian, rotate_rank3
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


def test_continuous_harmonic_score_obeys_left_right_covariance_theorem():
    # s(R; gx, he) = s(g^-1 R h; x, e).  The two sides are evaluated at the
    # exact transformed rotation nodes, not after nearest-grid reindexing.
    torch.manual_seed(17)
    condition = _piezo(2)
    directions = torch.nn.functional.normalize(torch.randn(12, 3, dtype=torch.float64), dim=-1)
    edge_graph = torch.tensor([0] * 6 + [1] * 6)
    grid = deterministic_so3_grid(37, dtype=torch.float64)
    g, h = grid[11], grid[23]
    transformed_directions = directions @ g.transpose(-1, -2)
    from gaugeflow.tensor import piezo_from_irreps

    transformed_condition = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), h))
    weights = {
        "weight_l1": torch.tensor([0.7, -1.1], dtype=torch.float64),
        "weight_l2": torch.tensor([0.3], dtype=torch.float64),
        "weight_l3": torch.tensor([-0.5], dtype=torch.float64),
    }
    left, _ = harmonic_alignment_scores(
        transformed_condition, transformed_directions, edge_graph, grid, **weights
    )
    transformed_nodes = g.transpose(-1, -2).unsqueeze(0) @ grid @ h.unsqueeze(0)
    right, _ = harmonic_alignment_scores(condition, directions, edge_graph, transformed_nodes, **weights)
    assert torch.allclose(left, right, atol=3e-5, rtol=3e-5)


def test_finite_grid_shift_is_exact_for_identity_and_reports_nonclosure_otherwise():
    grid = deterministic_so3_grid(24, dtype=torch.float64)
    identity_residual = finite_grid_shift_residual(grid)
    assert torch.allclose(identity_residual, torch.zeros_like(identity_residual), atol=1e-12)
    # A rotation from a distinct QMC grid is deliberately not a special
    # reindexing symmetry of the 24-node grid.  The positive residual is a
    # diagnostic result, not a failed assertion of the continuous theorem.
    shift = deterministic_so3_grid(29, dtype=torch.float64)[17]
    residual = finite_grid_shift_residual(grid, left=shift, right=shift.transpose(-1, -2))
    assert torch.isfinite(residual).all()
    assert float(residual.max()) > 1e-4


def test_harmonic_representative_and_high_symmetry_conditions_are_handled_explicitly():
    # This nonzero polar tensor is invariant under a C4z rotation.  A high
    # symmetry representative must therefore leave every continuous score
    # unchanged, while a generic representative is covered by the theorem.
    condition = torch.zeros(1, 3, 6, dtype=torch.float64)
    condition[0, 2, 0] = 1.0
    condition[0, 2, 1] = 1.0
    condition = piezo_to_irreps(piezo_voigt_to_cartesian(condition))
    c4z = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    from gaugeflow.tensor import piezo_from_irreps

    assert torch.allclose(
        rotate_rank3(piezo_from_irreps(condition), c4z), piezo_from_irreps(condition), atol=2e-8, rtol=2e-8
    )
    directions = torch.nn.functional.normalize(torch.randn(8, 3, dtype=torch.float64), dim=-1)
    edge_graph = torch.zeros(8, dtype=torch.long)
    grid = deterministic_so3_grid(19, dtype=torch.float64)
    weights = {
        "weight_l1": torch.tensor([1.0, -0.4], dtype=torch.float64),
        "weight_l2": torch.tensor([0.2], dtype=torch.float64),
        "weight_l3": torch.tensor([0.6], dtype=torch.float64),
    }
    original, _ = harmonic_alignment_scores(condition, directions, edge_graph, grid, **weights)
    symmetric, _ = harmonic_alignment_scores(
        piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), c4z)), directions, edge_graph, grid, **weights
    )
    assert torch.allclose(original, symmetric, atol=3e-5, rtol=3e-5)


def test_zero_tensor_has_uniform_harmonic_scores_and_finite_gradients():
    condition = torch.zeros(1, 18, requires_grad=True)
    directions = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    edge_graph = torch.zeros(6, dtype=torch.long)
    score, _ = harmonic_alignment_scores(
        condition,
        directions,
        edge_graph,
        deterministic_so3_grid(16),
        weight_l1=torch.tensor([1.0, 1.0]),
        weight_l2=torch.tensor([1.0]),
        weight_l3=torch.tensor([1.0]),
    )
    posterior = torch.softmax(score, dim=-1)
    assert torch.allclose(posterior, torch.full_like(posterior, 1.0 / 16.0), atol=1e-6)
    posterior.square().sum().backward()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()


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
