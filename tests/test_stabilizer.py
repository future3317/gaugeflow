import hashlib

import torch
from pymatgen.core import Lattice, Structure

from gaugeflow.manifold import torus_logmap
from gaugeflow.stabilizer import (
    batched_soft_crystal_stabilizer_actions,
    crystal_point_group_operations,
    observed_tensor_stabilizer_rotations,
    proper_stabilizer_rotations,
    proper_unimodular_candidates,
    soft_crystal_stabilizer_actions,
)
from gaugeflow.tensor import rotate_rank3


def test_proper_stabilizer_excludes_improper_operations():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])
    rotations = proper_stabilizer_rotations(structure)
    determinants = torch.linalg.det(rotations)
    assert rotations.shape[-2:] == (3, 3)
    assert torch.allclose(determinants, torch.ones_like(determinants), atol=1e-4)
    assert any(torch.allclose(rotation, torch.eye(3), atol=1e-5) for rotation in rotations)


def test_full_crystal_point_group_retains_improper_operations_for_compatibility_only():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])
    operations = crystal_point_group_operations(structure, proper_only=False)
    determinants = torch.linalg.det(operations)
    assert torch.allclose(determinants.abs(), torch.ones_like(determinants), atol=1e-4)
    assert bool((determinants < 0).any())
    assert operations.shape[0] > proper_stabilizer_rotations(structure).shape[0]


def test_observed_tensor_stabilizer_keeps_only_response_preserving_rotations():
    identity = torch.eye(3)
    c2z = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    raw = torch.randn(3, 3, 3)
    raw = 0.5 * (raw + raw.transpose(-1, -2))
    tensor = 0.5 * (raw + rotate_rank3(raw, c2z))
    observed = observed_tensor_stabilizer_rotations(tensor, torch.stack((identity, c2z)))
    assert observed.shape == (2, 3, 3)


def test_soft_stabilizer_comes_from_the_current_state_and_is_differentiable():
    frac = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    lattice = (4.0 * torch.eye(3)).requires_grad_()
    type_state = torch.zeros(1, 3, requires_grad=True)
    actions, weights = soft_crystal_stabilizer_actions(frac, lattice, type_state)
    assert actions.shape == (792, 3, 3)
    assert torch.allclose(actions.transpose(-1, -2) @ actions, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(actions), torch.ones(792), atol=1e-5)
    assert torch.allclose(weights.sum(), torch.ones(()))
    # A single atom in a cubic cell has the 24 proper cubic rotations.  The
    # score is built from the present state, so it has a usable flow gradient.
    (weights.square().sum() + actions.square().sum()).backward()
    assert lattice.grad is not None
    assert frac.grad is not None


def test_soft_automorphism_proposals_are_always_proper_rotations():
    frac = torch.tensor([[0.12, 0.24, 0.35], [0.41, 0.63, 0.78]])
    lattice = torch.tensor([[3.1, 0.0, 0.0], [0.7, 4.2, 0.0], [0.2, 0.4, 5.3]])
    actions, _ = soft_crystal_stabilizer_actions(frac, lattice, torch.randn(2, 4))
    identity = torch.eye(3)
    assert torch.allclose(actions.transpose(-1, -2) @ actions, identity, atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.linalg.det(actions), torch.ones(actions.shape[0]), atol=2e-5, rtol=2e-5)


def test_integer_automorphism_catalogue_contains_only_crystallographic_finite_orders():
    candidates = proper_unimodular_candidates().to(dtype=torch.int64)
    identity = torch.eye(3, dtype=torch.int64)
    assert candidates.shape == (792, 3, 3)
    for candidate in candidates:
        power = identity
        has_finite_order = False
        for order in range(1, 7):
            power = power @ candidate
            if order in {1, 2, 3, 4, 6} and torch.equal(power, identity):
                has_finite_order = True
                break
        assert has_finite_order


def test_integer_automorphism_candidate_ids_are_frozen():
    candidates = proper_unimodular_candidates().to(dtype=torch.int8).contiguous()
    digest = hashlib.sha256(candidates.numpy().tobytes()).hexdigest()
    assert digest == "46a55a1e479d76cc02a6487060f9a9c526051411227d34e9edd5186b8669a792"


def _reference_all_candidate_actions(frac, lattice, type_state, candidates):
    proposed = torch.linalg.solve(lattice, candidates @ lattice)
    left, _, right_t = torch.linalg.svd(proposed)
    raw = left @ right_t
    correction = torch.eye(3, dtype=lattice.dtype).expand_as(raw).clone()
    correction[:, -1, -1] = torch.where(torch.linalg.det(raw) < 0, -1.0, 1.0)
    row_rotations = left @ correction @ right_t
    lattice_error = (proposed - row_rotations).square().mean((-1, -2))
    cartesian = frac @ lattice
    rotated = (cartesian @ row_rotations) @ torch.linalg.inv(lattice)
    translations = frac.unsqueeze(0) - rotated[:, :1, :]
    transformed = rotated.unsqueeze(1) + translations.unsqueeze(2)
    delta = torus_logmap(
        transformed.unsqueeze(3), frac.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )
    cartesian_delta = torch.einsum("stnij,jk->stnik", delta, lattice)
    distances = cartesian_delta.square().sum(dim=-1)
    probabilities = torch.softmax(type_state, dim=-1)
    distances = distances + 4.0 * (
        1.0 - probabilities @ probabilities.transpose(0, 1)
    ).unsqueeze(0).unsqueeze(0)
    nearest = -0.1 * torch.logsumexp(-distances / 0.1, dim=-1)
    translation_error = nearest.mean(dim=-1)
    atom_error = -0.1 * torch.logsumexp(-translation_error / 0.1, dim=-1)
    weights = torch.softmax(-lattice_error / 0.02 - atom_error / 0.1, dim=0)
    return row_rotations.transpose(-1, -2), weights


def test_batched_all_candidate_posterior_matches_unbatched_reference():
    torch.manual_seed(23)
    candidates = proper_unimodular_candidates()
    frac = torch.tensor(
        [[0.12, 0.24, 0.35], [0.41, 0.63, 0.78], [0.18, 0.33, 0.57]],
        requires_grad=True,
    )
    lattices = torch.tensor(
        [
            [[3.1, 0.0, 0.0], [0.7, 4.2, 0.0], [0.2, 0.4, 5.3]],
            [[2.7, 0.0, 0.0], [0.3, 3.8, 0.0], [0.4, 0.2, 4.9]],
        ],
        requires_grad=True,
    )
    types = torch.randn(3, 4, requires_grad=True)
    batch = torch.tensor([0, 0, 1])
    actions, weights = batched_soft_crystal_stabilizer_actions(
        frac, lattices, types, batch, candidates=candidates, candidate_chunk_size=97
    )
    references = [
        _reference_all_candidate_actions(frac[:2], lattices[0], types[:2], candidates),
        _reference_all_candidate_actions(frac[2:], lattices[1], types[2:], candidates),
    ]
    reference_actions = torch.stack([value[0] for value in references])
    reference_weights = torch.stack([value[1] for value in references])
    assert torch.allclose(actions, reference_actions, atol=3e-6, rtol=3e-6)
    assert torch.allclose(weights, reference_weights, atol=3e-6, rtol=3e-6)
    optimized_objective = weights.square().sum() + 1e-4 * actions.square().sum()
    reference_objective = reference_weights.square().sum() + 1e-4 * reference_actions.square().sum()
    optimized_gradients = torch.autograd.grad(
        optimized_objective, (frac, lattices, types), retain_graph=True
    )
    reference_gradients = torch.autograd.grad(reference_objective, (frac, lattices, types))
    assert all(torch.isfinite(value).all() for value in optimized_gradients)
    # Coordinate and type derivatives do not differentiate the polar factor
    # itself and therefore remain directly comparable. The legacy SVD lattice
    # derivative is undefined for repeated singular values and contains NaNs.
    for index in (0, 2):
        assert torch.allclose(
            optimized_gradients[index], reference_gradients[index], atol=2e-5, rtol=2e-4
        )
    assert torch.isfinite(optimized_gradients[1]).all()
