import torch

from gaugeflow.stabilizer import (
    observed_tensor_stabilizer_rotations,
    proper_unimodular_candidates,
    proper_stabilizer_rotations,
    soft_crystal_stabilizer_actions,
)
from gaugeflow.tensor import rotate_rank3
from pymatgen.core import Lattice, Structure


def test_proper_stabilizer_excludes_improper_operations():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])
    rotations = proper_stabilizer_rotations(structure)
    determinants = torch.linalg.det(rotations)
    assert rotations.shape[-2:] == (3, 3)
    assert torch.allclose(determinants, torch.ones_like(determinants), atol=1e-4)
    assert any(torch.allclose(rotation, torch.eye(3), atol=1e-5) for rotation in rotations)


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
    assert actions.shape == (24, 3, 3)
    assert torch.allclose(actions.transpose(-1, -2) @ actions, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(actions), torch.ones(24), atol=1e-5)
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
    assert candidates.shape[0] < 3480
    for candidate in candidates:
        power = identity
        has_finite_order = False
        for order in range(1, 7):
            power = power @ candidate
            if order in {1, 2, 3, 4, 6} and torch.equal(power, identity):
                has_finite_order = True
                break
        assert has_finite_order
