import torch

from gaugeflow.stabilizer import proper_stabilizer_rotations
from pymatgen.core import Lattice, Structure


def test_proper_stabilizer_excludes_improper_operations():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])
    rotations = proper_stabilizer_rotations(structure)
    determinants = torch.linalg.det(rotations)
    assert rotations.shape[-2:] == (3, 3)
    assert torch.allclose(determinants, torch.ones_like(determinants), atol=1e-4)
    assert any(torch.allclose(rotation, torch.eye(3), atol=1e-5) for rotation in rotations)
