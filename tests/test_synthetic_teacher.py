import torch

from gaugeflow.synthetic_teacher import (
    directed_species_rank3_teacher,
    symmetric_weighted_directed_rank3_sum,
)
from gaugeflow.tensor import fixed_so3_frames, rotate_rank3


def _example():
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.23, 0.27, 0.1], [0.61, 0.1, 0.42]])
    lattice = torch.tensor([[[3.2, 0.2, 0.1], [0.3, 4.1, 0.4], [0.1, 0.2, 5.0]]])
    batch = torch.zeros(3, dtype=torch.long)
    scalar = torch.tensor([1.0, 7.0, 49.0])
    return frac, lattice, batch, scalar


def test_directed_antisymmetric_synthetic_rank3_teacher_is_nonzero_and_symmetric():
    frac, lattice, batch, scalar = _example()
    tensor = directed_species_rank3_teacher(frac, lattice, batch, scalar)
    assert tensor.shape == (1, 3, 3, 3)
    assert torch.linalg.vector_norm(tensor) > 1e-5
    assert torch.allclose(tensor, tensor.transpose(-1, -2))
    naive = symmetric_weighted_directed_rank3_sum(frac, lattice, batch, scalar)
    assert torch.linalg.vector_norm(naive) < 1e-5


def test_synthetic_rank3_teacher_is_cartesian_so3_equivariant():
    frac, lattice, batch, scalar = _example()
    rotation = fixed_so3_frames(5)[2]
    original = directed_species_rank3_teacher(frac, lattice, batch, scalar)
    rotated_lattice = lattice @ rotation.T
    rotated = directed_species_rank3_teacher(frac, rotated_lattice, batch, scalar)
    assert torch.allclose(rotated, rotate_rank3(original, rotation), atol=3e-5, rtol=3e-5)
