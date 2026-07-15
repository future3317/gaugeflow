import torch

from gaugeflow.direct_irrep import CompleteDirectIrrepCoupling
from gaugeflow.tensor import fixed_so3_frames, rotate_rank3


def test_complete_direct_irrep_has_all_six_vector_cg_pathways():
    torch.manual_seed(21)
    tensor = torch.randn(5, 3, 3, 3)
    tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
    direction = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    coupling = CompleteDirectIrrepCoupling()
    pathways = coupling(tensor, direction)
    assert pathways.shape == (5, 6, 3)
    assert torch.isfinite(pathways).all()


def test_complete_direct_irrep_vector_pathways_are_so3_equivariant():
    torch.manual_seed(22)
    tensor = torch.randn(4, 3, 3, 3)
    tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
    direction = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    rotation = fixed_so3_frames(5)[3]
    coupling = CompleteDirectIrrepCoupling()
    original = coupling(tensor, direction)
    rotated = coupling(rotate_rank3(tensor, rotation), direction @ rotation.T)
    assert torch.allclose(rotated, original @ rotation.T, atol=2e-5, rtol=2e-5)
