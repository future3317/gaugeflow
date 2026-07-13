import torch

from gaugeflow.tensor import (
    fixed_so3_frames,
    piezo_cartesian_to_voigt,
    piezo_voigt_to_cartesian,
    response_field,
    response_field_error,
    rotate_rank3,
)


def test_voigt_round_trip():
    value = torch.randn(2, 3, 6, dtype=torch.float64)
    assert torch.allclose(piezo_cartesian_to_voigt(piezo_voigt_to_cartesian(value)), value)


def test_full_response_field_detects_nonzero_tensor_difference():
    tensor = piezo_voigt_to_cartesian(torch.randn(3, 6))
    directions = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    field = response_field(tensor.unsqueeze(0).expand(64, -1, -1, -1), directions)
    assert field.shape == (64, 3)
    assert response_field_error(tensor, torch.zeros_like(tensor), directions) > 0


def test_rank_three_rotation_preserves_tensor_norm():
    tensor = piezo_voigt_to_cartesian(torch.randn(3, 6))
    rotation = fixed_so3_frames(4)[2]
    assert torch.allclose(torch.linalg.vector_norm(rotate_rank3(tensor, rotation)), torch.linalg.vector_norm(tensor), atol=1e-5)
