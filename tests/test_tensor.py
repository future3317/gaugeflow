import torch

from gaugeflow.tensor import (
    fixed_lossless_response_probes,
    fixed_so3_frames,
    piezo_cartesian_to_voigt,
    piezo_voigt_to_cartesian,
    polarized_response,
    response_field,
    response_field_error,
    maximum_response_field_error,
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
    assert maximum_response_field_error(tensor, torch.zeros_like(tensor), directions) > 0


def test_response_field_is_lossless_by_polarization():
    tensor = piezo_voigt_to_cartesian(torch.randn(3, 6))
    basis = torch.eye(3)
    recovered = torch.stack(
        [polarized_response(tensor, basis[j], basis[k]) for j in range(3) for k in range(3)], dim=-1
    ).reshape(3, 3, 3)
    assert torch.allclose(recovered, tensor, atol=1e-6, rtol=1e-6)


def test_fixed_response_probes_recover_all_voigt_components():
    probes = fixed_lossless_response_probes()
    dyads = torch.stack((
        probes[:, 0].square(), probes[:, 1].square(), probes[:, 2].square(),
        2 * probes[:, 1] * probes[:, 2], 2 * probes[:, 0] * probes[:, 2], 2 * probes[:, 0] * probes[:, 1],
    ), dim=-1)
    assert torch.linalg.matrix_rank(dyads) == 6
    value = torch.randn(3, 6)
    tensor = piezo_voigt_to_cartesian(value)
    fields = response_field(tensor.unsqueeze(0).expand(probes.shape[0], -1, -1, -1), probes)
    recovered = torch.linalg.solve(dyads, fields).transpose(0, 1)
    assert torch.allclose(recovered, value, atol=2e-5, rtol=2e-5)


def test_rank_three_rotation_preserves_tensor_norm():
    tensor = piezo_voigt_to_cartesian(torch.randn(3, 6))
    rotation = fixed_so3_frames(4)[2]
    assert torch.allclose(torch.linalg.vector_norm(rotate_rank3(tensor, rotation)), torch.linalg.vector_norm(tensor), atol=1e-5)
