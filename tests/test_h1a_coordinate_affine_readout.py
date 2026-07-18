import pytest
import torch

from scripts.audit_h1a_coordinate_affine_readout import (
    affine_readout_solution,
    project_common_translation,
)


def test_common_translation_projection_preserves_relative_vectors():
    value = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 8.0]])
    projected = project_common_translation(value)
    assert torch.allclose(projected.mean(0), torch.zeros(3))
    assert torch.equal(projected[1] - projected[0], value[1] - value[0])


def test_affine_readout_solution_fits_reachable_output_and_reports_rank():
    jacobian = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    desired = torch.tensor([3.0, -4.0])
    delta, metrics = affine_readout_solution(
        jacobian, desired, relative_threshold=1e-10
    )
    assert metrics["rank"] == 2
    assert metrics["target_projection_relative_residual"] < 1e-12
    assert torch.allclose(jacobian.double() @ delta, desired.double())


def test_affine_readout_solution_reports_unreachable_component():
    jacobian = torch.tensor([[1.0], [0.0]])
    _, metrics = affine_readout_solution(
        jacobian, torch.tensor([0.0, 2.0]), relative_threshold=1e-10
    )
    assert metrics["rank"] == 1
    assert metrics["target_projection_relative_residual"] == pytest.approx(1.0)
