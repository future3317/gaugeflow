import pytest
import torch

from scripts.audit_h1a_coordinate_ntk_trust_step import (
    damped_output_coefficients,
    project_common_translation,
    relative_loss_reduction,
)


def test_common_translation_projection_removes_only_graph_mean():
    value = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 8.0]])
    projected = project_common_translation(value)
    assert torch.allclose(projected.mean(0), torch.zeros(3))
    assert torch.equal(projected[1] - projected[0], value[1] - value[0])


def test_damped_output_coefficients_use_relative_fp64_damping():
    gram = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float32))
    desired = torch.tensor([2.0, 3.0])
    coefficients, damping = damped_output_coefficients(
        gram, desired, relative_damping=0.25
    )
    assert damping == pytest.approx(1.0)
    assert coefficients.dtype == torch.float64
    assert torch.allclose(coefficients, torch.tensor([0.4, 1.5], dtype=torch.float64))


def test_relative_loss_reduction_has_expected_endpoints():
    initial = torch.tensor([2.0, -1.0])
    assert relative_loss_reduction(initial, initial) == pytest.approx(0.0)
    assert relative_loss_reduction(initial, torch.zeros_like(initial)) == pytest.approx(1.0)
