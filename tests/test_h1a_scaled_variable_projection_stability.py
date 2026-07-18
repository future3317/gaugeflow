from __future__ import annotations

import torch
from torch import nn

from scripts.audit_h1a_scaled_variable_projection_stability import (
    ScaledCoordinateReadout,
    gradient_agreement,
    is_power_of_two,
    weighted_affine_fit,
)


class _ReadoutToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coordinate_vector_head = nn.Linear(2, 1, bias=False)
        self.coordinate_edge_head = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Linear(2, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.coordinate_vector_head(value) + self.coordinate_edge_head(value)


def test_power_of_two_readout_chart_is_exactly_reversible() -> None:
    torch.manual_seed(7)
    model = _ReadoutToy()
    value = torch.randn(8, 2)
    original_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    original = model(value)
    with ScaledCoordinateReadout(model, 1024.0):
        scaled = model(value)
        assert torch.equal(original, scaled)
    assert all(
        torch.equal(tensor, model.state_dict()[name])
        for name, tensor in original_state.items()
    )
    assert is_power_of_two(1024.0)
    assert not is_power_of_two(1000.0)
    assert not is_power_of_two(0.0)


def test_graph_equal_affine_fit_respects_design_scaling() -> None:
    design = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    target = design @ torch.tensor([2.0, -3.0])
    graph = torch.tensor([0, 0, 1, 1])
    solution, spectrum = weighted_affine_fit(
        1024.0 * design, target, graph, 2, rcond=1e-10
    )
    expected = torch.tensor([2.0, -3.0], dtype=torch.float64) / 1024.0
    torch.testing.assert_close(solution, expected)
    torch.testing.assert_close(
        1024.0 * design.double() @ solution, target.double()
    )
    assert spectrum["rank"] == 2


def test_gradient_agreement_reports_direction_and_scale() -> None:
    reference = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([-1.0])}
    candidate = {name: 1.5 * value for name, value in reference.items()}
    result = gradient_agreement(reference, candidate)
    assert abs(result["cosine"] - 1.0) < 1e-7
    assert abs(result["candidate_over_reference_norm"] - 1.5) < 1e-7
