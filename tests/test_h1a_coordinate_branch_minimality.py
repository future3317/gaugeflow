from __future__ import annotations

import torch
from torch import nn

from scripts.audit_h1a_coordinate_branch_minimality import (
    assign_branch,
    branch_slices,
    helmert_quotient_basis,
    weighted_fit,
)


class _ToyReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coordinate_vector_head = nn.Linear(2, 1, bias=False)
        self.coordinate_edge_head = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Linear(3, 1)
        )


def test_branch_slices_partition_the_affine_columns() -> None:
    slices = branch_slices(2, 6)
    assert slices["vector_only"] == slice(0, 2)
    assert slices["edge_only"] == slice(2, 6)
    assert slices["combined"] == slice(0, 6)


def test_helmert_basis_removes_exactly_three_translation_modes() -> None:
    quotient = helmert_quotient_basis(
        5, device=torch.device("cpu"), dtype=torch.float64
    )
    assert quotient.shape == (15, 12)
    torch.testing.assert_close(
        quotient.transpose(0, 1) @ quotient, torch.eye(12, dtype=torch.float64)
    )
    translations = torch.kron(
        torch.ones((5, 1), dtype=torch.float64), torch.eye(3, dtype=torch.float64)
    )
    torch.testing.assert_close(
        quotient.transpose(0, 1) @ translations,
        torch.zeros((12, 3), dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )


def test_assign_branch_zeros_the_unselected_readout() -> None:
    model = _ToyReadout()
    assign_branch(model, "vector_only", torch.tensor([2.0, -3.0]))
    torch.testing.assert_close(
        model.coordinate_vector_head.weight,
        torch.tensor([[2.0, -3.0]]),
    )
    assert torch.count_nonzero(model.coordinate_edge_head[2].weight) == 0
    assert torch.count_nonzero(model.coordinate_edge_head[2].bias) == 0
    assign_branch(model, "edge_only", torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.count_nonzero(model.coordinate_vector_head.weight) == 0
    torch.testing.assert_close(
        model.coordinate_edge_head[2].weight,
        torch.tensor([[1.0, 2.0, 3.0]]),
    )
    torch.testing.assert_close(model.coordinate_edge_head[2].bias, torch.tensor([4.0]))


def test_weighted_fit_recovers_a_full_rank_branch() -> None:
    design = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    target = design @ torch.tensor([2.0, -3.0])
    graph = torch.tensor([0, 0, 1, 1])
    solution, spectrum = weighted_fit(design, target, graph, 2, rcond=1e-10)
    torch.testing.assert_close(solution, torch.tensor([2.0, -3.0], dtype=torch.float64))
    assert spectrum["rank"] == 2
