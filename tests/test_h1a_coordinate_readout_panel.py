import pytest
import torch

from scripts.audit_h1a_coordinate_readout_panel import weighted_affine_fit


def test_weighted_affine_fit_gives_equal_graph_weight_to_unequal_rows():
    design = torch.tensor([[1.0], [1.0], [1.0]])
    target = torch.tensor([0.0, 2.0, 10.0])
    row_graph = torch.tensor([0, 0, 1])
    solution, metrics = weighted_affine_fit(
        design, target, row_graph, 2, rcond=1e-10
    )
    # Graph 0 mean target is 1 and graph 1 mean target is 10.
    assert solution.item() == pytest.approx(5.5)
    assert metrics["rank"] == 1


def test_weighted_affine_fit_recovers_full_rank_solution():
    design = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    truth = torch.tensor([3.0, -2.0])
    target = design @ truth
    solution, metrics = weighted_affine_fit(
        design, target, torch.tensor([0, 0, 0]), 1, rcond=1e-10
    )
    assert torch.allclose(solution, truth.double())
    assert metrics["rank"] == 2
