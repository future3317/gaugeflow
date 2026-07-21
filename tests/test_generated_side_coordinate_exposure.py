from __future__ import annotations

import torch

from scripts.evaluate_h1a_generated_side_coordinate_exposure import (
    _geometry_metrics,
    _nearest_neighbours,
)


def test_exact_nearest_neighbour_metrics_match_simple_periodic_cell() -> None:
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    lattice = (2.0 * torch.eye(3)).unsqueeze(0)
    batch = torch.zeros(2, dtype=torch.long)
    counts = torch.tensor([2], dtype=torch.long)

    node, graph = _nearest_neighbours(coordinates, lattice, batch)

    torch.testing.assert_close(node, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(graph, torch.ones(1, dtype=torch.float64))
    reference = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    metrics = _geometry_metrics(
        coordinates,
        lattice,
        batch,
        counts,
        reference,
        reference / (4.0**(1.0 / 3.0)),
        points=33,
        minimum_distance=0.5,
    )
    assert metrics["finite"] is True
    assert metrics["valid_minimum_distance_fraction"] == 1.0
