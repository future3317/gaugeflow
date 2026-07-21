import torch

from gaugeflow.production.generation_metrics import (
    formula_keys,
    jensen_shannon,
    minimum_periodic_distances,
    quantile_wasserstein,
    robust_scale,
)


def test_formula_keys_are_site_order_invariant() -> None:
    batch = torch.tensor([0, 0, 0, 1, 1])
    first = formula_keys(torch.tensor([0, 7, 0, 5, 5]), batch, 2)
    second = formula_keys(torch.tensor([7, 0, 0, 5, 5]), batch, 2)
    assert first == second == ["1:2;8:1", "6:2"]


def test_distribution_metrics_have_exact_reference_zero() -> None:
    values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    assert jensen_shannon(torch.tensor([1, 3]), torch.tensor([1, 3])) == 0.0
    assert quantile_wasserstein(values, values, points=17) == 0.0
    assert robust_scale(values) > 0.0


def test_minimum_periodic_distance_uses_nonself_edges() -> None:
    fractional = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    lattice = 4.0 * torch.eye(3).unsqueeze(0)
    observed = minimum_periodic_distances(
        fractional,
        lattice,
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.allclose(observed, torch.tensor([1.0]))
