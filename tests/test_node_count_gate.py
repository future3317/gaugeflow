from __future__ import annotations

import torch

from scripts.qualify_h1a_node_count_law import (
    _bootstrap_nll_difference_ucb95,
    _integer_wasserstein,
    _js_divergence,
)


def test_node_count_distribution_metrics_are_exact_on_identical_laws() -> None:
    probability = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    assert _js_divergence(probability, probability) == 0.0
    assert _integer_wasserstein(probability, probability) == 0.0


def test_node_count_integer_wasserstein_uses_ordered_support() -> None:
    left = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    right = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert _integer_wasserstein(left, right) == 2.0


def test_node_count_bootstrap_is_structure_level_and_reproducible() -> None:
    values = torch.linspace(-2.0, -1.0, 101)
    first = _bootstrap_nll_difference_ucb95(values, resamples=1000, seed=17)
    second = _bootstrap_nll_difference_ucb95(values, resamples=1000, seed=17)
    assert first == second
    assert first < 0.0
