from __future__ import annotations

import torch

from gaugeflow.production.stage_c_evaluation import (
    balanced_functional_panel,
    select_pareto_minimax_checkpoint,
)


def test_balanced_functional_panel_is_deterministic_disjoint_and_balanced() -> None:
    groups = torch.tensor([0] * 9 + [1] * 8 + [2] * 7, dtype=torch.long)
    first = balanced_functional_panel(
        groups,
        functional_count=3,
        graphs_per_functional=5,
        seed=5705,
    )
    second = balanced_functional_panel(
        groups,
        functional_count=3,
        graphs_per_functional=5,
        seed=5705,
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(panel.numel() == 5 for panel in first)
    assert all(bool((groups[panel] == functional).all()) for functional, panel in enumerate(first))
    assert torch.unique(torch.cat(first)).numel() == 15


def test_balanced_functional_panel_fails_closed_on_missing_support() -> None:
    groups = torch.tensor([0, 0, 1], dtype=torch.long)
    try:
        balanced_functional_panel(
            groups,
            functional_count=2,
            graphs_per_functional=2,
            seed=1,
        )
    except ValueError as error:
        assert "support" in str(error)
    else:
        raise AssertionError("missing functional support was silently accepted")


def test_pareto_minimax_selection_discards_dominated_and_balances_regret() -> None:
    result = select_pareto_minimax_checkpoint(
        {
            10: {"physical": 1.0, "geometry": 0.0},
            20: {"physical": 0.5, "geometry": 0.5},
            30: {"physical": 0.0, "geometry": 1.0},
            40: {"physical": 0.8, "geometry": 0.8},
        }
    )
    assert result["pareto_stage_c_steps"] == [10, 20, 30]
    assert result["selected_stage_c_step"] == 20
    assert result["maximum_regret"] == 0.5
