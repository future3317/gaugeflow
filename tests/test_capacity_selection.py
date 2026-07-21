from scripts.evaluate_gaugeflow_base_capacity import select_capacity_candidate


def _row(name: str, parameters: int, ratio: float, w1: float, eligible: bool = True) -> dict:
    return {
        "candidate": name,
        "parameter_count": parameters,
        "validation_coordinate_ratio": ratio,
        "clean_side_conditional_rollout": {"node_nearest_w1_normalized": w1},
        "eligible": eligible,
    }


def test_capacity_selection_prefers_smallest_jointly_sufficient_model() -> None:
    rows = [
        _row("small", 34, 0.24, 0.30),
        _row("base", 58, 0.225, 0.28),
        _row("large", 98, 0.20, 0.27),
    ]
    specification = {
        "validation_ratio_best_absolute_margin": 0.02,
        "full_prior_w1_best_absolute_margin": 0.03,
    }
    assert select_capacity_candidate(rows, specification) == "large"


def test_capacity_selection_ignores_ineligible_and_avoids_unnecessary_scale() -> None:
    rows = [
        _row("small", 34, 0.215, 0.29),
        _row("base", 58, 0.20, 0.27),
        _row("large", 98, 0.18, 0.20, eligible=False),
    ]
    specification = {
        "validation_ratio_best_absolute_margin": 0.02,
        "full_prior_w1_best_absolute_margin": 0.03,
    }
    assert select_capacity_candidate(rows, specification) == "small"


def test_capacity_selection_stops_when_no_candidate_is_eligible() -> None:
    rows = [_row("small", 34, 0.2, 0.2, eligible=False)]
    specification = {
        "validation_ratio_best_absolute_margin": 0.02,
        "full_prior_w1_best_absolute_margin": 0.03,
    }
    assert select_capacity_candidate(rows, specification) is None
