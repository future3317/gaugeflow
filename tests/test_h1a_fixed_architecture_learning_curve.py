from scripts.evaluate_h1a_fixed_architecture_learning_curve import (
    _classify_learning_curve,
    _is_finite_tree,
)

SPECIFICATION = {
    "one_pass_reference_ratio_absolute_tolerance": 0.02,
    "one_to_two_pass_relative_validation_improvement_undertraining_min": 0.10,
    "one_to_two_pass_relative_validation_improvement_plateau_max": 0.05,
}


def test_learning_curve_classification_is_frozen_and_boundary_complete() -> None:
    decision, improvement, matches = _classify_learning_curve(0.5, 0.44, 0.54, 0.544, SPECIFICATION)
    assert decision == "undertraining" and improvement > 0.10 and matches

    decision, improvement, matches = _classify_learning_curve(0.5, 0.48, 0.54, 0.544, SPECIFICATION)
    assert decision == "representation_ceiling" and improvement < 0.05 and matches

    decision, improvement, matches = _classify_learning_curve(0.5, 0.465, 0.54, 0.544, SPECIFICATION)
    assert decision == "ambiguous" and 0.05 < improvement < 0.10 and matches


def test_learning_curve_reference_mismatch_preempts_interpretation() -> None:
    decision, _, matches = _classify_learning_curve(0.5, 0.4, 0.60, 0.544, SPECIFICATION)
    assert decision == "reference_mismatch" and not matches


def test_nested_training_log_finiteness() -> None:
    assert _is_finite_tree({"loss": 1.0, "gradients": {"block": 0.5}})
    assert not _is_finite_tree({"loss": float("nan")})
