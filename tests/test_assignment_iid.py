from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gaugeflow.production.assignment_data import AssignmentCarrierExample
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT
from scripts.evaluate_h1a_assignment_iid_precision import _updated_checks
from scripts.train_h1a_assignment_iid import (
    _material_bootstrap_ucb95,
    _relabel_example,
    _select_exact_panel,
)


def _example() -> AssignmentCarrierExample:
    target, source = torch.nonzero(~torch.eye(4, dtype=torch.bool), as_tuple=True)
    assignment = torch.tensor([2, 5, 2, 5], dtype=torch.long)
    counts = torch.bincount(assignment, minlength=CHEMICAL_ELEMENT_COUNT)
    return AssignmentCarrierExample(
        embedding_key="synthetic",
        material_id_audit_only="material",
        evidence_role_audit_only="iid_test",
        site_features=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        graph_features=torch.tensor([0.25, -0.5]),
        edge_source=source,
        edge_target=target,
        edge_rbf=torch.arange(source.numel() * 2, dtype=torch.float32).reshape(-1, 2),
        composition_counts=counts,
        target_assignment=assignment,
        parent_permutations=torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ],
            dtype=torch.long,
        ),
        parent_space_group=1,
        cell_index=1,
    )


def test_iid_relabel_preserves_edges_counts_and_parent_orbit() -> None:
    example = _example()
    order = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inverse = torch.argsort(order)
    changed = _relabel_example(example, order)
    changed_key = changed.edge_target * 4 + changed.edge_source
    expected_source = inverse[example.edge_source]
    expected_target = inverse[example.edge_target]
    expected_order = torch.argsort(expected_target * 4 + expected_source, stable=True)
    assert torch.equal(changed.site_features, example.site_features[order])
    assert torch.equal(changed.target_assignment, example.target_assignment[order])
    assert torch.equal(changed.composition_counts, example.composition_counts)
    assert torch.equal(changed.edge_source, expected_source[expected_order])
    assert torch.equal(changed.edge_target, expected_target[expected_order])
    assert torch.equal(changed.edge_rbf, example.edge_rbf[expected_order])
    assert bool((changed_key[1:] > changed_key[:-1]).all())
    original_orbit = torch.unique(
        example.target_assignment[example.parent_permutations],
        dim=0,
    )
    changed_orbit = torch.unique(
        changed.target_assignment[changed.parent_permutations],
        dim=0,
    )
    assert torch.equal(changed_orbit, torch.unique(original_orbit[:, order], dim=0))


def test_material_bootstrap_weights_materials_not_duplicate_carriers() -> None:
    rows = [
        {"material_id": "a", "model_minus_uniform_nll": -2.0},
        {"material_id": "b", "model_minus_uniform_nll": -1.0},
    ]
    duplicated = [rows[0], rows[0], rows[1], rows[1]]
    reference = _material_bootstrap_ucb95(rows, resamples=1000, seed=17)
    changed = _material_bootstrap_ucb95(duplicated, resamples=1000, seed=17)
    assert reference == changed
    assert reference < 0.0


def test_exact_panel_is_validated_before_training() -> None:
    calibration = [_example() for _ in range(3)]
    test = [_example() for _ in range(4)]
    for index, example in enumerate(calibration):
        calibration[index] = replace(
            example,
            material_id_audit_only=f"calibration-{index}",
        )
    for index, example in enumerate(test):
        test[index] = replace(
            example,
            material_id_audit_only=f"test-{index}",
        )
    groups = {
        "iid_calibration_supported": calibration,
        "iid_test_supported": test,
    }
    panel = _select_exact_panel(groups, maximum_sites=4, carriers_per_split=3)
    assert len(panel) == 6
    with pytest.raises(ValueError, match="iid_calibration_supported has 3"):
        _select_exact_panel(groups, maximum_sites=4, carriers_per_split=4)


def test_precision_closure_recomputes_only_likelihood_checks() -> None:
    checks = {"iid_calibration_mc_precision": False, "exact_composition": True}
    summary = {
        "relative_nll_reduction_from_uniform": 0.5,
        "model_minus_uniform_nll_ucb95": -1.0,
        "maximum_order_elbo_mc_standard_error": 0.2,
    }
    acceptance = {
        "iid_calibration_relative_nll_reduction_min": 0.05,
        "iid_test_relative_nll_reduction_min": 0.05,
        "iid_calibration_model_minus_uniform_nll_ucb95_max": 0.0,
        "iid_test_model_minus_uniform_nll_ucb95_max": 0.0,
        "maximum_order_elbo_mc_standard_error": 0.6,
    }
    updated = _updated_checks(
        checks,
        {
            "iid_calibration_supported": summary,
            "iid_test_supported": summary,
        },
        acceptance,
    )
    assert all(updated.values())
