from __future__ import annotations

import torch

from gaugeflow.production.assignment_data import AssignmentCarrierExample
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT
from scripts.train_h1a_assignment_iid import (
    _material_bootstrap_ucb95,
    _relabel_example,
)


def _example() -> AssignmentCarrierExample:
    source, target = torch.nonzero(~torch.eye(4, dtype=torch.bool), as_tuple=True)
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
    assert torch.equal(changed.site_features, example.site_features[order])
    assert torch.equal(changed.target_assignment, example.target_assignment[order])
    assert torch.equal(changed.composition_counts, example.composition_counts)
    assert torch.equal(changed.edge_source, inverse[example.edge_source])
    assert torch.equal(changed.edge_target, inverse[example.edge_target])
    assert torch.equal(changed.edge_rbf, example.edge_rbf)
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
