from __future__ import annotations

import torch

from gaugeflow.production.assignment_scorer import (
    OrbitAwareAssignmentScorer,
    faithful_parent_action,
    parent_action_site_features,
    parent_carrier_graph_features,
)
from gaugeflow.production.composition_assignment import CountConstrainedAssignmentLaw
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _cyclic_action() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )


def _relabel_action(action: torch.Tensor, relabel: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(relabel)
    inverse[relabel] = torch.arange(relabel.numel())
    return inverse[action[:, relabel]]


def test_parent_action_features_are_kernel_and_relabeling_invariant() -> None:
    action = _cyclic_action()
    image = faithful_parent_action(action)
    assert image.shape == (4, 4)

    relabel = torch.tensor([2, 0, 3, 1])
    original = parent_action_site_features(action)
    transformed = parent_action_site_features(_relabel_action(action, relabel))
    assert torch.allclose(transformed, original[relabel], atol=0.0, rtol=0.0)


def test_parent_graph_features_ignore_row_order_and_physical_rotation() -> None:
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]]
    )
    lattice = torch.diag(torch.tensor([3.0, 3.0, 4.0]))
    action = _cyclic_action()
    reference = parent_carrier_graph_features(
        fractional,
        lattice,
        action,
        cell_index=1,
    )

    relabel = torch.tensor([2, 0, 3, 1])
    angle = torch.tensor(0.37)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = parent_carrier_graph_features(
        fractional[relabel],
        lattice @ rotation.T,
        _relabel_action(action, relabel),
        cell_index=1,
    )
    assert torch.allclose(transformed, reference, atol=2e-6, rtol=2e-6)


def test_assignment_scorer_preserves_complete_quotient_probability() -> None:
    torch.manual_seed(7)
    action = _cyclic_action()
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]]
    )
    lattice = torch.diag(torch.tensor([3.0, 3.0, 4.0]))
    site_features = parent_action_site_features(action)
    graph_features = parent_carrier_graph_features(
        fractional,
        lattice,
        action,
        cell_index=1,
    ).unsqueeze(0)
    counts = torch.zeros((1, CHEMICAL_ELEMENT_COUNT), dtype=torch.long)
    counts[0, 4] = 2
    counts[0, 7] = 2
    assignment = torch.tensor([4, 7, 4, 7], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    scorer = OrbitAwareAssignmentScorer(hidden_dim=32)
    scores = scorer(
        site_features,
        graph_features,
        batch,
        counts,
        torch.tensor([123]),
        torch.tensor([1]),
    )
    law = CountConstrainedAssignmentLaw()
    reference = law.quotient_log_prob(scores, batch, counts, assignment, [action])

    relabel = torch.tensor([2, 0, 3, 1])
    relabeled_action = _relabel_action(action, relabel)
    relabeled_scores = scorer(
        parent_action_site_features(relabeled_action),
        graph_features,
        batch,
        counts,
        torch.tensor([123]),
        torch.tensor([1]),
    )
    transformed = law.quotient_log_prob(
        relabeled_scores,
        batch,
        counts,
        assignment[relabel],
        [relabeled_action],
    )
    assert torch.allclose(relabeled_scores, scores[relabel], atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        transformed.log_probability,
        reference.log_probability,
        atol=2e-6,
        rtol=2e-6,
    )

    (-reference.log_probability.mean()).backward()
    gradients = [parameter.grad for parameter in scorer.parameters() if parameter.requires_grad]
    assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
