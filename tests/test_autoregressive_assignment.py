from __future__ import annotations

import itertools
import math

import pytest
import torch

from gaugeflow.production.autoregressive_assignment import (
    GeometryAwareRemainingCountScorer,
    RemainingCountAssignmentLaw,
    complete_pair_rbf,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _counts(*values: int) -> torch.Tensor:
    output = torch.zeros(CHEMICAL_ELEMENT_COUNT, dtype=torch.long)
    for value in values:
        output[value] += 1
    return output


def _zero_score(sites: int):
    def score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        del partial, remaining
        return torch.zeros(sites, CHEMICAL_ELEMENT_COUNT, dtype=torch.float64)

    return score


def test_order_marginal_law_is_exactly_normalized_and_count_exact() -> None:
    law = RemainingCountAssignmentLaw()
    counts = _counts(0, 0, 1, 1)
    assignments = sorted(set(itertools.permutations((0, 0, 1, 1))))
    probabilities = [
        law.exact_order_marginal_probability(
            _zero_score(4),
            torch.tensor(assignment, dtype=torch.long),
            counts,
        )
        for assignment in assignments
    ]
    assert all(abs(value - 1.0 / 6.0) <= 1e-12 for value in probabilities)
    assert abs(sum(probabilities) - 1.0) <= 1e-12

    generator = torch.Generator().manual_seed(5705)
    for _ in range(32):
        order = torch.randperm(4, generator=generator)
        sampled = law.sample(_zero_score(4), counts, order, generator=generator)
        assert torch.equal(torch.bincount(sampled, minlength=CHEMICAL_ELEMENT_COUNT), counts)


def test_batched_step_law_matches_individual_steps() -> None:
    law = RemainingCountAssignmentLaw()
    generator = torch.Generator().manual_seed(5705)
    logits = torch.randn(3, CHEMICAL_ELEMENT_COUNT, generator=generator)
    remaining = torch.stack((_counts(0, 0, 1), _counts(2, 3, 3), _counts(4)))
    batched = law.batched_step_log_probabilities(logits, remaining)
    expected = torch.stack(
        [law.step_log_probabilities(logits[index], remaining[index]) for index in range(3)]
    )
    assert torch.equal(batched, expected)


def test_unique_orbit_probability_deduplicates_group_operations() -> None:
    law = RemainingCountAssignmentLaw()
    counts = _counts(0, 0, 1, 1)
    target = torch.tensor([0, 0, 1, 1])
    action_with_duplicate = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    probability = law.exact_quotient_probability(
        _zero_score(4),
        target,
        counts,
        action_with_duplicate,
    )
    assert abs(probability - 4.0 / 6.0) <= 1e-12


def _complete_edges(distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = distance.shape[0]
    source, target = torch.nonzero(
        ~torch.eye(nodes, dtype=torch.bool, device=distance.device),
        as_tuple=True,
    )
    rbf = complete_pair_rbf(distance[source, target], radial_channels=6)
    return source, target, rbf


def test_geometry_scorer_is_node_permutation_equivariant_with_finite_gradients() -> None:
    torch.manual_seed(5705)
    nodes = 4
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=7,
        graph_feature_dim=5,
        radial_channels=6,
        hidden_dim=24,
        message_blocks=2,
    )
    site = torch.randn(nodes, 7)
    graph = torch.randn(1, 5)
    batch = torch.zeros(nodes, dtype=torch.long)
    distance = torch.tensor(
        [
            [0.0, 0.5, math.sqrt(0.5), 0.5],
            [0.5, 0.0, 0.5, math.sqrt(0.5)],
            [math.sqrt(0.5), 0.5, 0.0, 0.5],
            [0.5, math.sqrt(0.5), 0.5, 0.0],
        ]
    )
    source, target, rbf = _complete_edges(distance)
    partial = torch.tensor([0, -1, 1, -1], dtype=torch.long)
    composition = _counts(0, 0, 1, 1).unsqueeze(0)
    remaining = _counts(0, 1).unsqueeze(0)
    parent = torch.tensor([1], dtype=torch.long)
    cell = torch.tensor([1], dtype=torch.long)

    logits = model(
        site,
        graph,
        batch,
        source,
        target,
        rbf,
        partial,
        composition,
        remaining,
        parent,
        cell,
    )
    order = torch.tensor([2, 0, 3, 1])
    changed_source, changed_target, changed_rbf = _complete_edges(distance[order][:, order])
    changed = model(
        site[order],
        graph,
        batch,
        changed_source,
        changed_target,
        changed_rbf,
        partial[order],
        composition,
        remaining,
        parent,
        cell,
    )
    assert torch.allclose(changed, logits[order], atol=1e-6, rtol=1e-6)

    law = RemainingCountAssignmentLaw()
    loss = -law.step_log_probabilities(logits[1], remaining[0])[1]
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(torch.linalg.vector_norm(value)) for value in gradients) > 0.0


def test_residual_stabilizer_related_unassigned_sites_have_equal_logits() -> None:
    torch.manual_seed(5706)
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=3,
        graph_feature_dim=2,
        radial_channels=6,
        hidden_dim=16,
        message_blocks=2,
    )
    nodes = 4
    distance = torch.tensor(
        [
            [0.0, 0.5, math.sqrt(0.5), 0.5],
            [0.5, 0.0, 0.5, math.sqrt(0.5)],
            [math.sqrt(0.5), 0.5, 0.0, 0.5],
            [0.5, math.sqrt(0.5), 0.5, 0.0],
        ]
    )
    source, target, rbf = _complete_edges(distance)
    logits = model(
        torch.ones(nodes, 3),
        torch.zeros(1, 2),
        torch.zeros(nodes, dtype=torch.long),
        source,
        target,
        rbf,
        torch.tensor([0, -1, 0, -1]),
        _counts(0, 0, 1, 1).unsqueeze(0),
        _counts(1, 1).unsqueeze(0),
        torch.tensor([1]),
        torch.tensor([1]),
    )
    assert torch.allclose(logits[1], logits[3], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_geometry_scorer_cuda_bfloat16_autocast_is_finite() -> None:
    device = torch.device("cuda")
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=3,
        graph_feature_dim=2,
        radial_channels=6,
        hidden_dim=16,
        message_blocks=2,
    ).to(device)
    nodes = 4
    distance = torch.tensor(
        [
            [0.0, 0.5, math.sqrt(0.5), 0.5],
            [0.5, 0.0, 0.5, math.sqrt(0.5)],
            [math.sqrt(0.5), 0.5, 0.0, 0.5],
            [0.5, math.sqrt(0.5), 0.5, 0.0],
        ],
        device=device,
    )
    source, target, rbf = _complete_edges(distance)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(
            torch.ones(nodes, 3, device=device),
            torch.zeros(1, 2, device=device),
            torch.zeros(nodes, dtype=torch.long, device=device),
            source,
            target,
            rbf,
            torch.tensor([0, -1, 0, -1], dtype=torch.long, device=device),
            _counts(0, 0, 1, 1).unsqueeze(0).to(device),
            _counts(1, 1).unsqueeze(0).to(device),
            torch.tensor([1], device=device),
            torch.tensor([1], device=device),
        )
    assert torch.isfinite(logits).all()
