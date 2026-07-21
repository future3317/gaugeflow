from __future__ import annotations

import itertools

import torch

from gaugeflow.production.assignment_training import (
    AssignmentCarrierBatch,
    orderless_assignment_objective,
    sample_uniform_reveal_ranks,
)
from gaugeflow.production.autoregressive_assignment import (
    GeometryAwareRemainingCountScorer,
    RemainingCountAssignmentLaw,
    complete_pair_rbf,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _counts(values: torch.Tensor) -> torch.Tensor:
    return torch.bincount(values, minlength=CHEMICAL_ELEMENT_COUNT)


def _carrier(
    model: GeometryAwareRemainingCountScorer,
    *,
    order: torch.Tensor | None = None,
) -> AssignmentCarrierBatch:
    target = torch.tensor([2, 5, 2, 5], dtype=torch.long)
    site = torch.tensor(
        [
            [0.2, -0.4, 0.7],
            [1.1, 0.3, -0.2],
            [-0.6, 0.8, 0.1],
            [0.4, -0.9, 0.5],
        ]
    )
    distance = torch.tensor(
        [
            [0.0, 0.5, 0.8, 0.6],
            [0.5, 0.0, 0.7, 0.9],
            [0.8, 0.7, 0.0, 0.4],
            [0.6, 0.9, 0.4, 0.0],
        ]
    )
    if order is not None:
        target = target[order]
        site = site[order]
        distance = distance[order][:, order]
    source, edge_target = torch.nonzero(~torch.eye(4, dtype=torch.bool), as_tuple=True)
    return AssignmentCarrierBatch(
        site_features=site,
        graph_features=torch.tensor([[0.25, -0.5]]),
        batch=torch.zeros(4, dtype=torch.long),
        edge_source=source,
        edge_target=edge_target,
        edge_rbf=complete_pair_rbf(
            distance[source, edge_target],
            radial_channels=model.radial_channels,
        ),
        composition_counts=_counts(target).unsqueeze(0),
        target_assignment=target,
        parent_space_group=torch.tensor([1], dtype=torch.long),
        cell_index=torch.tensor([1], dtype=torch.long),
    )


def _model() -> GeometryAwareRemainingCountScorer:
    torch.manual_seed(5705)
    return GeometryAwareRemainingCountScorer(
        site_feature_dim=3,
        graph_feature_dim=2,
        radial_channels=6,
        hidden_dim=24,
        message_blocks=2,
    )


def test_vectorized_objective_matches_explicit_next_site_average() -> None:
    model = _model().double()
    carrier = _carrier(model)
    carrier = AssignmentCarrierBatch(
        **{
            name: value.double() if value.is_floating_point() else value
            for name, value in carrier.__dict__.items()
        }
    )
    rank = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    vectorized = orderless_assignment_objective(model, carrier, reveal_rank=rank)
    law = RemainingCountAssignmentLaw()

    def score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        return model(
            carrier.site_features,
            carrier.graph_features,
            carrier.batch,
            carrier.edge_source,
            carrier.edge_target,
            carrier.edge_rbf,
            partial,
            carrier.composition_counts,
            remaining.unsqueeze(0),
            carrier.parent_space_group,
            carrier.cell_index,
        )

    explicit = torch.zeros((), dtype=torch.float64)
    for depth in range(4):
        partial = torch.where(
            rank < depth,
            carrier.target_assignment,
            torch.full_like(carrier.target_assignment, -1),
        )
        remaining = carrier.composition_counts[0] - torch.bincount(
            carrier.target_assignment[rank < depth],
            minlength=CHEMICAL_ELEMENT_COUNT,
        )
        logits = score(partial, remaining)
        eligible = torch.nonzero(rank >= depth, as_tuple=False).flatten()
        terms = torch.stack(
            [
                law.step_log_probabilities(logits[site], remaining)[carrier.target_assignment[site]]
                for site in eligible.tolist()
            ]
        )
        explicit = explicit + terms.mean()
    assert torch.allclose(vectorized.graph_log_probability[0], explicit, atol=1e-12, rtol=1e-12)


def test_rao_blackwellized_and_path_estimators_have_same_order_expectation() -> None:
    model = _model().double()
    carrier = _carrier(model)
    carrier = AssignmentCarrierBatch(
        **{
            name: value.double() if value.is_floating_point() else value
            for name, value in carrier.__dict__.items()
        }
    )
    law = RemainingCountAssignmentLaw()

    def score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        return model(
            carrier.site_features,
            carrier.graph_features,
            carrier.batch,
            carrier.edge_source,
            carrier.edge_target,
            carrier.edge_rbf,
            partial,
            carrier.composition_counts,
            remaining.unsqueeze(0),
            carrier.parent_space_group,
            carrier.cell_index,
        )

    path_values = []
    averaged_values = []
    for order_tuple in itertools.permutations(range(4)):
        order = torch.tensor(order_tuple, dtype=torch.long)
        rank = torch.argsort(order)
        path_values.append(
            law.path_log_probability(
                score,
                carrier.target_assignment,
                order,
                carrier.composition_counts[0],
            )
        )
        averaged_values.append(
            orderless_assignment_objective(model, carrier, reveal_rank=rank).graph_log_probability[0]
        )
    assert torch.allclose(
        torch.stack(averaged_values).mean(),
        torch.stack(path_values).mean(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_neural_subset_dp_matches_exhaustive_reveal_orders() -> None:
    model = _model().double()
    carrier = _carrier(model)
    carrier = AssignmentCarrierBatch(
        **{
            name: value.double() if value.is_floating_point() else value
            for name, value in carrier.__dict__.items()
        }
    )
    law = RemainingCountAssignmentLaw()

    def score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        return model(
            carrier.site_features,
            carrier.graph_features,
            carrier.batch,
            carrier.edge_source,
            carrier.edge_target,
            carrier.edge_rbf,
            partial,
            carrier.composition_counts,
            remaining.unsqueeze(0),
            carrier.parent_space_group,
            carrier.cell_index,
        )

    exhaustive = torch.stack(
        [
            law.path_log_probability(
                score,
                carrier.target_assignment,
                torch.tensor(order, dtype=torch.long),
                carrier.composition_counts[0],
            ).exp()
            for order in itertools.permutations(range(4))
        ]
    ).mean()
    subset = law.exact_order_marginal_probability(
        score,
        carrier.target_assignment,
        carrier.composition_counts[0],
    )
    assert abs(float(exhaustive) - subset) <= 1e-12


def test_vectorized_objective_is_relabel_consistent_and_has_finite_gradients() -> None:
    model = _model()
    carrier = _carrier(model)
    rank = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    reference = orderless_assignment_objective(model, carrier, reveal_rank=rank)
    relabel = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    changed = orderless_assignment_objective(
        model,
        _carrier(model, order=relabel),
        reveal_rank=rank[relabel],
    )
    assert torch.allclose(changed.graph_nll, reference.graph_nll, atol=1e-6, rtol=1e-6)

    reference.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(torch.linalg.vector_norm(value)) for value in gradients) > 0.0


def test_reveal_order_sampling_is_target_independent_and_graphwise_permuted() -> None:
    batch = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    first = sample_uniform_reveal_ranks(batch, generator=torch.Generator().manual_seed(11))
    second = sample_uniform_reveal_ranks(batch, generator=torch.Generator().manual_seed(11))
    assert torch.equal(first, second)
    for graph in range(3):
        selected = first[batch == graph]
        assert torch.equal(torch.sort(selected).values, torch.arange(selected.numel()))
