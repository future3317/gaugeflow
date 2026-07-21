from __future__ import annotations

import itertools

import torch

from gaugeflow.production.assignment_training import (
    AssignmentCarrierBatch,
    orderless_assignment_objective,
    sample_parent_orbit_representatives,
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


def test_vectorized_path_objective_matches_explicit_sequential_evaluation() -> None:
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
    reveal_order = torch.argsort(rank)
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

    explicit = law.path_log_probability(
        score,
        carrier.target_assignment,
        reveal_order,
        carrier.composition_counts[0],
    )
    assert torch.allclose(vectorized.graph_log_probability[0], explicit, atol=1e-12, rtol=1e-12)


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


def _two_graph_orbit_inputs() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    target = torch.tensor([2, 5, 2, 5, 3, 7, 7], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    actions = (
        torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 1],
            ],
            dtype=torch.long,
        ),
    )
    return target, batch, actions


def test_parent_orbit_sampling_preserves_counts_and_uses_only_unique_orbit() -> None:
    target, batch, actions = _two_graph_orbit_inputs()
    for seed in range(32):
        sampled = sample_parent_orbit_representatives(
            target,
            batch,
            actions,
            generator=torch.Generator().manual_seed(seed),
        )
        for graph, action in enumerate(actions):
            selected = batch == graph
            local_target = target[selected]
            orbit = torch.unique(local_target[action], dim=0)
            assert torch.any(torch.all(sampled[selected] == orbit, dim=1))
            assert torch.equal(
                torch.sort(sampled[selected]).values,
                torch.sort(local_target).values,
            )


def test_parent_orbit_sampling_ignores_duplicate_operations_and_enumeration_order() -> None:
    target, batch, actions = _two_graph_orbit_inputs()
    expanded = (
        torch.cat((actions[0][[2, 0, 3, 1]], actions[0][[1, 1, 3]]), dim=0),
        torch.cat((actions[1].flip(0), actions[1][[0, 0]]), dim=0),
    )
    for seed in range(32):
        reference = sample_parent_orbit_representatives(
            target,
            batch,
            actions,
            generator=torch.Generator().manual_seed(seed),
        )
        changed = sample_parent_orbit_representatives(
            target,
            batch,
            expanded,
            generator=torch.Generator().manual_seed(seed),
        )
        assert torch.equal(changed, reference)


def test_parent_orbit_sampling_is_node_relabel_equivariant() -> None:
    target = torch.tensor([2, 5, 2, 5], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    action = _two_graph_orbit_inputs()[2][0]
    relabel = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inverse = torch.argsort(relabel)
    relabeled_action = inverse[action[:, relabel]]
    original_orbit = torch.unique(target[action], dim=0)
    relabeled_orbit = torch.unique(target[relabel][relabeled_action], dim=0)
    assert torch.equal(relabeled_orbit, torch.unique(original_orbit[:, relabel], dim=0))

    for seed in range(32):
        reference = sample_parent_orbit_representatives(
            target,
            batch,
            (action,),
            generator=torch.Generator().manual_seed(seed),
        )
        changed = sample_parent_orbit_representatives(
            target[relabel],
            batch,
            (relabeled_action,),
            generator=torch.Generator().manual_seed(seed),
        )
        assert torch.equal(changed, reference[relabel])


def test_orbit_representative_and_reveal_order_use_independent_random_draws() -> None:
    target, batch, actions = _two_graph_orbit_inputs()
    alternative = target.clone()
    alternative[:4] = torch.tensor([11, 13, 11, 13])
    ranks = []
    for label in (target, alternative):
        generator = torch.Generator().manual_seed(2718)
        sample_parent_orbit_representatives(label, batch, actions, generator=generator)
        ranks.append(sample_uniform_reveal_ranks(batch, generator=generator))
    assert torch.equal(ranks[0], ranks[1])
