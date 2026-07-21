import math

import torch

from gaugeflow.production.orderless_product_state import (
    orderless_next_reveal_nll,
    partial_occupation_from_reveal_rank,
    sample_orderless_partial_occupation,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assignment = torch.tensor([0, 1, 0, 2, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    counts = torch.tensor([[2, 1, 0, 0], [0, 0, 2, 0]], dtype=torch.long)
    reveal_rank = torch.tensor([2, 0, 1, 1, 0], dtype=torch.long)
    reveal_count = torch.tensor([1, 1], dtype=torch.long)
    return assignment, batch, counts, reveal_rank, reveal_count


def test_partial_state_preserves_exact_remaining_counts_and_selects_one_next_site() -> None:
    assignment, batch, counts, rank, depth = _fixture()
    state = partial_occupation_from_reveal_rank(
        assignment,
        batch,
        counts,
        rank,
        depth,
        vocabulary_size=4,
        mask_token=4,
    )
    assert torch.equal(state.partial_tokens, torch.tensor([4, 1, 4, 4, 2]))
    assert torch.equal(state.remaining_counts, torch.tensor([[2, 0, 0, 0], [0, 0, 1, 0]]))
    assert torch.equal(state.next_site, torch.tensor([2, 3]))
    assert torch.equal(state.next_token, torch.tensor([0, 2]))


def test_zero_logits_use_exchangeable_remaining_count_base_measure() -> None:
    assignment, batch, counts, rank, depth = _fixture()
    state = partial_occupation_from_reveal_rank(
        assignment,
        batch,
        counts,
        rank,
        depth,
        vocabulary_size=4,
        mask_token=4,
    )
    graph_nll, mean_nll = orderless_next_reveal_nll(
        torch.zeros((assignment.numel(), 4)), state, vocabulary_size=4
    )
    # Both pending targets are the only legal species after the one reveal.
    assert torch.allclose(graph_nll, torch.zeros_like(graph_nll))
    assert float(mean_nll) == 0.0

    # At depth zero, species 0 has two of three remaining slots in graph 0.
    state_zero = partial_occupation_from_reveal_rank(
        assignment,
        batch,
        counts,
        rank,
        torch.zeros(2, dtype=torch.long),
        vocabulary_size=4,
        mask_token=4,
    )
    graph_nll, _ = orderless_next_reveal_nll(
        torch.zeros((assignment.numel(), 4)), state_zero, vocabulary_size=4
    )
    assert math.isclose(float(graph_nll[0]), -math.log(2.0 / 3.0), rel_tol=0.0, abs_tol=1.0e-7)
    assert float(graph_nll[1]) == 0.0


def test_partial_state_is_equivariant_under_packed_node_relabeling() -> None:
    assignment, batch, counts, rank, depth = _fixture()
    logits = torch.tensor(
        [[0.3, -0.5, 0.1, -0.2], [0.0, 0.4, -0.2, 0.5], [0.2, -0.1, 0.7, 0.0],
         [-0.4, 0.5, 0.6, 0.1], [0.2, -0.3, 0.9, -0.1]]
    )
    state = partial_occupation_from_reveal_rank(
        assignment, batch, counts, rank, depth, vocabulary_size=4, mask_token=4
    )
    expected, _ = orderless_next_reveal_nll(logits, state, vocabulary_size=4)

    # Permute sites within each packed graph and transport all node fields.
    order = torch.tensor([2, 0, 1, 4, 3], dtype=torch.long)
    changed = partial_occupation_from_reveal_rank(
        assignment[order], batch[order], counts, rank[order], depth, vocabulary_size=4, mask_token=4
    )
    actual, _ = orderless_next_reveal_nll(logits[order], changed, vocabulary_size=4)
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-7)


def test_time_parameterized_partial_state_always_leaves_a_legal_next_reveal() -> None:
    assignment, batch, counts, _, _ = _fixture()
    time = torch.tensor([0.0, 1.0])
    state = sample_orderless_partial_occupation(
        assignment,
        batch,
        counts,
        time,
        generator=torch.Generator().manual_seed(5705),
        vocabulary_size=4,
        mask_token=4,
    )
    # t=0 reveals all but one site; t=1 reveals none.
    assert torch.equal(state.reveal_count, torch.tensor([2, 0]))
    assert torch.equal(state.remaining_counts.sum(dim=1), torch.tensor([1, 2]))
