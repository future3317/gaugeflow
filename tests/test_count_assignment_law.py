from __future__ import annotations

import itertools

import pytest
import torch

from gaugeflow.production.composition_assignment import (
    CountConstrainedAssignmentLaw,
    composition_counts_from_tokens,
    occupation_block_composition_feasible,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _scores(nodes: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    torch.manual_seed(101)
    return torch.randn(nodes, CHEMICAL_ELEMENT_COUNT, dtype=dtype)


def _counts(values: dict[int, int]) -> torch.Tensor:
    output = torch.zeros((1, CHEMICAL_ELEMENT_COUNT), dtype=torch.long)
    for token, count in values.items():
        output[0, token] = count
    return output


def test_exact_count_assignment_probability_normalizes_without_factorial_enumeration() -> None:
    scores = _scores(4).requires_grad_()
    batch = torch.zeros(4, dtype=torch.long)
    counts = _counts({1: 2, 7: 1, 11: 1})
    unique = sorted(set(itertools.permutations([1, 1, 7, 11])))
    assignments = torch.tensor(unique, dtype=torch.long)
    logp = torch.stack(
        [
            CountConstrainedAssignmentLaw().log_prob(
                scores, batch, counts, assignment
            ).log_probability[0]
            for assignment in assignments
        ]
    )
    assert torch.allclose(
        torch.logsumexp(logp, dim=0),
        torch.zeros((), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    (-logp[0]).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_assignment_law_is_site_permutation_equivariant_and_count_exact() -> None:
    scores = _scores(5)
    batch = torch.zeros(5, dtype=torch.long)
    counts = _counts({2: 3, 9: 2})
    assignment = torch.tensor([2, 9, 2, 9, 2])
    law = CountConstrainedAssignmentLaw()
    reference = law.log_prob(scores, batch, counts, assignment)
    permutation = torch.tensor([4, 1, 3, 0, 2])
    transformed = law.log_prob(
        scores[permutation], batch, counts, assignment[permutation]
    )
    assert torch.allclose(reference.log_probability, transformed.log_probability, atol=1e-12)
    sample = law.sample(
        scores,
        batch,
        counts,
        generator=torch.Generator().manual_seed(103),
    )
    assert torch.equal(composition_counts_from_tokens(sample.tokens, batch, 1), counts)
    recomputed = law.log_prob(scores, batch, counts, sample.tokens)
    assert torch.allclose(sample.log_probability, recomputed.log_probability, atol=1e-12)


def test_assignment_mode_uses_global_max_sum_not_greedy_conditional_mode() -> None:
    scores = torch.full((3, CHEMICAL_ELEMENT_COUNT), -100.0, dtype=torch.float64)
    scores[0, 1] = 0.0
    scores[0, 2] = 0.0
    scores[1:, 1] = -0.1
    scores[1:, 2] = 0.0
    batch = torch.zeros(3, dtype=torch.long)
    counts = _counts({1: 1, 2: 2})
    law = CountConstrainedAssignmentLaw()

    mode = law.sample(scores, batch, counts, mode=True)
    assignments = torch.tensor(
        sorted(set(itertools.permutations([1, 2, 2]))),
        dtype=torch.long,
    )
    logp = torch.stack(
        [law.log_prob(scores, batch, counts, value).log_probability[0] for value in assignments]
    )
    assert torch.equal(mode.tokens, assignments[torch.argmax(logp)])
    assert torch.allclose(mode.log_probability[0], logp.max(), atol=1e-12, rtol=0.0)


def test_indivisible_occupation_blocks_enforce_legal_multiplicity() -> None:
    scores = _scores(4)
    batch = torch.zeros(4, dtype=torch.long)
    blocks = torch.tensor([8, 8, 3, 3], dtype=torch.long)
    law = CountConstrainedAssignmentLaw()
    legal_counts = _counts({4: 2, 6: 2})
    legal = law.sample(scores, batch, legal_counts, block_index=blocks, mode=True)
    assert legal.tokens[0] == legal.tokens[1]
    assert legal.tokens[2] == legal.tokens[3]
    with pytest.raises(ValueError, match="incompatible"):
        law.sample(scores, batch, _counts({4: 3, 6: 1}), block_index=blocks)


def test_occupation_block_support_matches_complete_assignment_enumeration() -> None:
    block_multiplicity = torch.tensor([2, 2, 1], dtype=torch.long)
    for first in range(6):
        counts = torch.tensor([first, 5 - first], dtype=torch.long)
        expected = any(
            sum(
                int(block_multiplicity[index])
                for index, token in enumerate(labeling)
                if token == 0
            )
            == first
            for labeling in itertools.product((0, 1), repeat=3)
        )
        observed = occupation_block_composition_feasible(counts, block_multiplicity)
        assert observed is expected


def test_occupation_block_support_rejects_mismatched_total_without_sampling() -> None:
    assert not occupation_block_composition_feasible(
        torch.tensor([2, 1], dtype=torch.long),
        torch.tensor([2, 2], dtype=torch.long),
    )


def test_parent_quotient_deduplicates_group_elements_and_cif_rows() -> None:
    scores = _scores(2)
    batch = torch.zeros(2, dtype=torch.long)
    counts = _counts({0: 1, 5: 1})
    assignment = torch.tensor([0, 5])
    base_group = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    duplicated_group = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.long)
    law = CountConstrainedAssignmentLaw()
    base = law.quotient_log_prob(scores, batch, counts, assignment, [base_group])
    duplicate = law.quotient_log_prob(
        scores, batch, counts, assignment, [duplicated_group]
    )
    assert torch.allclose(base.log_probability, torch.zeros(1, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(base.log_probability, duplicate.log_probability, atol=1e-12)


def test_vectorized_quotient_matches_explicit_orbit_with_blocks_and_fp32_gradient() -> None:
    scores = _scores(4, dtype=torch.float32).requires_grad_()
    batch = torch.zeros(4, dtype=torch.long)
    counts = _counts({3: 2, 8: 2})
    assignment = torch.tensor([3, 3, 8, 8], dtype=torch.long)
    blocks = torch.tensor([4, 4, 9, 9], dtype=torch.long)
    group = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
        dtype=torch.long,
    )
    law = CountConstrainedAssignmentLaw()
    observed = law.quotient_log_prob(
        scores,
        batch,
        counts,
        assignment,
        [group[[2, 0, 3, 1, 2]]],
        block_index=blocks,
    )
    labelings = torch.unique(assignment[group], dim=0)
    explicit = torch.logsumexp(
        torch.stack(
            [
                law.log_prob(
                    scores,
                    batch,
                    counts,
                    labeling,
                    block_index=blocks,
                ).log_probability[0]
                for labeling in labelings
            ]
        ),
        dim=0,
    )
    assert torch.allclose(observed.log_probability[0], explicit, atol=2e-6, rtol=2e-6)
    (-observed.log_probability.mean()).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_exact_assignment_entropy_matches_enumeration() -> None:
    scores = _scores(4)
    batch = torch.zeros(4, dtype=torch.long)
    counts = _counts({1: 2, 7: 1, 11: 1})
    assignments = torch.tensor(
        sorted(set(itertools.permutations([1, 1, 7, 11]))),
        dtype=torch.long,
    )
    law = CountConstrainedAssignmentLaw()
    logp = torch.stack(
        [law.log_prob(scores, batch, counts, value).log_probability[0] for value in assignments]
    )
    expected = -(logp.exp() * logp).sum()
    observed = law.entropy(scores, batch, counts)[0]
    assert torch.allclose(observed, expected, atol=1e-11, rtol=1e-11)


def test_quotient_rejects_parent_action_that_splits_an_indivisible_block() -> None:
    scores = _scores(4)
    batch = torch.zeros(4, dtype=torch.long)
    counts = _counts({3: 2, 8: 2})
    assignment = torch.tensor([3, 3, 8, 8], dtype=torch.long)
    blocks = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    invalid_for_blocks = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="does not preserve"):
        CountConstrainedAssignmentLaw().quotient_log_prob(
            scores,
            batch,
            counts,
            assignment,
            [invalid_for_blocks],
            block_index=blocks,
        )


def test_quotient_rejects_block_split_even_when_target_species_hide_it() -> None:
    scores = _scores(4)
    batch = torch.zeros(4, dtype=torch.long)
    counts = _counts({3: 2, 8: 2})
    assignment = torch.tensor([3, 3, 8, 8], dtype=torch.long)
    blocks = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    hidden_split = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="does not preserve"):
        CountConstrainedAssignmentLaw().quotient_log_prob(
            scores,
            batch,
            counts,
            assignment,
            [hidden_split],
            block_index=blocks,
        )
