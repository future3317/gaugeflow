from __future__ import annotations

import itertools

import pytest
import torch

from gaugeflow.production.composition_assignment import (
    CountConstrainedAssignmentLaw,
    composition_counts_from_tokens,
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
