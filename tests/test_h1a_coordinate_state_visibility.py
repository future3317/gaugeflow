import torch

from scripts.audit_h1a_coordinate_state_visibility import (
    type_preserving_cycle_permutation,
)


def test_type_preserving_cycle_permutation_is_vectorized_and_graph_local():
    elements = torch.tensor([4, 4, 6, 4, 6, 6, 6], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    permutation = type_preserving_cycle_permutation(elements, batch)
    assert torch.equal(permutation, torch.tensor([1, 0, 2, 3, 5, 6, 4]))
    assert torch.equal(elements[permutation], elements)
    assert torch.equal(batch[permutation], batch)
    assert torch.equal(torch.sort(permutation).values, torch.arange(7))


def test_type_preserving_cycle_permutation_handles_empty_input():
    empty = torch.empty(0, dtype=torch.long)
    assert type_preserving_cycle_permutation(empty, empty).numel() == 0
