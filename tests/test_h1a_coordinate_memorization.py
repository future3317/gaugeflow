import torch

from scripts.audit_h1a_coordinate_memorization import _coordinate_loss, _fixed_indices


def test_fixed_indices_are_reproducible_and_exclude_training_panel():
    first = _fixed_indices(100, 16, 41)
    second = _fixed_indices(100, 16, 41)
    unseen = _fixed_indices(100, 16, 42, excluded=first)
    assert torch.equal(first, second)
    assert torch.unique(first).numel() == 16
    assert not torch.isin(unseen, first).any()


def test_coordinate_loss_is_equal_weighted_over_graphs():
    batch = torch.tensor([0, 1, 1], dtype=torch.long)
    target = torch.zeros((3, 3))
    prediction = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
    # Each graph has mean squared vector error 9; division by three gives 3.
    assert torch.equal(_coordinate_loss(prediction, target, batch, 2), torch.tensor(3.0))
