import torch

from scripts.train_h1a_coordinate_variable_projection import assign_affine_readout


def test_assign_affine_readout_copies_only_declared_parameters():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(2, 1)
    model.second = torch.nn.Linear(1, 1)
    untouched = model.second.weight.detach().clone()
    solution = torch.tensor([1.0, 2.0, 3.0])
    assign_affine_readout(model, solution, ["first.weight", "first.bias"])
    assert torch.equal(model.first.weight, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(model.first.bias, torch.tensor([3.0]))
    assert torch.equal(model.second.weight, untouched)
