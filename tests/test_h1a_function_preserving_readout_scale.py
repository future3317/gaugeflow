import torch

from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from scripts.audit_h1a_function_preserving_readout_scale import (
    quotient_solution,
    unscaled_reference,
)


def test_unscaled_reference_reverses_powers_of_two_parameterization():
    model = HybridCrystalDenoiser(
        hidden_dim=32, vector_dim=8, layers=1, radial_dim=4
    )
    scaled_vector = model.coordinate_vector_head.weight.detach().clone()
    reference = unscaled_reference(model, 1024.0)
    assert model.coordinate_readout_scale == 1024.0
    assert reference.coordinate_readout_scale == 1.0
    assert torch.equal(reference.coordinate_vector_head.weight, scaled_vector * 1024.0)
    assert torch.equal(model.coordinate_vector_head.weight, scaled_vector)


def test_quotient_solution_fits_full_rank_target():
    jacobian = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    desired = torch.tensor([3.0, -4.0])
    solution, metrics = quotient_solution(jacobian, desired, threshold=1e-10)
    assert metrics["rank"] == 2
    assert torch.allclose(jacobian.double() @ solution, desired.double())
