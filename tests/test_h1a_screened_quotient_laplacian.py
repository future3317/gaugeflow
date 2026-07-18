import torch

from scripts.audit_h1a_screened_quotient_laplacian import (
    _spectrum_and_solution,
    helmert_quotient_basis,
)


def test_helmert_quotient_basis_removes_translation():
    basis = helmert_quotient_basis(3, dtype=torch.float64, device=torch.device("cpu"))
    assert basis.shape == (9, 6)
    assert torch.allclose(basis.T @ basis, torch.eye(6, dtype=torch.float64))
    assert torch.allclose(
        basis.T @ torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).repeat(3),
        torch.zeros(6, dtype=torch.float64),
    )


def test_spectrum_solution_fits_full_rank_target():
    jacobian = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    desired = torch.tensor([3.0, -4.0])
    solution, metrics = _spectrum_and_solution(
        jacobian, desired, threshold=1e-10
    )
    assert metrics["rank"] == 2
    assert torch.allclose(jacobian.double() @ solution, desired.double())
