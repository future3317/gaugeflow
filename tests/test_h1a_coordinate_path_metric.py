import torch

from scripts.audit_h1a_coordinate_path_metric import (
    physical_fractional_noise_covariance,
    unimodular_shear_covariance_relative_difference,
)


def test_fractional_noise_induces_lattice_metric_in_cartesian_space():
    lattice = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)).unsqueeze(0)
    observed = physical_fractional_noise_covariance(lattice, 0.2)
    expected = torch.diag(torch.tensor([0.16, 0.36, 1.0], dtype=torch.float64)).unsqueeze(0)
    assert torch.allclose(observed, expected)


def test_fractional_isotropic_path_changes_under_unimodular_shear_but_not_axis_swap():
    lattice = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)).unsqueeze(0)
    shear = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    swap = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    assert float(unimodular_shear_covariance_relative_difference(lattice, shear)) > 0.1
    assert torch.allclose(
        unimodular_shear_covariance_relative_difference(lattice, swap),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
    )
