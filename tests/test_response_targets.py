import pytest
import torch

from gaugeflow.production.response_targets import (
    canonical_gamma_spectrum,
    cartesian_stiffness_to_kelvin,
    engineering_stiffness_to_kelvin,
    kelvin_stiffness_to_cartesian,
    scatter_internal_strain_blocks,
)


def test_engineering_kelvin_cartesian_stiffness_round_trip():
    generator = torch.Generator().manual_seed(4)
    raw = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    engineering = 0.5 * (raw + raw.T)
    kelvin = engineering_stiffness_to_kelvin(engineering)
    cartesian = kelvin_stiffness_to_cartesian(kelvin)
    assert torch.allclose(cartesian_stiffness_to_kelvin(cartesian), kelvin, atol=1e-12)
    assert torch.allclose(cartesian, cartesian.transpose(-4, -3), atol=1e-12)
    assert torch.allclose(cartesian, cartesian.transpose(-2, -1), atol=1e-12)
    assert torch.allclose(cartesian, cartesian.permute(2, 3, 0, 1), atol=1e-12)


def test_gamma_spectrum_is_sorted_padded_and_sign_preserving():
    values = torch.tensor([3.0, -2.0, 0.0, 1.0, -0.5, 4.0])
    target = canonical_gamma_spectrum(values, maximum_atoms=3, eigenvalue_scale=2.0)
    assert target.mask.tolist() == [True] * 6 + [False] * 3
    assert target.soft.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected = torch.log1p(torch.tensor([2.0, 0.5, 0.0, 1.0, 3.0, 4.0]).abs() / 2.0)
    assert torch.allclose(target.log_magnitude[:6], expected)


def test_partial_internal_strain_remains_partial_and_rejects_duplicates():
    blocks = torch.stack((torch.eye(3), 2.0 * torch.eye(3)))
    ions = torch.tensor([0, 2])
    directions = torch.tensor([1, 0])
    target = scatter_internal_strain_blocks(blocks, ions, directions, atom_count=3)
    assert target.mask.sum() == 18
    assert torch.equal(target.value[0, 1], torch.eye(3))
    assert torch.equal(target.value[2, 0], 2.0 * torch.eye(3))
    assert not bool(target.mask[1].any())
    with pytest.raises(ValueError, match="duplicate"):
        scatter_internal_strain_blocks(
            blocks,
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
            atom_count=3,
        )


def test_internal_strain_projects_physical_symmetry_and_audits_source_precision():
    blocks = torch.tensor([[[1.0, 0.21, 0.0], [0.19, 2.0, 0.0], [0.0, 0.0, 3.0]]])
    halfwidth = torch.full_like(blocks, 0.011)
    target = scatter_internal_strain_blocks(
        blocks,
        torch.tensor([0]),
        torch.tensor([2]),
        atom_count=1,
        rounding_halfwidth=halfwidth,
    )
    assert torch.allclose(
        target.value[0, 2, torch.tensor([0, 1]), torch.tensor([1, 0])],
        torch.full((2,), 0.2),
    )
    assert target.source_symmetric_within_rounding
    outside = scatter_internal_strain_blocks(
        blocks,
        torch.tensor([0]),
        torch.tensor([2]),
        atom_count=1,
        rounding_halfwidth=torch.full_like(blocks, 0.001),
    )
    assert not outside.source_symmetric_within_rounding
    assert outside.maximum_antisymmetric_residual == pytest.approx(0.02)
