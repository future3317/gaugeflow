import numpy as np
import pytest
import torch

from gaugeflow.model import safe_norm
from gaugeflow.tensor import (
    fixed_lossless_response_probes,
    icosahedral_response_probes,
    piezo_cartesian_to_voigt,
    response_probe_measurement_matrix,
    rotate_rank3,
)
from gaugeflow.unit_cell import transform_row_lattice_basis
from gaugeflow.vocabulary import (
    CHEMICAL_ELEMENT_COUNT,
    MASK_TOKEN,
    atomic_numbers_to_tokens,
    tokens_to_atomic_numbers,
    validate_type_tokens,
)


def test_dense_chemical_vocabulary_roundtrip_and_mask_separation():
    atomic_numbers = torch.tensor([1, 6, 7, 49, 118])
    tokens = atomic_numbers_to_tokens(atomic_numbers)
    assert torch.equal(tokens, torch.tensor([0, 5, 6, 48, 117]))
    assert torch.equal(tokens_to_atomic_numbers(tokens), atomic_numbers)
    assert validate_type_tokens(torch.tensor([0, 117, MASK_TOKEN]), allow_mask=True).dtype == torch.long
    with pytest.raises(ValueError):
        atomic_numbers_to_tokens(torch.tensor([0]))
    with pytest.raises(ValueError):
        tokens_to_atomic_numbers(torch.tensor([MASK_TOKEN]))
    with pytest.raises(ValueError):
        validate_type_tokens(torch.tensor([CHEMICAL_ELEMENT_COUNT]))


def test_safe_norm_is_finite_at_zero_and_has_nonflat_near_zero_gradient():
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    safe_norm(zero).backward()
    assert torch.isfinite(zero.grad).all()
    assert torch.equal(zero.grad, torch.zeros_like(zero))

    tiny = torch.tensor([1e-10, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    safe_norm(tiny).backward()
    assert torch.isfinite(tiny.grad).all()
    assert tiny.grad[0] > 0.99


def test_engineering_voigt_constitutive_action_and_rotation_are_independent():
    # Construct the Cartesian tensor directly; do not obtain it from a Voigt helper.
    tensor = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.2, 0.7, 0.4], [-0.1, 0.4, -0.3]],
            [[-0.2, 0.6, 0.3], [0.6, 0.1, -0.5], [0.3, -0.5, 0.8]],
            [[0.4, -0.7, 0.2], [-0.7, 0.9, 0.6], [0.2, 0.6, -0.4]],
        ],
        dtype=torch.float64,
    )
    strain = torch.tensor(
        [[0.3, -0.1, 0.2], [-0.1, -0.4, 0.5], [0.2, 0.5, 0.7]], dtype=torch.float64
    )
    engineering_strain = torch.tensor(
        [strain[0, 0], strain[1, 1], strain[2, 2], 2 * strain[1, 2], 2 * strain[0, 2], 2 * strain[0, 1]]
    )
    direct_polarization = torch.einsum("ijk,jk->i", tensor, strain)
    voigt = piezo_cartesian_to_voigt(tensor)
    assert torch.allclose(voigt @ engineering_strain, direct_polarization, atol=1e-12, rtol=1e-12)

    theta = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta), 0.0], [torch.sin(theta), torch.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated_tensor = rotate_rank3(tensor, rotation)
    rotated_strain = rotation @ strain @ rotation.T
    rotated_eta = torch.tensor(
        [rotated_strain[0, 0], rotated_strain[1, 1], rotated_strain[2, 2],
         2 * rotated_strain[1, 2], 2 * rotated_strain[0, 2], 2 * rotated_strain[0, 1]]
    )
    assert torch.allclose(
        piezo_cartesian_to_voigt(rotated_tensor) @ rotated_eta,
        rotation @ direct_polarization,
        atol=1e-12,
        rtol=1e-12,
    )


def test_row_lattice_gl3z_basis_change_preserves_cartesian_sites_for_det_minus_one():
    lattice = np.array([[3.1, 0.2, 0.1], [0.7, 4.0, -0.3], [0.2, 0.5, 5.3]])
    fractional = np.array([[0.10, 0.20, 0.30], [0.45, 0.65, 0.15]])
    basis = np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 1]])
    assert round(np.linalg.det(basis)) == -1
    transformed_lattice, transformed_fractional = transform_row_lattice_basis(lattice, fractional, basis)
    assert np.allclose(transformed_fractional @ transformed_lattice, fractional @ lattice)
    with pytest.raises(ValueError):
        transform_row_lattice_basis(lattice, fractional, np.diag([2, 1, 1]))


def test_response_probe_measurement_rank_and_conditioning_are_pinned():
    axis_face = response_probe_measurement_matrix(fixed_lossless_response_probes(dtype=torch.float64))
    icosahedral = response_probe_measurement_matrix(icosahedral_response_probes(dtype=torch.float64))
    assert torch.linalg.matrix_rank(axis_face) == 6
    assert torch.linalg.matrix_rank(icosahedral) == 6
    assert torch.allclose(torch.linalg.cond(axis_face), torch.tensor(3.225504926677693, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(torch.linalg.cond(icosahedral), torch.tensor(1.58113883008419, dtype=torch.float64), atol=1e-12)
