from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import torch


def _load_script() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "audit_h1a_midnoise_reciprocal_attribution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_h1a_midnoise_reciprocal_attribution", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projective_grid_has_exact_duplicate_correction() -> None:
    module = _load_script()
    grid = module._projective_integer_grid_cpu(2)
    values = {tuple(row) for row in grid.tolist()}
    assert (0, 0, 0) not in values
    assert len(values) == ((2 * 2 + 1) ** 3 - 1) // 2
    assert all(tuple(-value for value in row) not in values for row in values)


def test_reciprocal_carrier_is_translation_and_permutation_equivariant() -> None:
    module = _load_script()
    coordinates = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.25, 0.8], [0.75, 0.7, 0.1]],
        dtype=torch.float64,
    )
    atom_types = torch.tensor([4, 7, 4])
    lattice = torch.diag(torch.tensor([4.0, 4.5, 5.0], dtype=torch.float64)).unsqueeze(0)
    batch = torch.zeros(3, dtype=torch.long)
    basis = module._reciprocal_basis(
        coordinates, atom_types, lattice, batch, q_max=4.0
    )
    reference = module._reciprocal_carrier(
        basis, batch, band=(0.0, 1.5), radial_channels=4
    )
    shifted_basis = module._reciprocal_basis(
        coordinates + torch.tensor([0.37, -0.18, 0.49]),
        atom_types,
        lattice,
        batch,
        q_max=4.0,
    )
    shifted = module._reciprocal_carrier(
        shifted_basis, batch, band=(0.0, 1.5), radial_channels=4
    )
    torch.testing.assert_close(shifted, reference, atol=1.0e-11, rtol=1.0e-11)
    permutation = torch.tensor([2, 0, 1])
    permuted_basis = module._reciprocal_basis(
        coordinates[permutation], atom_types[permutation], lattice, batch, q_max=4.0
    )
    permuted = module._reciprocal_carrier(
        permuted_basis, batch, band=(0.0, 1.5), radial_channels=4
    )
    torch.testing.assert_close(
        permuted, reference[permutation], atol=1.0e-11, rtol=1.0e-11
    )
    torch.testing.assert_close(reference.mean(dim=0), torch.zeros_like(reference.mean(dim=0)))


def test_pair_lattice_descriptor_quotients_permutation_translation_and_o3() -> None:
    module = _load_script()
    coordinates = torch.tensor(
        [[0.05, 0.1, 0.2], [0.4, 0.2, 0.7], [0.8, 0.75, 0.15]],
        dtype=torch.float64,
    )
    atom_types = torch.tensor([5, 7, 5])
    lattice = torch.tensor(
        [[4.0, 0.0, 0.0], [0.3, 4.5, 0.0], [0.2, 0.1, 5.0]],
        dtype=torch.float64,
    )
    pair, cell = module._pair_lattice_descriptor(coordinates, lattice, atom_types)
    permutation = torch.tensor([2, 0, 1])
    angle = 0.37
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    transformed = module._pair_lattice_descriptor(
        coordinates[permutation] + torch.tensor([0.31, -0.22, 0.17]),
        lattice @ rotation,
        atom_types[permutation],
    )
    torch.testing.assert_close(transformed[0], pair, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(transformed[1], cell, atol=1.0e-11, rtol=1.0e-11)


def test_frozen_probe_recovers_synthetic_carrier_residual() -> None:
    module = _load_script()
    generator = torch.Generator().manual_seed(42)
    carrier = torch.randn((12, 5, 3), generator=generator, dtype=torch.float64)
    batch = torch.tensor([0] * 5 + [1] * 7)
    carrier = carrier - module.graph_mean(carrier, batch, 2)[batch]
    coefficient = torch.tensor([0.5, -0.2, 0.1, 0.7, -0.4], dtype=torch.float64)
    residual = torch.einsum("ncd,c->nd", carrier, coefficient)
    xtx, xty = module._probe_sufficient_statistics(carrier, residual, batch, 2)
    fitted, audit = module._fit_ridge_probe(xtx, xty, 1.0e-10)
    assert audit["rank"] == 5
    torch.testing.assert_close(fitted, coefficient.double(), atol=1.0e-8, rtol=1.0e-8)
