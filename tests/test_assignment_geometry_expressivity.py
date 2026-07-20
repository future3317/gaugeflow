from __future__ import annotations

import numpy as np
import torch

from scripts.audit_h1a_assignment_geometry_expressivity import (
    audit_geometry_carrier,
    geometry_site_signatures,
    representation_invariance_checks,
)


def _square_carrier() -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    action = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    fractional = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    lattice = np.diag([4.0, 4.0, 5.0])
    return action, fractional, lattice


def test_geometry_pair_signature_resolves_adjacent_square_coloring() -> None:
    action, fractional, lattice = _square_carrier()
    signatures = geometry_site_signatures(action, fractional, lattice)
    assert len(set(signatures)) == 1

    result = audit_geometry_carrier(
        [0, 0, 1, 1],
        action,
        fractional,
        lattice,
        maximum_sites=20,
        maximum_collision_class=100,
        chunk_size=16,
        distance_resolution_angstrom=1e-6,
    )
    assert result["geometry_unary_collision_class_size"] == 6
    assert result["target_orbit_size"] == 4
    assert result["geometry_pair_collision_class_size"] == 4
    assert result["geometry_pair_resolved"]


def test_geometry_descriptors_are_relabel_and_cell_basis_invariant() -> None:
    action, fractional, lattice = _square_carrier()
    relabel, cell_basis = representation_invariance_checks(
        action,
        fractional,
        lattice,
        seed=5705,
        maximum_sites=20,
        distance_resolution_angstrom=1e-6,
    )
    assert relabel
    assert cell_basis
