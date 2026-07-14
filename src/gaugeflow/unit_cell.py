"""Strict unit-cell quotienting for GaugeFlow targets."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure


def niggli_reduce_structure_with_transform(
    structure: Structure, *, atol: float = 1e-5
) -> tuple[Structure, np.ndarray]:
    """Return a Niggli-reduced equivalent structure with tracked coordinates.

    If row-vector lattice bases obey ``L_reduced = B @ L_original``, fractional
    coordinates must transform as ``f_reduced = f_original @ inv(B)``.  The
    change of basis is required to be unimodular; an invalid numerical reduction
    is an error rather than an untracked representation fallback.
    """
    original = np.asarray(structure.lattice.matrix, dtype=float)
    reduced_lattice = structure.lattice.get_niggli_reduced_lattice()
    reduced = np.asarray(reduced_lattice.matrix, dtype=float)
    change = reduced @ np.linalg.inv(original)
    integer_change = np.rint(change)
    if not np.allclose(change, integer_change, atol=atol, rtol=0.0):
        raise ValueError("Niggli reduction did not yield an integer lattice-basis transform")
    determinant = round(float(np.linalg.det(integer_change)))
    if abs(determinant) != 1:
        raise ValueError("Niggli reduction did not yield a unimodular lattice-basis transform")
    fractional = np.remainder(structure.frac_coords @ np.linalg.inv(integer_change), 1.0)
    reduced_structure = Structure(
        reduced_lattice,
        structure.species,
        fractional,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )
    return reduced_structure, integer_change.astype(np.int64)


def niggli_reduce_structure(structure: Structure, *, atol: float = 1e-5) -> Structure:
    """Backward-compatible structure-only Niggli reduction."""
    return niggli_reduce_structure_with_transform(structure, atol=atol)[0]
