"""Strict unit-cell quotienting for GaugeFlow targets."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure


def transform_row_lattice_basis(
    lattice: np.ndarray, fractional: np.ndarray, basis_change: np.ndarray, *, atol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a row-lattice ``GL(3,Z)`` change of unit-cell basis exactly.

    GaugeFlow stores Cartesian positions as ``r = f @ L`` where lattice rows
    are real-space basis vectors.  Therefore an arbitrary unimodular basis
    change (including orientation-reversing ``det(B) = -1`` gauges) acts as
    ``L' = B @ L`` and ``f' = f @ B^{-1}``.  This is a crystallographic cell
    gauge, not a Cartesian proper rotation; callers must not use it to
    quotient a polar rank-three response tensor.
    """
    lattice_array = np.asarray(lattice, dtype=float)
    fractional_array = np.asarray(fractional, dtype=float)
    basis = np.asarray(basis_change, dtype=float)
    if lattice_array.shape != (3, 3) or basis.shape != (3, 3) or fractional_array.shape[-1] != 3:
        raise ValueError("lattice and basis_change must be [3,3]; fractional must end in 3")
    integer_basis = np.rint(basis)
    if not np.allclose(basis, integer_basis, atol=atol, rtol=0.0):
        raise ValueError("basis_change must have integer entries")
    determinant = round(float(np.linalg.det(integer_basis)))
    if abs(determinant) != 1:
        raise ValueError("basis_change must lie in GL(3,Z), with determinant +1 or -1")
    transformed_lattice = integer_basis @ lattice_array
    transformed_fractional = fractional_array @ np.linalg.inv(integer_basis)
    return transformed_lattice, transformed_fractional


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
    _, fractional = transform_row_lattice_basis(original, structure.frac_coords, integer_change, atol=atol)
    fractional = np.remainder(fractional, 1.0)
    reduced_structure = Structure(
        reduced_lattice,
        structure.species,
        fractional,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )
    return reduced_structure, integer_change.astype(np.int64)

