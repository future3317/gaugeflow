"""Finite affine space-group quotients and compact displacement actions.

The quotient keeps exact fractional translations.  Dense ``3N x 3N``
displacement matrices are never materialized: a group action is represented by
one node permutation and one Cartesian 3x3 rotation per group element.  This
is mathematically identical and substantially cheaper for H0-D/H0-E.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import lcm

import numpy as np
from numpy.typing import NDArray

from gaugeflow.catalogue.finite_group import FiniteGroup, RealIrrep

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def _integer_determinant(matrix: IntArray) -> int:
    return int(round(np.linalg.det(matrix)))


def _validate_upper_hnf(matrix: IntArray, maximum_index: int = 4) -> int:
    if matrix.shape != (3, 3) or not np.array_equal(matrix, np.triu(matrix)):
        raise ValueError("supercell matrix must be a 3x3 upper HNF")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise ValueError("supercell HNF diagonal must be positive")
    for column in range(1, 3):
        if any(not 0 <= int(matrix[row, column]) < int(matrix[column, column]) for row in range(column)):
            raise ValueError("supercell HNF off-diagonal entry is outside its canonical range")
    determinant = _integer_determinant(matrix)
    if not 1 <= determinant <= maximum_index:
        raise ValueError(f"supercell index must lie in 1..{maximum_index}")
    return determinant


def _integer_lattice_quotient(
    basis: IntArray,
    *,
    maximum_index: int,
) -> tuple[tuple[int, int, int], IntArray, IntArray]:
    """Return Smith invariants, the left map and coset representatives.

    The quotient is ``Z^3 / basis Z^3`` with column-vector integer
    translations.  This lower-level form accepts any nonsingular integral
    basis; canonical HNF validation remains the responsibility of catalogue
    interfaces that require one unique supercell label.
    """
    from hsnf import smith_normal_form

    matrix = np.asarray(basis, dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("integer lattice basis must have shape [3,3]")
    determinant = abs(_integer_determinant(matrix))
    if not 1 <= determinant <= maximum_index:
        raise ValueError(f"integer lattice index must lie in 1..{maximum_index}")
    smith, left, _ = smith_normal_form(matrix)
    diagonal = tuple(abs(int(value)) for value in np.diag(smith))
    if np.prod(diagonal) != determinant or any(value < 1 for value in diagonal):
        raise RuntimeError("Smith factors do not match the integer lattice index")
    left = np.asarray(left, dtype=np.int64)
    left_inverse = np.rint(np.linalg.inv(left)).astype(np.int64)
    if not np.array_equal(left @ left_inverse, np.eye(3, dtype=np.int64)):
        raise RuntimeError("Smith left transform is not unimodular")
    residues = tuple(
        tuple(int(x) for x in value) for value in product(*(range(d) for d in diagonal))
    )
    representatives = np.stack(
        [left_inverse @ np.asarray(value, dtype=np.int64) for value in residues]
    )
    return diagonal, left, representatives


def integer_lattice_coset_representatives(
    basis: NDArray[np.integer],
    *,
    maximum_index: int = 4,
) -> IntArray:
    """Enumerate ``Z^3 / basis Z^3`` once without assuming an HNF gauge."""
    _, _, representatives = _integer_lattice_quotient(
        np.asarray(basis, dtype=np.int64), maximum_index=maximum_index
    )
    return representatives


def enumerate_upper_hnfs(maximum_index: int = 4) -> tuple[IntArray, ...]:
    """Enumerate canonical upper HNFs in deterministic lexicographic order."""
    matrices: list[IntArray] = []
    for a in range(1, maximum_index + 1):
        for b in range(1, maximum_index + 1):
            for c in range(1, maximum_index + 1):
                if a * b * c > maximum_index:
                    continue
                for ab in range(b):
                    for ac in range(c):
                        for bc in range(c):
                            matrix = np.array([[a, ab, ac], [0, b, bc], [0, 0, c]], dtype=np.int64)
                            _validate_upper_hnf(matrix, maximum_index)
                            matrices.append(matrix)
    return tuple(sorted(matrices, key=lambda value: tuple(int(x) for x in value.ravel())))


def standard_hall_numbers() -> dict[int, int]:
    """Return the frozen lowest-Hall setting for each of the 230 space groups."""
    import spglib

    selected: dict[int, int] = {}
    for hall_number in range(1, 531):
        record = spglib.get_spacegroup_type(hall_number)
        if record is None:
            continue
        selected.setdefault(int(record.number), hall_number)
    if set(selected) != set(range(1, 231)):
        raise RuntimeError("spglib Hall database did not cover all 230 space groups")
    return selected


def primitive_space_group_from_hall(hall_number: int) -> "PrimitiveSpaceGroup":
    """Load and exactly rationalize one primitive spglib Hall setting."""
    import spglib
    from spgrep.symmetry.transform import (
        get_primitive_transformation_matrix,
        transform_symmetry_and_kpoint,
        unique_primitive_symmetry,
    )

    source = spglib.get_symmetry_from_database(hall_number)
    if source is None:
        raise ValueError(f"unknown spglib Hall number {hall_number}")
    transformation = get_primitive_transformation_matrix(hall_number)
    rotations, translations, _ = transform_symmetry_and_kpoint(
        transformation,
        np.asarray(source["rotations"], dtype=np.int64),
        np.asarray(source["translations"], dtype=np.float64),
        np.zeros(3),
    )
    unique_rotations, unique_translations, _ = unique_primitive_symmetry(
        rotations, translations
    )
    return PrimitiveSpaceGroup.from_operations(unique_rotations, unique_translations)


def canonical_supercell_orbits(
    parent: "PrimitiveSpaceGroup", maximum_index: int = 4
) -> tuple[IntArray, ...]:
    """Quotient HNFs by parent rotations and unimodular row-basis changes."""
    from hsnf import row_style_hermite_normal_form

    canonical: dict[tuple[int, ...], IntArray] = {}
    for matrix in enumerate_upper_hnfs(maximum_index):
        orbit: list[IntArray] = []
        for rotation in parent.rotations:
            transformed = matrix @ rotation.T
            hnf, _ = row_style_hermite_normal_form(transformed)
            candidate = np.asarray(hnf, dtype=np.int64)
            _validate_upper_hnf(candidate, maximum_index)
            orbit.append(candidate)
        representative = min(orbit, key=lambda value: tuple(int(x) for x in value.ravel()))
        canonical.setdefault(tuple(int(x) for x in representative.ravel()), representative)
    return tuple(canonical[key] for key in sorted(canonical))


@dataclass(frozen=True)
class PrimitiveSpaceGroup:
    """Primitive Seitz coset representatives with one exact common denominator."""

    rotations: IntArray
    translation_numerators: IntArray
    translation_denominator: int
    operation_keys: tuple[str, ...]

    @classmethod
    def from_operations(
        cls,
        rotations: NDArray[np.integer],
        translations: NDArray[np.floating],
        *,
        maximum_denominator: int = 48,
        tolerance: float = 1e-9,
    ) -> "PrimitiveSpaceGroup":
        integer_rotations = np.asarray(rotations, dtype=np.int64)
        fractional = np.asarray(translations, dtype=np.float64)
        if (
            integer_rotations.ndim != 3
            or integer_rotations.shape[1:] != (3, 3)
            or fractional.shape != (integer_rotations.shape[0], 3)
        ):
            raise ValueError("primitive operations require aligned [G,3,3] rotations and [G,3] translations")
        fractions = [
            Fraction(float(value)).limit_denominator(maximum_denominator)
            for value in fractional.ravel()
        ]
        denominator = 1
        for value in fractions:
            denominator = lcm(denominator, value.denominator)
        numerators = np.array(
            [value.numerator * (denominator // value.denominator) for value in fractions],
            dtype=np.int64,
        ).reshape(fractional.shape)
        if not np.allclose(numerators / denominator, fractional, atol=tolerance, rtol=0.0):
            raise ValueError("space-group translations are not rational within the frozen tolerance")
        numerators %= denominator
        records = []
        for rotation, translation in zip(integer_rotations, numerators):
            key = (
                "R=" + ",".join(str(int(value)) for value in rotation.ravel())
                + "|t=" + ",".join(str(int(value)) for value in translation)
                + f"/{denominator}"
            )
            records.append((key, rotation, translation))
        records.sort(key=lambda value: value[0])
        keys = tuple(value[0] for value in records)
        if len(set(keys)) != len(keys):
            raise ValueError("primitive operation catalogue contains duplicate Seitz representatives")
        return cls(
            np.stack([value[1] for value in records]),
            np.stack([value[2] for value in records]),
            denominator,
            keys,
        )

    @property
    def order(self) -> int:
        return self.rotations.shape[0]


@dataclass(frozen=True)
class TranslationQuotient:
    """Exact ``Z^3 / B^T Z^3`` coordinates obtained from Smith normal form."""

    supercell_matrix: IntArray
    smith_left: IntArray
    invariant_factors: tuple[int, int, int]
    residues: tuple[tuple[int, int, int], ...]
    representatives: IntArray

    @classmethod
    def from_supercell(cls, supercell_matrix: NDArray[np.integer]) -> "TranslationQuotient":
        matrix = np.asarray(supercell_matrix, dtype=np.int64)
        determinant = _validate_upper_hnf(matrix)
        diagonal, left, representatives = _integer_lattice_quotient(
            matrix.T, maximum_index=4
        )
        if np.prod(diagonal) != determinant:
            raise RuntimeError("Smith factors do not match the supercell index")
        residues = tuple(tuple(int(x) for x in value) for value in product(*(range(d) for d in diagonal)))
        return cls(matrix, left, diagonal, residues, representatives)

    @property
    def order(self) -> int:
        return len(self.residues)

    def encode(self, translation: NDArray[np.integer]) -> tuple[int, int, int]:
        transformed = self.smith_left @ np.asarray(translation, dtype=np.int64)
        return tuple(
            int(value % factor) for value, factor in zip(transformed, self.invariant_factors)
        )


@dataclass(frozen=True)
class AffineQuotient:
    """Finite affine quotient ``G_parent^B / T_B``."""

    group: FiniteGroup
    parent: PrimitiveSpaceGroup
    translations: TranslationQuotient
    parent_operation_index: IntArray
    integer_translation: IntArray
    translation_residue: tuple[tuple[int, int, int], ...]

    @classmethod
    def build(
        cls,
        parent: PrimitiveSpaceGroup,
        supercell_matrix: NDArray[np.integer],
        *,
        tolerance: float = 1e-9,
    ) -> "AffineQuotient":
        quotient = TranslationQuotient.from_supercell(supercell_matrix)
        matrix = quotient.supercell_matrix.astype(np.float64)
        inverse = np.linalg.inv(matrix)
        transformed = (
            matrix[None, :, :]
            @ parent.rotations.transpose(0, 2, 1).astype(np.float64)
            @ inverse[None, :, :]
        )
        compatible = np.max(np.abs(transformed - np.rint(transformed)), axis=(1, 2)) <= tolerance
        selected = np.flatnonzero(compatible)
        if selected.size == 0:
            raise RuntimeError("supercell-compatible parent group lost its identity")
        rotation_to_parent = {
            tuple(int(value) for value in parent.rotations[index].ravel()): int(index)
            for index in selected
        }
        raw: list[tuple[str, int, tuple[int, int, int], IntArray]] = []
        for operation in selected:
            for residue, representative in zip(quotient.residues, quotient.representatives):
                key = parent.operation_keys[int(operation)] + "|z=" + ",".join(str(x) for x in residue)
                raw.append((key, int(operation), residue, representative))
        raw.sort(key=lambda value: value[0])
        lookup = {(operation, residue): index for index, (_, operation, residue, _) in enumerate(raw)}
        table = np.empty((len(raw), len(raw)), dtype=np.int64)
        denominator = parent.translation_denominator
        for left_index, (_, left_op, _, left_n) in enumerate(raw):
            for right_index, (_, right_op, _, right_n) in enumerate(raw):
                rotation = parent.rotations[left_op] @ parent.rotations[right_op]
                target_op = rotation_to_parent.get(tuple(int(value) for value in rotation.ravel()))
                if target_op is None:
                    raise RuntimeError("supercell-compatible point operations are not closed")
                numerator = (
                    parent.translation_numerators[left_op]
                    + denominator * left_n
                    + parent.rotations[left_op]
                    @ (parent.translation_numerators[right_op] + denominator * right_n)
                    - parent.translation_numerators[target_op]
                )
                if np.any(numerator % denominator != 0):
                    raise RuntimeError("Seitz multiplication did not close modulo primitive translations")
                residue = quotient.encode(numerator // denominator)
                table[left_index, right_index] = lookup[(target_op, residue)]
        group = FiniteGroup.from_cayley_table(table, [value[0] for value in raw])
        return cls(
            group,
            parent,
            quotient,
            np.asarray([value[1] for value in raw], dtype=np.int64),
            np.stack([value[3] for value in raw]),
            tuple(value[2] for value in raw),
        )


@dataclass(frozen=True)
class CompactDisplacementAction:
    """Permutation-plus-3x3 representation of a supercell displacement action."""

    group: FiniteGroup
    permutations: IntArray
    cartesian_rotations: FloatArray
    character: FloatArray

    def apply(self, vectors: FloatArray) -> FloatArray:
        """Apply every group element to row-vector displacements in one batch."""
        values = np.asarray(vectors, dtype=np.float64)
        if values.shape != (self.permutations.shape[1], 3):
            raise ValueError("displacements must have shape [supercell_nodes,3]")
        rotated = np.einsum("ni,gji->gnj", values, self.cartesian_rotations, optimize=True)
        output = np.empty_like(rotated)
        group_index = np.arange(self.group.order)[:, None]
        output[group_index, self.permutations] = rotated
        return output


def build_compact_displacement_action(
    quotient: AffineQuotient,
    lattice: NDArray[np.floating],
    positions: NDArray[np.floating],
    species: NDArray[np.integer],
    *,
    tolerance: float = 1e-7,
) -> CompactDisplacementAction:
    """Build the exact compact displacement representation for one parent."""
    cell = np.asarray(lattice, dtype=np.float64)
    coordinates = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(species, dtype=np.int64)
    if cell.shape != (3, 3) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("parent lattice/positions have invalid shapes")
    if numbers.shape != (coordinates.shape[0],):
        raise ValueError("parent species must align with positions")
    cells = quotient.translations.representatives
    node_count = cells.shape[0] * coordinates.shape[0]
    expanded = (coordinates[None, :, :] + cells[:, None, :]).reshape(node_count, 3)
    expanded_species = np.tile(numbers, cells.shape[0])
    residue_to_cell = {value: index for index, value in enumerate(quotient.translations.residues)}
    atom_count = coordinates.shape[0]
    permutations = np.empty((quotient.group.order, node_count), dtype=np.int64)
    cartesian_rotations = np.empty((quotient.group.order, 3, 3), dtype=np.float64)
    column_lattice = cell.T
    column_inverse = np.linalg.inv(column_lattice)
    denominator = quotient.parent.translation_denominator
    for element in range(quotient.group.order):
        operation = int(quotient.parent_operation_index[element])
        rotation = quotient.parent.rotations[operation]
        translation = (
            quotient.parent.translation_numerators[operation] / denominator
            + quotient.integer_translation[element]
        )
        transformed = expanded @ rotation.T + translation
        wrapped = transformed - np.floor(transformed)
        delta = wrapped[:, None, :] - coordinates[None, :, :]
        delta -= np.rint(delta)
        distance = np.max(np.abs(delta), axis=2)
        distance[numbers[None, :] != expanded_species[:, None]] = np.inf
        target_atom = np.argmin(distance, axis=1)
        if np.max(distance[np.arange(node_count), target_atom]) > tolerance:
            raise ValueError("space-group operation does not map the parent atoms bijectively")
        integer_shift = np.rint(transformed - coordinates[target_atom]).astype(np.int64)
        target_cell = np.asarray(
            [residue_to_cell[quotient.translations.encode(value)] for value in integer_shift],
            dtype=np.int64,
        )
        permutations[element] = target_cell * atom_count + target_atom
        if np.unique(permutations[element]).size != node_count:
            raise RuntimeError("affine operation did not induce a node permutation")
        cartesian_rotations[element] = column_lattice @ rotation @ column_inverse
    generators = np.asarray(quotient.group.generators(), dtype=np.int64)
    identity = quotient.group.identity
    if not np.array_equal(permutations[identity], np.arange(node_count)):
        raise RuntimeError("compact displacement identity does not preserve node order")
    if not np.allclose(
        cartesian_rotations[identity], np.eye(3), atol=1e-8, rtol=1e-8
    ):
        raise RuntimeError("compact displacement identity rotation is not I")
    expected_permutations = permutations[
        quotient.group.table[generators, :, None],
        np.arange(node_count)[None, None, :],
    ]
    composed_permutations = np.take_along_axis(
        permutations[generators, None, :], permutations[None, :, :], axis=2
    )
    if not np.array_equal(composed_permutations, expected_permutations):
        raise RuntimeError("compact node permutations violate the quotient group law")
    rotation_products = np.einsum(
        "aij,hjk->ahik",
        cartesian_rotations[generators],
        cartesian_rotations,
        optimize=True,
    )
    if not np.allclose(
        rotation_products,
        cartesian_rotations[quotient.group.table[generators]],
        atol=1e-8,
        rtol=1e-8,
    ):
        raise RuntimeError("Cartesian rotations violate the quotient group law")
    fixed_nodes = permutations == np.arange(node_count)[None, :]
    character = fixed_nodes.sum(axis=1) * np.trace(cartesian_rotations, axis1=1, axis2=2)
    return CompactDisplacementAction(
        quotient.group, permutations, cartesian_rotations, character
    )


def real_irrep_multiplicity(action: CompactDisplacementAction, irrep: RealIrrep) -> int:
    """Return occurrence multiplicity using the real-character inner product."""
    if irrep.matrices.shape[0] != action.group.order:
        raise ValueError("irrep and displacement action use different finite groups")
    character = np.trace(irrep.matrices, axis1=1, axis2=2)
    norm = float(np.mean(character * character))
    inner = float(np.mean(action.character * character))
    multiplicity = int(round(inner / norm))
    if multiplicity < 0 or not np.isclose(inner, multiplicity * norm, atol=1e-7, rtol=1e-7):
        raise RuntimeError("displacement character did not decompose into real irreps")
    return multiplicity
