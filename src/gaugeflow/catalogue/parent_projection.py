"""Vectorized primitive-cell parent projection for H0-E occurrence search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from gaugeflow.catalogue.subgroup_embeddings import RationalAffineTransform
from gaugeflow.geometry import closest_image_displacements_numpy

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ParentProjection:
    lattice: FloatArray
    fractional: FloatArray
    species: IntArray
    permutations: IntArray
    source_max_displacement_angstrom: float
    source_rms_displacement_angstrom: float
    source_hencky_norm: float
    projected_group_max_error_angstrom: float


def conventional_to_primitive_structure(
    lattice: FloatArray,
    fractional: FloatArray,
    species: IntArray,
    primitive_basis: FloatArray,
    *,
    duplicate_tolerance: float = 1e-7,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Transform one standardized conventional structure without site loops.

    ``primitive_basis`` follows the spglib/spgrep convention
    ``f_conv = P f_prim``.  Centering copies are removed by a species-aware
    periodic equivalence matrix.  This is an exact coordinate change plus
    quotient, not a new symmetry search or an idealization.
    """
    cell = np.asarray(lattice, dtype=np.float64)
    positions = np.asarray(fractional, dtype=np.float64)
    numbers = np.asarray(species, dtype=np.int64)
    basis = np.asarray(primitive_basis, dtype=np.float64)
    if (
        cell.shape != (3, 3)
        or positions.ndim != 2
        or positions.shape[1] != 3
        or numbers.shape != (positions.shape[0],)
        or basis.shape != (3, 3)
        or duplicate_tolerance <= 0.0
    ):
        raise ValueError("conventional-to-primitive inputs have inconsistent shapes")
    determinant = abs(float(np.linalg.det(basis)))
    expected = int(round(positions.shape[0] * determinant))
    if expected < 1 or not np.isclose(positions.shape[0] * determinant, expected, atol=1e-8, rtol=0.0):
        raise ValueError("primitive basis is incompatible with the site count")
    primitive_cell = basis.T @ cell
    primitive_positions = (positions @ np.linalg.inv(basis).T) % 1.0
    delta = primitive_positions[:, None, :] - primitive_positions[None, :, :]
    delta -= np.rint(delta)
    equivalent = (numbers[:, None] == numbers[None, :]) & (np.max(np.abs(delta), axis=2) <= duplicate_tolerance)
    earlier = np.tril(equivalent, k=-1).any(axis=1)
    keep = ~earlier
    if int(keep.sum()) != expected:
        raise ValueError("centering quotient did not produce the expected site count")
    primitive_positions = primitive_positions[keep]
    primitive_species = numbers[keep]
    if np.linalg.det(primitive_cell) < 0.0:
        primitive_cell = primitive_cell.copy()
        primitive_positions = primitive_positions.copy()
        primitive_cell[0] *= -1.0
        primitive_positions[:, 0] *= -1.0
        primitive_positions %= 1.0
    return primitive_cell, primitive_positions, primitive_species


def conjugate_embedding_to_primitive(
    conventional: RationalAffineTransform,
    parent_primitive_basis: FloatArray,
    child_primitive_basis: FloatArray,
) -> RationalAffineTransform:
    """Return ``P_G^-1 T_conv P_H`` including the transformed origin."""
    parent_basis = np.asarray(parent_primitive_basis, dtype=np.float64)
    child_basis = np.asarray(child_primitive_basis, dtype=np.float64)
    if parent_basis.shape != (3, 3) or child_basis.shape != (3, 3):
        raise ValueError("primitive basis transforms must have shape [3,3]")
    source = conventional.as_float()
    parent_inverse = np.linalg.inv(parent_basis)
    primitive = np.eye(4, dtype=np.float64)
    primitive[:3, :3] = parent_inverse @ source[:3, :3] @ child_basis
    primitive[:3, 3] = parent_inverse @ source[:3, 3]
    return RationalAffineTransform.from_array(primitive)


def _proper_procrustes_lattice(metric: FloatArray, reference: FloatArray) -> FloatArray:
    base = np.linalg.cholesky(metric)
    left, _, right = np.linalg.svd(base.T @ reference)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    lattice = base @ rotation
    if np.linalg.det(lattice) <= 0.0:
        raise RuntimeError("projected parent lattice lost right-handedness")
    return lattice


def project_lattice_metric(lattice: FloatArray, rotations: IntArray) -> FloatArray:
    """Reynolds-project a row-lattice metric and retain its Cartesian gauge."""
    cell = np.asarray(lattice, dtype=np.float64)
    group = np.asarray(rotations, dtype=np.int64)
    if cell.shape != (3, 3) or group.ndim != 3 or group.shape[1:] != (3, 3):
        raise ValueError("lattice and rotations must have shapes [3,3] and [G,3,3]")
    metric = cell @ cell.T
    orbit = np.einsum(
        "gji,jk,gkl->gil",
        group.astype(np.float64),
        metric,
        group.astype(np.float64),
        optimize=True,
    )
    orbit_mean = orbit.mean(axis=0)
    projected = 0.5 * (orbit_mean + orbit_mean.T)
    eigenvalues = np.linalg.eigvalsh(projected)
    if np.min(eigenvalues) <= 0.0 or not np.isfinite(eigenvalues).all():
        raise RuntimeError("parent metric Reynolds projection is not positive definite")
    return _proper_procrustes_lattice(projected, cell)


def _quantized_translations(values: FloatArray) -> IntArray:
    """Vectorize one stable modulo-Z3 key for a table of translations."""
    scale = 10**10
    reduced = np.asarray(values, dtype=np.float64) - np.floor(values)
    quantized = np.rint(reduced * scale).astype(np.int64)
    return quantized % scale


def _operation_table(rotations: IntArray, translations: FloatArray) -> IntArray:
    """Build the full Seitz group table in quadratic rather than cubic work."""
    order = rotations.shape[0]
    product_rotations = np.einsum("gij,hjk->ghik", rotations, rotations, optimize=True)
    product_translations = (
        np.einsum("gij,hj->ghi", rotations.astype(np.float64), translations, optimize=True) + translations[:, None, :]
    )
    operation_keys = np.concatenate([rotations.reshape(order, 9), _quantized_translations(translations)], axis=1)
    product_keys = np.concatenate(
        [
            product_rotations.reshape(order * order, 9),
            _quantized_translations(product_translations).reshape(order * order, 3),
        ],
        axis=1,
    )
    lookup = {row.tobytes(): index for index, row in enumerate(operation_keys)}
    try:
        table = np.fromiter(
            (lookup[row.tobytes()] for row in product_keys),
            dtype=np.int64,
            count=order * order,
        ).reshape(order, order)
    except KeyError as error:
        raise ValueError("primitive Seitz operations do not close") from error
    selected_rotations = rotations[table]
    selected_translations = translations[table]
    difference = product_translations - selected_translations
    difference -= np.rint(difference)
    if not np.array_equal(product_rotations, selected_rotations) or np.any(np.max(np.abs(difference), axis=2) > 1e-9):
        raise ValueError("primitive Seitz operation hash failed exact verification")
    return table


def _assignment_permutations(
    moved: FloatArray,
    reference: FloatArray,
    species: IntArray,
    lattice: FloatArray,
    operation_table: IntArray,
) -> tuple[IntArray, float]:
    """Solve exact species blocks after one batched periodic-distance pass."""
    group_order, node_count, _ = moved.shape
    permutations = np.empty((group_order, node_count), dtype=np.int64)
    maximum = 0.0
    for number in np.unique(species):
        nodes = np.flatnonzero(species == number)
        delta = (moved[:, nodes, None, :] - reference[None, None, nodes, :]).reshape(-1, 3)
        cartesian, _ = closest_image_displacements_numpy(delta, lattice)
        costs = np.linalg.norm(cartesian, axis=1).reshape(group_order, nodes.size, nodes.size)
        for group_index, cost in enumerate(costs):
            source, target = linear_sum_assignment(cost)
            permutations[group_index, nodes[source]] = nodes[target]
            maximum = max(maximum, float(cost[source, target].max(initial=0.0)))
    for group_index in range(group_order):
        if np.unique(permutations[group_index]).size != node_count:
            raise RuntimeError("species-preserving assignment is not a permutation")
    expected = permutations[operation_table[:, :, None], np.arange(node_count)[None, None, :]]
    composed = np.take_along_axis(permutations[:, None, :], permutations[None, :, :], axis=2)
    if not np.array_equal(composed, expected):
        raise ValueError("nearest species assignment violates the parent group law")
    return permutations, maximum


def _logarithmic_strain_norm(reference: FloatArray, projected: FloatArray) -> float:
    deformation = np.linalg.solve(reference, projected)
    metric = deformation @ deformation.T
    eigenvalues = np.linalg.eigvalsh(metric)
    if np.min(eigenvalues) <= 0.0:
        raise RuntimeError("parent metric deformation is not positive definite")
    return float(0.5 * np.linalg.norm(np.log(eigenvalues)))


def project_translationengleiche_parent(
    child_lattice: FloatArray,
    child_fractional: FloatArray,
    species: IntArray,
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    primitive_embedding: RationalAffineTransform,
    *,
    maximum_source_displacement_angstrom: float,
) -> ParentProjection | None:
    """Project one primitive child through a maximal t-subgroup embedding.

    The primitive embedding must be unimodular.  The method freezes one
    species-preserving permutation/lift action before applying metric and site
    Reynolds projections; no per-time or fallback reassignment is retained.
    """
    child_cell = np.asarray(child_lattice, dtype=np.float64)
    child_positions = np.asarray(child_fractional, dtype=np.float64) % 1.0
    numbers = np.asarray(species, dtype=np.int64)
    rotations = np.asarray(parent_rotations, dtype=np.int64)
    translations = np.asarray(parent_translations, dtype=np.float64)
    if (
        child_cell.shape != (3, 3)
        or child_positions.ndim != 2
        or child_positions.shape[1] != 3
        or numbers.shape != (child_positions.shape[0],)
        or translations.shape != (rotations.shape[0], 3)
    ):
        raise ValueError("primitive parent-projection inputs have inconsistent shapes")
    transform = primitive_embedding.as_float()
    basis = transform[:3, :3]
    if not np.isclose(abs(np.linalg.det(basis)), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("translationengleiche primitive embedding must be unimodular")
    raw_parent_lattice = np.linalg.inv(basis).T @ child_cell
    raw_parent_fractional = (child_positions @ basis.T + transform[:3, 3]) % 1.0
    projected_lattice = project_lattice_metric(raw_parent_lattice, rotations)
    operation_table = _operation_table(rotations, translations)
    rotations_float = rotations.astype(np.float64)
    moved = (
        np.einsum(
            "ni,gji->gnj",
            raw_parent_fractional,
            rotations_float,
            optimize=True,
        )
        + translations[:, None, :]
    )
    try:
        permutations, assignment_maximum = _assignment_permutations(
            moved,
            raw_parent_fractional,
            numbers,
            projected_lattice,
            operation_table,
        )
    except (ValueError, RuntimeError):
        return None
    # The raw orbit defect compares two source points related through a parent
    # operation.  Its triangle bound is twice the one-sided distance from the
    # source to the parent fixed set; it is not itself the source displacement.
    if assignment_maximum > 2.0 * maximum_source_displacement_angstrom:
        return None
    target = raw_parent_fractional[permutations]
    delta = (moved - target).reshape(-1, 3)
    displacement, _ = closest_image_displacements_numpy(delta, projected_lattice)
    estimates = target @ projected_lattice + displacement.reshape(moved.shape)
    ordered = np.empty_like(estimates)
    group_index = np.arange(rotations.shape[0])[:, None]
    ordered[group_index, permutations] = estimates
    projected_cartesian = ordered.mean(axis=0)
    projected_lattice_inverse = np.linalg.inv(projected_lattice)
    projected_fractional = (projected_cartesian @ projected_lattice_inverse) % 1.0
    source_delta = projected_fractional - raw_parent_fractional
    source_displacement, _ = closest_image_displacements_numpy(source_delta, projected_lattice)
    source_norm = np.linalg.norm(source_displacement, axis=1)
    if float(source_norm.max(initial=0.0)) > maximum_source_displacement_angstrom:
        return None
    try:
        projected_moved = (
            np.einsum(
                "ni,gji->gnj",
                projected_fractional,
                rotations_float,
                optimize=True,
            )
            + translations[:, None, :]
        )
        _, projected_error = _assignment_permutations(
            projected_moved,
            projected_fractional,
            numbers,
            projected_lattice,
            operation_table,
        )
    except (ValueError, RuntimeError):
        return None
    return ParentProjection(
        lattice=projected_lattice,
        fractional=projected_fractional,
        species=numbers,
        permutations=permutations,
        source_max_displacement_angstrom=float(source_norm.max(initial=0.0)),
        source_rms_displacement_angstrom=float(np.sqrt(np.mean(source_norm * source_norm))),
        source_hencky_norm=_logarithmic_strain_norm(raw_parent_lattice, projected_lattice),
        projected_group_max_error_angstrom=projected_error,
    )
