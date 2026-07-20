"""Vectorized primitive-cell parent projection for H0-E occurrence search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from gaugeflow.catalogue.affine_quotient import integer_lattice_coset_representatives
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


@dataclass(frozen=True)
class GeometryParentProjection:
    """Species-free parent geometry and its exact node permutation action."""

    lattice: FloatArray
    fractional: FloatArray
    permutations: IntArray
    source_max_displacement_angstrom: float
    source_rms_displacement_angstrom: float
    source_hencky_norm: float
    projected_group_max_error_angstrom: float


@dataclass(frozen=True)
class CompleteGeometryParentProjection:
    """Primitive carrier plus its aligned supercell realization.

    ``expanded_fractional`` retains the node order on which ``parent``'s
    permutation action was certified.  The integral embedding is kept so an
    offline compiler can change to a canonical HNF chart without consulting
    terminal species or a child-space-group label.
    """

    parent: GeometryParentProjection
    expanded_lattice: FloatArray
    expanded_fractional: FloatArray
    embedding_basis: IntArray
    embedding_origin: FloatArray
    full_action_order: int


def _strip_internal_carrier_labels(projection: ParentProjection) -> GeometryParentProjection:
    return GeometryParentProjection(
        lattice=projection.lattice,
        fractional=projection.fractional,
        permutations=projection.permutations,
        source_max_displacement_angstrom=projection.source_max_displacement_angstrom,
        source_rms_displacement_angstrom=projection.source_rms_displacement_angstrom,
        source_hencky_norm=projection.source_hencky_norm,
        projected_group_max_error_angstrom=projection.projected_group_max_error_angstrom,
    )


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


def _project_parent_action(
    source_lattice: FloatArray,
    source_fractional: FloatArray,
    species: IntArray,
    rotations: IntArray,
    translations: FloatArray,
    *,
    maximum_source_displacement_angstrom: float,
    projected_lattice: FloatArray | None = None,
) -> ParentProjection | None:
    """Project one fixed affine action without changing its discrete branch."""
    source_cell = np.asarray(source_lattice, dtype=np.float64)
    source_positions = np.asarray(source_fractional, dtype=np.float64) % 1.0
    numbers = np.asarray(species, dtype=np.int64)
    group_rotations = np.asarray(rotations, dtype=np.int64)
    group_translations = np.asarray(translations, dtype=np.float64)
    if (
        source_cell.shape != (3, 3)
        or source_positions.ndim != 2
        or source_positions.shape[1] != 3
        or numbers.shape != (source_positions.shape[0],)
        or group_rotations.ndim != 3
        or group_rotations.shape[1:] != (3, 3)
        or group_translations.shape != (group_rotations.shape[0], 3)
        or maximum_source_displacement_angstrom <= 0.0
    ):
        raise ValueError("parent-action projection inputs have inconsistent shapes")
    if projected_lattice is None:
        target_cell = project_lattice_metric(source_cell, group_rotations)
    else:
        target_cell = np.asarray(projected_lattice, dtype=np.float64)
        if target_cell.shape != (3, 3) or np.linalg.det(target_cell) <= 0.0:
            raise ValueError("provided projected lattice must be right-handed [3,3]")
    operation_table = _operation_table(group_rotations, group_translations)
    rotations_float = group_rotations.astype(np.float64)
    moved = (
        np.einsum(
            "ni,gji->gnj",
            source_positions,
            rotations_float,
            optimize=True,
        )
        + group_translations[:, None, :]
    )
    try:
        permutations, assignment_maximum = _assignment_permutations(
            moved,
            source_positions,
            numbers,
            target_cell,
            operation_table,
        )
    except (ValueError, RuntimeError):
        return None
    if assignment_maximum > 2.0 * maximum_source_displacement_angstrom:
        return None
    target = source_positions[permutations]
    delta = (moved - target).reshape(-1, 3)
    displacement, _ = closest_image_displacements_numpy(delta, target_cell)
    estimates = target @ target_cell + displacement.reshape(moved.shape)
    ordered = np.empty_like(estimates)
    group_index = np.arange(group_rotations.shape[0])[:, None]
    ordered[group_index, permutations] = estimates
    projected_cartesian = ordered.mean(axis=0)
    target_inverse = np.linalg.inv(target_cell)
    projected_fractional = (projected_cartesian @ target_inverse) % 1.0
    source_delta = projected_fractional - source_positions
    source_displacement, _ = closest_image_displacements_numpy(source_delta, target_cell)
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
            + group_translations[:, None, :]
        )
        _, projected_error = _assignment_permutations(
            projected_moved,
            projected_fractional,
            numbers,
            target_cell,
            operation_table,
        )
    except (ValueError, RuntimeError):
        return None
    return ParentProjection(
        lattice=target_cell,
        fractional=projected_fractional,
        species=numbers,
        permutations=permutations,
        source_max_displacement_angstrom=float(source_norm.max(initial=0.0)),
        source_rms_displacement_angstrom=float(np.sqrt(np.mean(source_norm * source_norm))),
        source_hencky_norm=_logarithmic_strain_norm(source_cell, target_cell),
        projected_group_max_error_angstrom=projected_error,
    )


def klassengleiche_supercell_operations(
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    embedding_basis: IntArray,
    embedding_origin: FloatArray,
    *,
    maximum_index: int = 4,
) -> tuple[IntArray, FloatArray]:
    """Lift a primitive parent action into one child-supercell coordinate chart.

    If ``f_parent = B f_child + o``, every parent operation and translation
    coset ``z in Z^3 / B Z^3`` acts through

    ``B^-1 R B`` and ``B^-1 (R o + t + z - o)``.
    """
    rotations = np.asarray(parent_rotations, dtype=np.int64)
    translations = np.asarray(parent_translations, dtype=np.float64)
    basis = np.asarray(embedding_basis, dtype=np.int64)
    origin = np.asarray(embedding_origin, dtype=np.float64)
    if (
        rotations.ndim != 3
        or rotations.shape[1:] != (3, 3)
        or translations.shape != (rotations.shape[0], 3)
        or basis.shape != (3, 3)
        or origin.shape != (3,)
    ):
        raise ValueError("klassengleiche operation inputs have inconsistent shapes")
    index = abs(int(round(np.linalg.det(basis))))
    if not 2 <= index <= maximum_index:
        raise ValueError(f"klassengleiche embedding index must lie in 2..{maximum_index}")
    inverse = np.linalg.inv(basis.astype(np.float64))
    cosets = integer_lattice_coset_representatives(basis, maximum_index=maximum_index)
    changed = np.einsum(
        "ij,gjk,kl->gil",
        inverse,
        rotations.astype(np.float64),
        basis.astype(np.float64),
        optimize=True,
    )
    rounded = np.rint(changed).astype(np.int64)
    if not np.allclose(changed, rounded, atol=1e-9, rtol=0.0):
        raise ValueError("parent rotations do not preserve the embedded child lattice")
    offsets = (
        np.einsum("gij,j->gi", rotations.astype(np.float64), origin, optimize=True) + translations - origin[None, :]
    )
    lifted_translations = np.einsum(
        "ij,grj->gri",
        inverse,
        offsets[:, None, :] + cosets[None, :, :],
        optimize=True,
    )
    lifted_rotations = np.repeat(rounded, index, axis=0)
    lifted_translations = lifted_translations.reshape(-1, 3) % 1.0
    keys = np.concatenate(
        [
            lifted_rotations.reshape(-1, 9),
            _quantized_translations(lifted_translations),
        ],
        axis=1,
    )
    if np.unique(keys, axis=0).shape[0] != rotations.shape[0] * index:
        raise RuntimeError("klassengleiche affine quotient contains duplicate operations")
    _operation_table(lifted_rotations, lifted_translations)
    return lifted_rotations, lifted_translations


def _quotient_parent_sites(
    supercell_fractional: FloatArray,
    species: IntArray,
    parent_lattice: FloatArray,
    embedding_basis: IntArray,
    embedding_origin: FloatArray,
    *,
    index: int,
    tolerance_angstrom: float = 1e-7,
) -> tuple[FloatArray, IntArray]:
    mapped = (
        np.asarray(supercell_fractional, dtype=np.float64) @ np.asarray(embedding_basis, dtype=np.float64).T
        + np.asarray(embedding_origin, dtype=np.float64)
    ) % 1.0
    numbers = np.asarray(species, dtype=np.int64)
    delta = (mapped[:, None, :] - mapped[None, :, :]).reshape(-1, 3)
    cartesian, _ = closest_image_displacements_numpy(delta, parent_lattice)
    distances = np.linalg.norm(cartesian, axis=1).reshape(mapped.shape[0], mapped.shape[0])
    equivalent = (numbers[:, None] == numbers[None, :]) & (distances <= tolerance_angstrom)
    if not np.all(equivalent.sum(axis=1) == index):
        raise ValueError("projected supercell sites do not form exact parent translation orbits")
    keep = ~np.tril(equivalent, k=-1).any(axis=1)
    expected = mapped.shape[0] // index
    if mapped.shape[0] % index or int(keep.sum()) != expected:
        raise ValueError("parent-cell quotient has the wrong site count")
    return mapped[keep], numbers[keep]


def _project_klassengleiche_parent_complete(
    child_lattice: FloatArray,
    child_fractional: FloatArray,
    species: IntArray,
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    primitive_embedding: RationalAffineTransform,
    *,
    maximum_source_displacement_angstrom: float,
    maximum_index: int = 4,
) -> tuple[ParentProjection, ParentProjection, IntArray, FloatArray] | None:
    """Return primitive and aligned-supercell projections in one pass."""
    child_cell = np.asarray(child_lattice, dtype=np.float64)
    child_positions = np.asarray(child_fractional, dtype=np.float64) % 1.0
    numbers = np.asarray(species, dtype=np.int64)
    rotations = np.asarray(parent_rotations, dtype=np.int64)
    translations = np.asarray(parent_translations, dtype=np.float64)
    transform = primitive_embedding.as_float()
    basis_float = transform[:3, :3]
    basis = np.rint(basis_float).astype(np.int64)
    if not np.allclose(basis_float, basis, atol=1e-9, rtol=0.0):
        raise ValueError("klassengleiche primitive embedding basis must be integral")
    index = abs(int(round(np.linalg.det(basis))))
    if not 2 <= index <= maximum_index:
        raise ValueError(f"klassengleiche primitive embedding index must lie in 2..{maximum_index}")
    if numbers.size % index:
        return None
    raw_parent_lattice = np.linalg.inv(basis.astype(np.float64)).T @ child_cell
    projected_parent_lattice = project_lattice_metric(raw_parent_lattice, rotations)
    projected_child_lattice = basis.T @ projected_parent_lattice
    lifted_rotations, lifted_translations = klassengleiche_supercell_operations(
        rotations,
        translations,
        basis,
        transform[:3, 3],
        maximum_index=maximum_index,
    )
    projected = _project_parent_action(
        child_cell,
        child_positions,
        numbers,
        lifted_rotations,
        lifted_translations,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
        projected_lattice=projected_child_lattice,
    )
    if projected is None:
        return None
    try:
        parent_fractional, parent_species = _quotient_parent_sites(
            projected.fractional,
            numbers,
            projected_parent_lattice,
            basis,
            transform[:3, 3],
            index=index,
        )
    except ValueError:
        return None
    parent = ParentProjection(
        lattice=projected_parent_lattice,
        fractional=parent_fractional,
        species=parent_species,
        permutations=projected.permutations,
        source_max_displacement_angstrom=projected.source_max_displacement_angstrom,
        source_rms_displacement_angstrom=projected.source_rms_displacement_angstrom,
        source_hencky_norm=projected.source_hencky_norm,
        projected_group_max_error_angstrom=projected.projected_group_max_error_angstrom,
    )
    return parent, projected, basis, transform[:3, 3].copy()


def project_klassengleiche_parent(
    child_lattice: FloatArray,
    child_fractional: FloatArray,
    species: IntArray,
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    primitive_embedding: RationalAffineTransform,
    *,
    maximum_source_displacement_angstrom: float,
    maximum_index: int = 4,
) -> tuple[ParentProjection, int] | None:
    """Project an index-2..4 child supercell onto its primitive parent."""
    complete = _project_klassengleiche_parent_complete(
        child_lattice,
        child_fractional,
        species,
        parent_rotations,
        parent_translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
        maximum_index=maximum_index,
    )
    if complete is None:
        return None
    parent, expanded, _, _ = complete
    return parent, int(expanded.permutations.shape[0])


def project_geometry_klassengleiche_parent(
    child_lattice: FloatArray,
    child_fractional: FloatArray,
    node_count: int,
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    primitive_embedding: RationalAffineTransform,
    *,
    maximum_source_displacement_angstrom: float,
    maximum_index: int = 4,
) -> tuple[GeometryParentProjection, int] | None:
    """Project a k-parent geometry without assigning a dummy physical species."""
    if node_count < 1 or np.asarray(child_fractional).shape != (node_count, 3):
        raise ValueError("geometry carrier node count does not match child coordinates")
    complete = project_complete_geometry_klassengleiche_parent(
        child_lattice,
        child_fractional,
        node_count,
        parent_rotations,
        parent_translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
        maximum_index=maximum_index,
    )
    if complete is None:
        return None
    return complete.parent, complete.full_action_order


def project_complete_geometry_klassengleiche_parent(
    child_lattice: FloatArray,
    child_fractional: FloatArray,
    node_count: int,
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    primitive_embedding: RationalAffineTransform,
    *,
    maximum_source_displacement_angstrom: float,
    maximum_index: int = 4,
) -> CompleteGeometryParentProjection | None:
    """Retain the expanded ideal geometry discarded by the legacy serializer."""
    if node_count < 1 or np.asarray(child_fractional).shape != (node_count, 3):
        raise ValueError("geometry carrier node count does not match child coordinates")
    internal_labels = np.zeros(node_count, dtype=np.int64)
    projected = _project_klassengleiche_parent_complete(
        child_lattice,
        child_fractional,
        internal_labels,
        parent_rotations,
        parent_translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
        maximum_index=maximum_index,
    )
    if projected is None:
        return None
    parent, expanded, basis, origin = projected
    return CompleteGeometryParentProjection(
        parent=_strip_internal_carrier_labels(parent),
        expanded_lattice=expanded.lattice,
        expanded_fractional=expanded.fractional,
        embedding_basis=basis,
        embedding_origin=origin,
        full_action_order=int(expanded.permutations.shape[0]),
    )


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
    return _project_parent_action(
        raw_parent_lattice,
        raw_parent_fractional,
        numbers,
        rotations,
        translations,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
    )


def project_geometry_translationengleiche_parent(
    child_lattice: FloatArray,
    child_fractional: FloatArray,
    node_count: int,
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    primitive_embedding: RationalAffineTransform,
    *,
    maximum_source_displacement_angstrom: float,
) -> GeometryParentProjection | None:
    """Project a t-parent geometry without assigning a dummy physical species."""
    if node_count < 1 or np.asarray(child_fractional).shape != (node_count, 3):
        raise ValueError("geometry carrier node count does not match child coordinates")
    internal_labels = np.zeros(node_count, dtype=np.int64)
    projected = project_translationengleiche_parent(
        child_lattice,
        child_fractional,
        internal_labels,
        parent_rotations,
        parent_translations,
        primitive_embedding,
        maximum_source_displacement_angstrom=maximum_source_displacement_angstrom,
    )
    return None if projected is None else _strip_internal_carrier_labels(projected)
