"""Read-only automorphism audits for unlabeled periodic site sets.

The atom-type decoder proposed after Gate A10 must not be evaluated against a
fixed CIF row labeling when that labeling is not identifiable from the
unlabeled periodic geometry.  This module explicitly constructs the site
permutations induced by space-group operations on an *all-identical-species*
copy of a structure.  Species are inspected only after the geometric orbits
have been computed.

The lightweight A11-G message proposal uses distances and dot products.  Those
features are O(3)-invariant, not merely SO(3)-invariant, so the audit reports
both proper and full point-set automorphisms.  The full O(3) partition is the
conservative identifiability test for that representation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .unit_cell import niggli_reduce_structure_with_transform


def _as_array(value: np.ndarray, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"Expected {name} with shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _cartesian_row_rotation(fractional_rotation: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert a column-fractional symmetry operation to a row-Cartesian map."""
    # Pymatgen applies f' = R f + t to column fractional coordinates.  With
    # GaugeFlow's row lattice convention x = f^T L, the corresponding row map
    # is x' = x S with L S = R^T L.
    return np.linalg.solve(lattice, fractional_rotation.T @ lattice)


def _site_permutation(
    fractional_rotation: np.ndarray,
    translation: np.ndarray,
    frac_coords: np.ndarray,
    lattice: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Map every transformed site to one original site without using species."""
    transformed = np.remainder(frac_coords @ fractional_rotation.T + translation, 1.0)
    delta = transformed[:, None, :] - frac_coords[None, :, :]
    delta -= np.rint(delta)
    distance = np.linalg.norm(delta @ lattice, axis=-1)
    permutation = distance.argmin(axis=1)
    nearest = distance[np.arange(frac_coords.shape[0]), permutation]
    if np.any(nearest > tolerance) or np.unique(permutation).size != frac_coords.shape[0]:
        raise ValueError(
            "A reported periodic symmetry operation did not induce a bijective "
            "site permutation within the configured tolerance"
        )
    return permutation.astype(np.int64)


def _orbits(site_count: int, permutations: list[np.ndarray]) -> list[list[int]]:
    parent = list(range(site_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for permutation in permutations:
        for source, target in enumerate(permutation.tolist()):
            union(source, target)
    grouped: dict[int, list[int]] = {}
    for index in range(site_count):
        grouped.setdefault(find(index), []).append(index)
    return sorted((sorted(values) for values in grouped.values()), key=lambda values: values[0])


@dataclass(frozen=True)
class PeriodicAutomorphisms:
    """Canonical-cell automorphisms and their induced site-orbit partition."""

    lattice: np.ndarray
    frac_coords: np.ndarray
    niggli_transform: np.ndarray
    operations: tuple[dict[str, object], ...]
    site_orbits: tuple[tuple[int, ...], ...]


def unlabeled_periodic_automorphisms(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    mapping_tolerance: float = 1e-3,
    proper_only: bool,
) -> PeriodicAutomorphisms:
    """Return canonical-cell point-set automorphisms without using elements.

    Niggli reduction quotients integer lattice-basis representations before
    symmetry analysis.  Each accepted operation is converted to an explicit
    permutation of the *input site indices*; duplicated geometric operations
    that induce the same permutation are retained once.
    """
    lattice = _as_array(lattice, shape=(3, 3), name="lattice")
    frac = np.asarray(frac_coords, dtype=float)
    if frac.ndim != 2 or frac.shape[0] < 1 or frac.shape[1] != 3:
        raise ValueError("Expected at least one [fractional-site, 3] coordinate")
    if not np.isfinite(frac).all():
        raise ValueError("frac_coords must be finite")
    if symprec <= 0 or angle_tolerance <= 0 or mapping_tolerance <= 0:
        raise ValueError("Symmetry and mapping tolerances must be positive")

    # ``H`` is deliberately identical at every site.  It is not a chemical
    # assumption: it removes all target species from the symmetry calculation.
    unlabeled = Structure(Lattice(lattice), ["H"] * frac.shape[0], np.remainder(frac, 1.0))
    canonical, basis_transform = niggli_reduce_structure_with_transform(unlabeled)
    canonical_lattice = np.asarray(canonical.lattice.matrix, dtype=float)
    canonical_frac = np.asarray(canonical.frac_coords, dtype=float)
    analyzer = SpacegroupAnalyzer(canonical, symprec=symprec, angle_tolerance=angle_tolerance)

    operations: list[dict[str, object]] = []
    seen_permutations: set[tuple[int, ...]] = set()
    for operation in analyzer.get_symmetry_operations(cartesian=False):
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        translation = np.asarray(operation.translation_vector, dtype=float)
        cartesian_row = _cartesian_row_rotation(rotation, canonical_lattice)
        determinant = float(np.linalg.det(cartesian_row))
        if not np.isclose(abs(determinant), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError("Space-group operation was not an orthogonal Cartesian map")
        if proper_only and determinant < 0:
            continue
        permutation = _site_permutation(
            rotation,
            translation,
            canonical_frac,
            canonical_lattice,
            tolerance=mapping_tolerance,
        )
        permutation_key = tuple(int(value) for value in permutation)
        if permutation_key in seen_permutations:
            continue
        seen_permutations.add(permutation_key)
        operations.append(
            {
                "fractional_rotation": rotation,
                "fractional_translation": np.remainder(translation, 1.0),
                "cartesian_row_rotation": cartesian_row,
                "cartesian_determinant": determinant,
                "permutation": permutation,
            }
        )
    if not operations:
        raise ValueError("No admissible periodic automorphism was found")
    identity = tuple(range(frac.shape[0]))
    if identity not in seen_permutations:
        raise ValueError("The periodic automorphism set did not contain identity")
    permutations = [np.asarray(item["permutation"], dtype=np.int64) for item in operations]
    return PeriodicAutomorphisms(
        lattice=canonical_lattice,
        frac_coords=canonical_frac,
        niggli_transform=np.asarray(basis_transform, dtype=np.int64),
        operations=tuple(operations),
        site_orbits=tuple(tuple(orbit) for orbit in _orbits(frac.shape[0], permutations)),
    )


def _orbit_label_summary(orbits: tuple[tuple[int, ...], ...], atomic_numbers: np.ndarray) -> dict[str, object]:
    records = []
    correct_if_constant = 0
    for orbit_index, orbit in enumerate(orbits):
        labels = [int(atomic_numbers[index]) for index in orbit]
        counts = Counter(labels)
        majority = max(counts.values())
        correct_if_constant += majority
        records.append(
            {
                "orbit_index": orbit_index,
                "site_indices": list(orbit),
                "atomic_number_counts": {str(key): int(value) for key, value in sorted(counts.items())},
                "is_species_mixed": len(counts) > 1,
                "deterministic_constant_label_ceiling": majority / len(orbit),
            }
        )
    return {
        "orbits": records,
        "mixed_orbit_count": sum(bool(record["is_species_mixed"]) for record in records),
        "deterministic_equivariant_fixed_cif_accuracy_ceiling": correct_if_constant / atomic_numbers.size,
    }


def audit_unlabeled_periodic_site_orbits(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    mapping_tolerance: float = 1e-3,
) -> dict[str, object]:
    """Audit proper and full unlabeled site orbits for one decorated crystal.

    The conservative ``o3_scalar`` result is the relevant one for A11-G's
    distance/dot-product message proposal.  A mixed full orbit means a
    deterministic O(3)-invariant site classifier cannot reproduce a unique
    fixed-CIF labeling on every member of that orbit.
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    frac = np.asarray(frac_coords, dtype=float)
    if atomic_numbers.ndim != 1 or atomic_numbers.size != frac.shape[0]:
        raise ValueError("atomic_numbers must contain one value per fractional site")
    proper = unlabeled_periodic_automorphisms(
        lattice,
        frac,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        mapping_tolerance=mapping_tolerance,
        proper_only=True,
    )
    full = unlabeled_periodic_automorphisms(
        lattice,
        frac,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        mapping_tolerance=mapping_tolerance,
        proper_only=False,
    )
    if not np.array_equal(proper.niggli_transform, full.niggli_transform):
        raise RuntimeError("Proper and full automorphism audits used different canonical cells")
    proper_labels = _orbit_label_summary(proper.site_orbits, atomic_numbers)
    full_labels = _orbit_label_summary(full.site_orbits, atomic_numbers)
    full_mixed = int(full_labels["mixed_orbit_count"])
    return {
        "site_count": int(atomic_numbers.size),
        "niggli_transform": proper.niggli_transform.tolist(),
        "canonical_lattice": proper.lattice.tolist(),
        "canonical_frac_coords": proper.frac_coords.tolist(),
        "proper_so3": {
            "operation_count": len(proper.operations),
            "operations": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in operation.items()
                }
                for operation in proper.operations
            ],
            **proper_labels,
        },
        "full_o3_scalar": {
            "operation_count": len(full.operations),
            "operations": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in operation.items()
                }
                for operation in full.operations
            ],
            **full_labels,
        },
        "a11_g_decision": (
            "geometry_only_authorized"
            if full_mixed == 0
            else "stochastic_assignment_and_quotient_supervision_required"
        ),
    }
