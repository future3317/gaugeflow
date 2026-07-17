"""Exact compact affine embeddings for maximal space-group subgroups.

The module contains no PyXtal runtime dependency.  An offline builder may read
an externally versioned maximal-subgroup table, rationalize its affine basis
changes, and certify every edge against explicit Seitz operations.  The active
representation stores one small integer ``3 x 4`` numerator and one common
denominator per embedding rather than materializing operation-by-operation
maps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import gcd, lcm
from typing import Hashable, Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RationalAffineTransform:
    """A nonsingular homogeneous affine transform with one exact denominator."""

    numerators: IntArray
    denominator: int

    def __post_init__(self) -> None:
        values = np.asarray(self.numerators, dtype=np.int64)
        denominator = int(self.denominator)
        if values.shape != (4, 4) or denominator <= 0:
            raise ValueError("rational affine transform requires [4,4] numerators and a positive denominator")
        if not np.array_equal(values[3], np.array([0, 0, 0, denominator], dtype=np.int64)):
            raise ValueError("rational affine transform must have homogeneous final row [0,0,0,1]")
        if abs(float(np.linalg.det(values[:3, :3].astype(np.float64)))) <= 1e-12:
            raise ValueError("rational affine transform must have a nonsingular basis")
        object.__setattr__(self, "numerators", values)
        object.__setattr__(self, "denominator", denominator)

    @classmethod
    def from_array(
        cls,
        values: NDArray[np.floating],
        *,
        maximum_denominator: int = 48,
        tolerance: float = 1e-10,
    ) -> "RationalAffineTransform":
        """Rationalize a source ``3 x 4``/``4 x 4`` affine matrix exactly.

        Integer parent-lattice shifts in the origin column are quotient
        equivalent and are canonicalized modulo one.  No approximation is
        accepted unless the reconstructed source matrix lies within the frozen
        tolerance.
        """
        source = np.asarray(values, dtype=np.float64)
        if source.shape == (3, 4):
            homogeneous = np.eye(4, dtype=np.float64)
            homogeneous[:3] = source
        elif source.shape == (4, 4):
            homogeneous = source.copy()
        else:
            raise ValueError("source affine transform must have shape [3,4] or [4,4]")
        if not np.isfinite(homogeneous).all() or not np.allclose(
            homogeneous[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=tolerance, rtol=0.0
        ):
            raise ValueError("source affine transform is nonfinite or nonhomogeneous")
        fractions = [
            Fraction(float(value)).limit_denominator(maximum_denominator)
            for value in homogeneous.ravel()
        ]
        denominator = lcm(*(value.denominator for value in fractions))
        numerators = np.asarray(
            [value.numerator * (denominator // value.denominator) for value in fractions],
            dtype=np.int64,
        ).reshape(4, 4)
        reconstructed = numerators.astype(np.float64) / denominator
        if not np.allclose(reconstructed, homogeneous, atol=tolerance, rtol=0.0):
            raise ValueError("source affine transform is not rational within the frozen tolerance")
        numerators[:3, 3] %= denominator
        divisor = gcd(denominator, *(abs(int(value)) for value in numerators.ravel()))
        if divisor > 1:
            numerators //= divisor
            denominator //= divisor
        return cls(numerators, denominator)

    def as_float(self) -> FloatArray:
        return self.numerators.astype(np.float64) / self.denominator

    def compact_numerators(self) -> IntArray:
        """Return the only twelve stored affine numerators."""
        return self.numerators[:3].copy()

    def key(self) -> str:
        payload = f"d={self.denominator}|n=" + ",".join(
            str(int(value)) for value in self.compact_numerators().ravel()
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AffineInclusionCertificate:
    parent_order: int
    child_order: int
    maximum_rotation_error: float
    maximum_periodic_translation_error: float
    representative_image_order: int
    representative_kernel_size: int
    contained: bool
    uniform_representative_kernel: bool

    @property
    def passed(self) -> bool:
        return (
            self.contained
            and self.maximum_rotation_error <= 1e-9
            and self.maximum_periodic_translation_error <= 1e-9
        )


def _validate_operations(rotations: IntArray, translations: FloatArray) -> tuple[IntArray, FloatArray]:
    integer_rotations = np.asarray(rotations, dtype=np.int64)
    fractional_translations = np.asarray(translations, dtype=np.float64)
    if (
        integer_rotations.ndim != 3
        or integer_rotations.shape[1:] != (3, 3)
        or fractional_translations.shape != (integer_rotations.shape[0], 3)
        or integer_rotations.shape[0] == 0
    ):
        raise ValueError("Seitz operations require rotations [G,3,3] and translations [G,3]")
    if not np.isfinite(fractional_translations).all():
        raise ValueError("Seitz translations must be finite")
    return integer_rotations, fractional_translations


def certify_affine_subgroup_inclusion(
    parent_rotations: IntArray,
    parent_translations: FloatArray,
    child_rotations: IntArray,
    child_translations: FloatArray,
    transform: RationalAffineTransform,
) -> AffineInclusionCertificate:
    """Certify ``T H T^-1`` as a subset of ``G`` in one batch.

    Translation equality is evaluated on the three-torus.  The implementation
    compares all child/parent operations with broadcasted NumPy arrays, so the
    scientific check has no per-operation Python matching loop.  Distinct
    conventional-cell representatives may collapse after quotienting by the
    parent integer translations.  This is an expected finite-representative
    bookkeeping effect, not a failure of the invertible infinite-space-group
    conjugation.  When the conventional translation lattices differ, the map
    between the two chosen finite representative tables need not itself be a
    homomorphism and its fibers need not be uniform.  Image/fiber counts are
    therefore diagnostic only; exact containment is the scientific check.
    """
    parent_r, parent_t = _validate_operations(parent_rotations, parent_translations)
    child_r, child_t = _validate_operations(child_rotations, child_translations)
    matrix = transform.as_float()
    basis = matrix[:3, :3]
    origin = matrix[:3, 3]
    inverse = np.linalg.inv(basis)
    mapped_rotations = np.einsum(
        "ij,hjk,kl->hil", basis, child_r.astype(np.float64), inverse, optimize=True
    )
    rounded_rotations = np.rint(mapped_rotations).astype(np.int64)
    maximum_rotation_error = float(
        np.max(np.abs(mapped_rotations - rounded_rotations), initial=0.0)
    )
    mapped_translations = (
        np.einsum("ij,hj->hi", basis, child_t, optimize=True)
        + origin[None, :]
        - np.einsum("hij,j->hi", mapped_rotations, origin, optimize=True)
    )
    same_rotation = np.all(
        rounded_rotations[:, None, :, :] == parent_r[None, :, :, :], axis=(2, 3)
    )
    difference = mapped_translations[:, None, :] - parent_t[None, :, :]
    difference -= np.rint(difference)
    translation_error = np.max(np.abs(difference), axis=2)
    translation_error = np.where(same_rotation, translation_error, np.inf)
    best_parent = np.argmin(translation_error, axis=1)
    best_error = translation_error[np.arange(child_r.shape[0]), best_parent]
    maximum_translation_error = float(np.max(best_error, initial=0.0))
    contained = bool(np.all(best_error <= 1e-9))
    valid_parent = best_parent[best_error <= 1e-9]
    _, fiber_sizes = np.unique(valid_parent, return_counts=True)
    image_order = int(fiber_sizes.size)
    uniform_kernel = bool(
        contained and image_order > 0 and np.unique(fiber_sizes).size == 1
    )
    kernel_size = int(fiber_sizes[0]) if uniform_kernel else 0
    return AffineInclusionCertificate(
        parent_order=int(parent_r.shape[0]),
        child_order=int(child_r.shape[0]),
        maximum_rotation_error=maximum_rotation_error,
        maximum_periodic_translation_error=maximum_translation_error,
        representative_image_order=image_order,
        representative_kernel_size=kernel_size,
        contained=contained,
        uniform_representative_kernel=uniform_kernel,
    )


def normalized_relation_variant(relations: Iterable[Iterable[str]]) -> tuple[tuple[str, ...], ...]:
    """Canonicalize physically unordered child-orbit labels for one embedding."""
    return tuple(tuple(sorted(str(label) for label in children)) for children in relations)


def wyckoff_multiset_has_exact_cover(
    observed_labels: Iterable[str],
    relation_variant: Iterable[Iterable[str]],
) -> bool:
    """Return whether parent Wyckoff splittings exactly cover child orbit labels.

    A Wyckoff type may occur more than once with different free parameters, so
    each nonempty splitting pattern is reusable.  The memoized multiset search
    chooses the first remaining label and only explores patterns containing
    it; it never enumerates atom-level permutations.
    """
    target = Counter(str(value) for value in observed_labels)
    labels = tuple(sorted(target))
    if not labels:
        return True
    patterns = sorted(
        {
            tuple(sorted(Counter(str(value) for value in pattern).items()))
            for pattern in relation_variant
            if tuple(pattern)
        }
    )
    by_label: dict[str, list[tuple[tuple[str, int], ...]]] = {
        label: [] for label in labels
    }
    for pattern in patterns:
        if any(label not in by_label for label, _ in pattern):
            continue
        for label, _ in pattern:
            by_label[label].append(pattern)
    initial = tuple(target[label] for label in labels)

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int, ...]) -> bool:
        if not any(remaining):
            return True
        first = next(index for index, value in enumerate(remaining) if value)
        for pattern in by_label[labels[first]]:
            changed = list(remaining)
            valid = True
            for label, count in pattern:
                index = labels.index(label)
                if changed[index] < count:
                    valid = False
                    break
                changed[index] -= count
            if valid and solve(tuple(changed)):
                return True
        return False

    return solve(initial)


def species_wyckoff_exact_cover(
    observed_sites: Iterable[tuple[Hashable, str]],
    relation_variant: Iterable[Iterable[str]],
) -> bool:
    """Apply the exact Wyckoff multiset cover independently to each species."""
    by_species: dict[Hashable, list[str]] = {}
    for species, label in observed_sites:
        by_species.setdefault(species, []).append(str(label))
    frozen_relation = tuple(tuple(value) for value in relation_variant)
    return all(
        wyckoff_multiset_has_exact_cover(labels, frozen_relation)
        for labels in by_species.values()
    )
