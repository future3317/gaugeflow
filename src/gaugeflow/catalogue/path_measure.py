"""Multiplicity-free reference measure for parent--distortion paths."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable


@dataclass(frozen=True, order=True)
class RealizedPathClass:
    """One already-realized physical class used by the H0-E compiler."""

    supercell_index: int
    active_branch_count: int
    physical_key: str

    def __post_init__(self) -> None:
        if self.supercell_index < 1:
            raise ValueError("supercell index must be positive")
        if self.active_branch_count not in (1, 2):
            raise ValueError("v1 paths permit exactly one or two active OPD branches")
        if not self.physical_key:
            raise ValueError("physical path key must be nonempty")


def allocate_reference_measure(
    classes: Iterable[RealizedPathClass],
    *,
    exact_mass: float = 0.5,
) -> tuple[tuple[RealizedPathClass, float], ...]:
    """Allocate the frozen physical-class measure without tuple bias.

    Distorted mass is uniform first over represented supercell indices, then
    over represented active-branch counts, and finally over unique physical
    classes in that bucket.  Duplicate enumeration tuples are removed before
    normalization and therefore cannot change the measure.
    """
    if not 0.0 <= exact_mass <= 1.0:
        raise ValueError("exact branch mass must lie in [0,1]")
    unique = tuple(sorted(set(classes)))
    if not unique:
        if exact_mass != 1.0:
            raise ValueError("nonzero distorted mass requires realized path classes")
        return ()
    indices = sorted({value.supercell_index for value in unique})
    distorted_mass = 1.0 - exact_mass
    output: list[tuple[RealizedPathClass, float]] = []
    for index in indices:
        by_index = [value for value in unique if value.supercell_index == index]
        branch_counts = sorted({value.active_branch_count for value in by_index})
        bucket_mass = distorted_mass / len(indices) / len(branch_counts)
        for branch_count in branch_counts:
            bucket = [
                value for value in by_index if value.active_branch_count == branch_count
            ]
            class_mass = bucket_mass / len(bucket)
            output.extend((value, class_mass) for value in bucket)
    expected = distorted_mass
    if abs(fsum(mass for _, mass in output) - expected) > 1e-12:
        raise RuntimeError("physical path measure failed normalization")
    return tuple(sorted(output))
