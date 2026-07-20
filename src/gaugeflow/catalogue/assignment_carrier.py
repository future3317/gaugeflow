"""Canonical species-free geometry for count-constrained occupation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from hsnf import row_style_hermite_normal_form
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from gaugeflow.catalogue.affine_quotient import TranslationQuotient
from gaugeflow.catalogue.parent_occurrence import (
    GeometryCompleteOccupationalOccurrence,
)
from gaugeflow.geometry import closest_image_displacements_numpy

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class CanonicalAssignmentCarrier:
    """Target-free parent geometry in one canonical HNF node chart."""

    primitive_parent_lattice: FloatArray
    primitive_parent_fractional: FloatArray
    expanded_parent_lattice: FloatArray
    expanded_parent_fractional: FloatArray
    supercell_hnf: IntArray
    translation_cosets: IntArray
    node_parent_site_indices: IntArray
    node_translation_coset_indices: IntArray
    parent_action_permutations: IntArray


@dataclass(frozen=True)
class CanonicalCarrierAlignment:
    """Offline-only certificate connecting canonical and source node gauges."""

    carrier: CanonicalAssignmentCarrier
    source_node_by_carrier_node: IntArray
    maximum_periodic_alignment_error_angstrom: float


def _validate_permutations(permutations: IntArray, node_count: int) -> None:
    if permutations.ndim != 2 or permutations.shape[1] != node_count:
        raise ValueError("parent action must cover every expanded carrier node")
    expected = np.arange(node_count, dtype=np.int64)
    if not np.all(np.sort(permutations, axis=1) == expected[None, :]):
        raise ValueError("parent action contains a non-permutation row")


def canonicalize_assignment_carrier(
    complete: GeometryCompleteOccupationalOccurrence,
    *,
    alignment_tolerance_angstrom: float = 1e-6,
) -> CanonicalCarrierAlignment:
    """Move a certified occurrence to HNF without using terminal coloring.

    The expanded ideal geometry is first changed from the embedding chart to
    the row-HNF chart.  A single periodic linear assignment then conjugates the
    already-certified finite-site action into a coset-major canonical order.
    This assignment uses only two representations of the same species-free
    parent geometry; terminal elements and child displacements are absent.
    """
    if alignment_tolerance_angstrom <= 0.0:
        raise ValueError("carrier alignment tolerance must be positive")
    occurrence = complete.occurrence
    projection = occurrence.projection
    parent_lattice = np.asarray(projection.lattice, dtype=np.float64)
    parent_fractional = np.asarray(projection.fractional, dtype=np.float64) % 1.0
    expanded_lattice = np.asarray(complete.expanded_lattice, dtype=np.float64)
    expanded_fractional = np.asarray(complete.expanded_fractional, dtype=np.float64) % 1.0
    basis = np.asarray(complete.embedding_basis, dtype=np.int64)
    origin = np.asarray(complete.embedding_origin, dtype=np.float64)
    permutations = np.asarray(projection.permutations, dtype=np.int64)
    if (
        parent_lattice.shape != (3, 3)
        or parent_fractional.ndim != 2
        or parent_fractional.shape[1] != 3
        or expanded_lattice.shape != (3, 3)
        or expanded_fractional.ndim != 2
        or expanded_fractional.shape[1] != 3
        or basis.shape != (3, 3)
        or origin.shape != (3,)
    ):
        raise ValueError("geometry-complete carrier has inconsistent shapes")
    if not all(
        np.isfinite(value).all()
        for value in (
            parent_lattice,
            parent_fractional,
            expanded_lattice,
            expanded_fractional,
            origin,
        )
    ):
        raise ValueError("geometry-complete carrier contains non-finite values")

    supercell = basis.T
    hnf, left = row_style_hermite_normal_form(supercell)
    hnf = np.asarray(hnf, dtype=np.int64)
    left = np.asarray(left, dtype=np.int64)
    cell_index = abs(int(round(np.linalg.det(hnf))))
    if cell_index != occurrence.cell_index or cell_index not in (1, 2, 3, 4):
        raise ValueError("canonical HNF does not preserve the certified cell index")
    if not np.array_equal(hnf, left @ supercell):
        raise RuntimeError("row-HNF transformation certificate does not close exactly")
    if abs(int(round(np.linalg.det(left)))) != 1:
        raise RuntimeError("row-HNF basis change is not unimodular")

    quotient = TranslationQuotient.from_supercell(hnf)
    cosets = np.asarray(quotient.representatives, dtype=np.int64)
    parent_sites = parent_fractional.shape[0]
    node_count = parent_sites * cell_index
    if expanded_fractional.shape[0] != node_count:
        raise ValueError("expanded geometry does not close on parent sites times index")
    _validate_permutations(permutations, node_count)

    hnf_inverse = np.linalg.inv(hnf.astype(np.float64))
    canonical_fractional = ((parent_fractional[None, :, :] + cosets[:, None, :]) @ hnf_inverse).reshape(
        node_count, 3
    ) % 1.0
    canonical_shifted = (canonical_fractional - origin @ hnf_inverse) % 1.0

    left_inverse = np.linalg.inv(left.astype(np.float64))
    source_fractional_hnf = (expanded_fractional @ left_inverse) % 1.0
    hnf_lattice = hnf @ parent_lattice
    changed_expanded_lattice = left @ expanded_lattice
    if not np.allclose(
        changed_expanded_lattice,
        hnf_lattice,
        atol=1e-7,
        rtol=1e-7,
    ):
        raise RuntimeError("expanded and primitive HNF lattice charts disagree")

    delta = (canonical_shifted[:, None, :] - source_fractional_hnf[None, :, :]).reshape(-1, 3)
    cartesian, _ = closest_image_displacements_numpy(delta, hnf_lattice)
    cost = np.linalg.norm(cartesian, axis=1).reshape(node_count, node_count)
    carrier_nodes, source_nodes = linear_sum_assignment(cost)
    if not np.array_equal(carrier_nodes, np.arange(node_count)):
        raise RuntimeError("periodic carrier assignment lost a canonical node")
    maximum_error = float(cost[carrier_nodes, source_nodes].max(initial=0.0))
    if maximum_error > alignment_tolerance_angstrom:
        raise ValueError("canonical HNF nodes do not recover the certified expanded geometry")

    carrier_by_source = np.empty(node_count, dtype=np.int64)
    carrier_by_source[source_nodes] = carrier_nodes
    canonical_permutations = carrier_by_source[permutations[:, source_nodes]]
    _validate_permutations(canonical_permutations, node_count)

    carrier = CanonicalAssignmentCarrier(
        primitive_parent_lattice=parent_lattice,
        primitive_parent_fractional=parent_fractional,
        expanded_parent_lattice=hnf_lattice,
        expanded_parent_fractional=canonical_fractional,
        supercell_hnf=hnf,
        translation_cosets=cosets,
        node_parent_site_indices=np.tile(
            np.arange(parent_sites, dtype=np.int64),
            cell_index,
        ),
        node_translation_coset_indices=np.repeat(
            np.arange(cell_index, dtype=np.int64),
            parent_sites,
        ),
        parent_action_permutations=canonical_permutations,
    )
    return CanonicalCarrierAlignment(
        carrier=carrier,
        source_node_by_carrier_node=source_nodes.astype(np.int64, copy=False),
        maximum_periodic_alignment_error_angstrom=maximum_error,
    )
