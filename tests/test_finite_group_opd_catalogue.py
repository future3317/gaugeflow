from __future__ import annotations

import numpy as np

from gaugeflow.catalogue.finite_group import (
    FiniteGroup,
    RealIrrep,
    enumerate_fixed_space_projectors,
    enumerate_opd_classes,
    enumerate_real_irreps,
    intersect_stabilizer_bitsets,
    stabilizer_bitset,
)
from scripts.build_h0_d_opd_catalogue_v2 import _spgrep_modulation_reference_agreement


def _cyclic_group(order: int) -> FiniteGroup:
    table = np.fromfunction(lambda i, j: (i + j) % order, (order, order), dtype=int)
    return FiniteGroup.from_cayley_table(table, [f"r{index}" for index in range(order)])


def test_finite_group_validation_and_complete_real_irreps():
    c3 = _cyclic_group(3)
    assert c3.identity == 0
    assert c3.inverses.tolist() == [0, 2, 1]
    irreps = enumerate_real_irreps(c3)
    assert [irrep.dimension for irrep in irreps] == [1, 2]
    assert [irrep.frobenius_schur_indicator for irrep in irreps] == [1, 0]
    assert sum(irrep.complex_regular_contribution for irrep in irreps) == c3.order


def test_two_dimensional_c4_vector_irrep_has_axial_and_generic_opd_classes():
    c4 = _cyclic_group(4)
    matrices = np.stack(
        [
            np.array(
                [[np.cos(index * np.pi / 2), -np.sin(index * np.pi / 2)],
                 [np.sin(index * np.pi / 2), np.cos(index * np.pi / 2)]],
                dtype=np.float64,
            )
            for index in range(4)
        ]
    )
    irrep = RealIrrep(matrices, 0, tuple(np.trace(matrices, axis1=1, axis2=2)))
    branches = enumerate_opd_classes(c4, irrep)
    assert len(branches) == 1
    assert branches[0].fixed_dimension == 2
    assert branches[0].stabilizer == (0,)
    assert branches[0].stabilizer_words == (1,)


def test_packed_stabilizer_intersection_is_exact_across_multiple_words():
    left = stabilizer_bitset([0, 63, 64, 130], group_order=192)
    right = stabilizer_bitset([0, 64, 129, 130], group_order=192)
    assert intersect_stabilizer_bitsets(left, right) == stabilizer_bitset(
        [0, 64, 130], group_order=192
    )


def test_opd_physical_key_is_basis_gauge_and_enumeration_invariant():
    c4 = _cyclic_group(4)
    angle = np.arange(4) * np.pi / 2
    matrices = np.stack(
        [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angle]
    )
    gauge = np.array([[0.6, -0.8], [0.8, 0.6]])
    base = RealIrrep(matrices, 0, tuple(np.trace(matrices, axis1=1, axis2=2)))
    changed = RealIrrep(
        np.einsum("ab,gbc,dc->gad", gauge, matrices, gauge),
        0,
        base.character_key,
    )
    assert [branch.physical_key for branch in enumerate_opd_classes(c4, base)] == [
        branch.physical_key for branch in enumerate_opd_classes(c4, changed)
    ]

    permutation = np.array([0, 3, 2, 1])
    inverse = np.argsort(permutation)
    permuted_table = inverse[c4.table[permutation[:, None], permutation[None, :]]]
    permuted = FiniteGroup.from_cayley_table(
        permuted_table,
        tuple(c4.element_keys[index] for index in permutation),
    )
    permuted_rep = RealIrrep(matrices[permutation], 0, base.character_key)
    assert [branch.physical_key for branch in enumerate_opd_classes(c4, base)] == [
        branch.physical_key for branch in enumerate_opd_classes(permuted, permuted_rep)
    ]


def test_fixed_space_lattice_is_equivalent_to_full_subgroup_enumeration():
    c4 = _cyclic_group(4)
    angle = np.arange(4) * np.pi / 2
    matrices = np.stack(
        [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angle]
    )
    irrep = RealIrrep(matrices, 0, tuple(np.trace(matrices, axis1=1, axis2=2)))
    direct = enumerate_fixed_space_projectors(matrices)
    subgroup_projectors = []
    for subgroup in c4.enumerate_subgroups():
        reynolds = matrices[np.asarray(subgroup)].mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (reynolds + reynolds.T))
        basis = eigenvectors[:, eigenvalues > 0.5]
        subgroup_projectors.append(basis @ basis.T)
    assert all(
        any(np.allclose(left, right, atol=1e-8, rtol=1e-8) for right in direct)
        for left in subgroup_projectors
    )
    assert all(
        any(np.allclose(left, right, atol=1e-8, rtol=1e-8) for right in subgroup_projectors)
        for left in direct
    )
    assert len(direct) <= len(c4.enumerate_subgroups())
    assert enumerate_opd_classes(c4, irrep)


def test_sg221_polar_vector_matches_independent_isotropy_enumerator():
    result = _spgrep_modulation_reference_agreement()
    assert result["passed"]
    assert result["our_class_count"] == result["reference_class_count"] == 6
