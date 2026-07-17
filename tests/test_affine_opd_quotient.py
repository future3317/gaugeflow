from __future__ import annotations

import numpy as np

from gaugeflow.catalogue import (
    AffineQuotient,
    PrimitiveSpaceGroup,
    RealizedPathClass,
    allocate_reference_measure,
    build_compact_displacement_action,
    canonical_supercell_orbits,
    enumerate_real_irreps,
    enumerate_upper_hnfs,
    integer_lattice_coset_representatives,
    real_irrep_multiplicity,
)


def test_integer_lattice_cosets_support_off_diagonal_non_hnf_embedding_basis():
    basis = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64)
    representatives = integer_lattice_coset_representatives(basis)
    assert representatives.shape == (2, 3)
    difference = np.linalg.solve(
        basis.astype(np.float64),
        (representatives[1] - representatives[0]).astype(np.float64),
    )
    assert not np.allclose(difference, np.rint(difference), atol=1e-12, rtol=0.0)


def _p_minus_one() -> PrimitiveSpaceGroup:
    return PrimitiveSpaceGroup.from_operations(
        np.stack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]),
        np.zeros((2, 3)),
    )


def test_hnf_enumeration_is_complete_through_index_four():
    hnfs = enumerate_upper_hnfs(4)
    counts = {
        determinant: sum(int(round(np.linalg.det(value))) == determinant for value in hnfs)
        for determinant in range(1, 5)
    }
    assert counts == {1: 1, 2: 7, 3: 13, 4: 35}


def test_affine_quotient_retains_translation_cosets_and_closes():
    quotient = AffineQuotient.build(_p_minus_one(), np.diag([2, 1, 1]))
    assert quotient.group.order == 4
    assert quotient.translations.order == 2
    assert len(set(quotient.group.element_keys)) == 4


def test_parent_rotation_and_unimodular_basis_reduce_hnf_orbits():
    parent = PrimitiveSpaceGroup.from_operations(
        np.stack(
            [
                np.eye(3, dtype=int),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=int),
            ]
        ),
        np.zeros((2, 3)),
    )
    orbits = canonical_supercell_orbits(parent, 2)
    assert len(orbits) < len(enumerate_upper_hnfs(2))
    assert any(np.array_equal(value, np.diag([1, 1, 2])) for value in orbits)


def test_row_unimodular_basis_change_preserves_hnf():
    from hsnf import row_style_hermite_normal_form

    hnf = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 2]], dtype=np.int64)
    unimodular = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.int64)
    transformed, _ = row_style_hermite_normal_form(unimodular @ hnf)
    assert np.array_equal(np.asarray(transformed, dtype=np.int64), hnf)


def test_affine_quotient_is_invariant_to_source_operation_order():
    parent = _p_minus_one()
    reversed_parent = PrimitiveSpaceGroup.from_operations(
        parent.rotations[::-1],
        (parent.translation_numerators[::-1] / parent.translation_denominator),
    )
    left = AffineQuotient.build(parent, np.diag([2, 1, 1]))
    right = AffineQuotient.build(reversed_parent, np.diag([2, 1, 1]))
    assert left.group.element_keys == right.group.element_keys
    assert np.array_equal(left.group.table, right.group.table)


def test_compact_displacement_action_matches_full_group_and_irrep_occurrence():
    quotient = AffineQuotient.build(_p_minus_one(), np.diag([2, 1, 1]))
    action = build_compact_displacement_action(
        quotient,
        np.eye(3),
        np.zeros((1, 3)),
        np.ones(1, dtype=int),
    )
    assert action.permutations.shape == (4, 2)
    assert action.cartesian_rotations.shape == (4, 3, 3)
    irreps = enumerate_real_irreps(quotient.group)
    multiplicities = [real_irrep_multiplicity(action, irrep) for irrep in irreps]
    assert sum(value * irrep.dimension for value, irrep in zip(multiplicities, irreps)) == 6
    vector = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    applied = action.apply(vector)
    assert applied.shape == (4, 2, 3)
    assert np.isfinite(applied).all()
    dense = np.zeros((action.group.order, vector.size, vector.size), dtype=np.float64)
    for group_index, (permutation, rotation) in enumerate(
        zip(action.permutations, action.cartesian_rotations)
    ):
        for source, target in enumerate(permutation):
            dense[group_index, 3 * target : 3 * target + 3, 3 * source : 3 * source + 3] = rotation
    expected = np.einsum("gij,j->gi", dense, vector.ravel()).reshape(applied.shape)
    assert np.allclose(applied, expected, atol=1e-12, rtol=1e-12)
    assert action.permutations.nbytes + action.cartesian_rotations.nbytes < dense.nbytes


def test_physical_path_measure_ignores_duplicate_enumeration_tuples():
    classes = [
        RealizedPathClass(2, 1, "a"),
        RealizedPathClass(2, 1, "b"),
        RealizedPathClass(3, 1, "c"),
        RealizedPathClass(3, 2, "d"),
    ]
    base = allocate_reference_measure(classes)
    expanded = allocate_reference_measure([*classes, classes[0], classes[0], classes[3]])
    assert base == expanded
    assert np.isclose(sum(mass for _, mass in base), 0.5, atol=1e-12, rtol=0.0)
