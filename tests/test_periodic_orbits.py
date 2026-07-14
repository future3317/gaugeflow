import numpy as np

from gaugeflow.periodic_orbits import audit_unlabeled_periodic_site_orbits


def test_unlabeled_bcc_sites_share_one_orbit_despite_different_species():
    audit = audit_unlabeled_periodic_site_orbits(
        np.eye(3),
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        np.array([5, 7]),
        symprec=1e-4,
        mapping_tolerance=1e-4,
    )
    full = audit["full_o3_scalar"]
    assert full["operation_count"] >= 2
    assert [record["site_indices"] for record in full["orbits"]] == [[0, 1]]
    assert full["mixed_orbit_count"] == 1
    assert np.isclose(full["deterministic_equivariant_fixed_cif_accuracy_ceiling"], 0.5)
    assert audit["a11_g_decision"] == "stochastic_assignment_and_quotient_supervision_required"


def test_unlabeled_asymmetric_sites_remain_distinguishable_and_species_are_not_inputs():
    lattice = np.array([[3.1, 0.0, 0.0], [0.2, 4.3, 0.0], [0.1, 0.3, 5.2]])
    frac = np.array([[0.0, 0.0, 0.0], [0.173, 0.291, 0.417], [0.611, 0.229, 0.083]])
    audit = audit_unlabeled_periodic_site_orbits(
        lattice,
        frac,
        np.array([5, 7, 49]),
        symprec=1e-4,
        mapping_tolerance=1e-4,
    )
    full = audit["full_o3_scalar"]
    assert full["operation_count"] == 1
    assert [record["site_indices"] for record in full["orbits"]] == [[0], [1], [2]]
    assert full["mixed_orbit_count"] == 0
    assert np.isclose(full["deterministic_equivariant_fixed_cif_accuracy_ceiling"], 1.0)
    assert audit["a11_g_decision"] == "geometry_only_authorized"


def test_proper_and_full_partitions_are_reported_with_identity_permutations():
    audit = audit_unlabeled_periodic_site_orbits(
        np.eye(3),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([14]),
        symprec=1e-4,
        mapping_tolerance=1e-4,
    )
    for mode in ("proper_so3", "full_o3_scalar"):
        operations = audit[mode]["operations"]
        assert any(operation["permutation"] == [0] for operation in operations)
    assert all(operation["cartesian_determinant"] > 0 for operation in audit["proper_so3"]["operations"])
