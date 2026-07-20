from scripts.audit_h1a_assignment_site_resolved_carrier import audit_candidate_geometry


def test_site_resolved_carrier_detects_missing_supercell_geometry() -> None:
    candidate = {
        "cell_index": 2,
        "child_site_count": 4,
        "parent_site_count": 2,
        "parent_fractional": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        "parent_lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "parent_action_permutations": [[0, 1, 2, 3]],
    }
    result = audit_candidate_geometry(candidate)
    assert result["action_node_aligned"]
    assert not result["full_site_geometry"]
    assert not result["expanded_geometry_field"]
    assert not result["supercell_hnf_present"]
    assert not result["translation_cosets_present"]
