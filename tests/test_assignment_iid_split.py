from scripts.build_h1a_assignment_iid_split import assign_iid_roles


def test_assignment_iid_split_has_fit_support_and_is_deterministic() -> None:
    partitions = {
        "a": (4, (2, 2)),
        "b": (4, (2, 2)),
        "c": (4, (2, 2)),
        "d": (4, (2, 2)),
        "e": (6, (3, 3)),
        "f": (6, (3, 3)),
    }
    first = assign_iid_roles(
        partitions,
        seed=5705,
        holdout_fraction=0.15,
        minimum_partition_materials=3,
    )
    second = assign_iid_roles(
        partitions,
        seed=5705,
        holdout_fraction=0.15,
        minimum_partition_materials=3,
    )
    assert first == second
    assert set(first.values()) == {"iid_fit", "iid_fit_rare", "iid_calibration", "iid_test"}
    supported = {material for material in "abcd"}
    assert any(first[material] == "iid_fit" for material in supported)
    assert any(first[material] == "iid_calibration" for material in supported)
    assert any(first[material] == "iid_test" for material in supported)
    assert first["e"] == first["f"] == "iid_fit_rare"
