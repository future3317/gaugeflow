import torch

from gaugeflow.production.matpes_data import (
    collate_matpes_records,
    matpes_iid_split,
    matpes_stress_kbar_to_kelvin_gpa,
    parse_matpes_row,
)
from gaugeflow.production.physical_pretraining import (
    CartesianPhysicalHeads,
    FunctionalPhysicalNormalizer,
    PhysicalPredictions,
    PhysicalTargets,
    kelvin_to_symmetric_cartesian,
    physical_multitask_loss,
    symmetric_cartesian_to_kelvin,
)


def _rotation() -> torch.Tensor:
    axis = torch.tensor([1.0, 2.0, -1.0])
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.73)
    cross = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return torch.eye(3) + angle.sin() * cross + (1.0 - angle.cos()) * (cross @ cross)


def test_cartesian_physical_heads_are_rotation_covariant() -> None:
    torch.manual_seed(4)
    head = CartesianPhysicalHeads(scalar_dim=8, vector_dim=3, teacher_dim=5)
    scalar = torch.randn(5, 8)
    vectors = torch.randn(5, 3, 3)
    batch = torch.tensor([0, 0, 1, 1, 1])
    rotation = _rotation()
    reference = head(scalar, vectors, batch, 2)
    transformed = head(scalar, vectors @ rotation.T, batch, 2)
    reference_stress = kelvin_to_symmetric_cartesian(reference.stress_kelvin)
    transformed_stress = kelvin_to_symmetric_cartesian(transformed.stress_kelvin)
    expected_stress = rotation @ reference_stress @ rotation.T
    assert torch.allclose(reference.energy_per_atom, transformed.energy_per_atom, atol=1e-6)
    assert torch.allclose(reference.teacher_features, transformed.teacher_features, atol=1e-6)
    assert torch.allclose(transformed.forces, reference.forces @ rotation.T, atol=2e-5)
    assert torch.allclose(transformed_stress, expected_stress, atol=2e-5)


def test_kelvin_round_trip_is_orthonormal() -> None:
    raw = torch.tensor(
        [[[2.0, -0.3, 0.7], [-0.3, 1.0, 0.2], [0.7, 0.2, -1.0]]]
    )
    kelvin = symmetric_cartesian_to_kelvin(raw)
    assert torch.allclose(kelvin_to_symmetric_cartesian(kelvin), raw)
    assert torch.allclose(kelvin.square().sum(), raw.square().sum())


def test_physical_loss_masks_missing_labels_and_weights_graphs_equally() -> None:
    batch = torch.tensor([0, 1, 1, 1])
    prediction = PhysicalPredictions(
        energy_per_atom=torch.zeros(2),
        forces=torch.zeros(4, 3),
        stress_kelvin=torch.zeros(2, 6),
        teacher_features=torch.zeros(2, 3),
    )
    target = PhysicalTargets(
        energy_per_atom=torch.tensor([1000.0, 2.0]),
        forces=torch.tensor([[1.0] * 3] + [[2.0] * 3] * 3),
        stress_kelvin=torch.full((2, 6), 1000.0),
        teacher_features=torch.full((2, 3), 1000.0),
        energy_mask=torch.tensor([False, True]),
        force_mask=torch.ones(4, dtype=torch.bool),
        stress_mask=torch.zeros(2, dtype=torch.bool),
        teacher_mask=torch.zeros(2, dtype=torch.bool),
    )
    output = physical_multitask_loss(prediction, target, batch)
    assert torch.allclose(output.energy_loss, torch.tensor(4.0))
    assert torch.allclose(output.force_loss, torch.tensor(2.5))
    assert output.stress_loss == 0.0 and output.feature_loss == 0.0
    assert torch.allclose(output.loss, torch.tensor(6.5))


def test_matpes_record_preserves_units_and_missing_label_masks() -> None:
    row = {
        "matpes_id": "sample-1",
        "functional": "PBE",
        "nsites": 2,
        "structure": {
            "lattice": {"matrix": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]},
            "sites": [
                {"species": [{"element": "Na", "occu": 1.0}], "abc": [0.0, 0.0, 0.0]},
                {"species": [{"element": "Cl", "occu": 1.0}], "abc": [0.5, 0.5, 0.5]},
            ],
        },
        "energy": -8.0,
        "cohesive_energy_per_atom": -2.5,
        "forces": None,
        "stress": [10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
    }
    record = parse_matpes_row(row)
    assert record.element_tokens.tolist() == [10, 16]
    assert record.energy_per_atom_ev == -4.0
    assert record.energy_present and not record.forces_present and record.stress_present
    assert torch.equal(record.forces_ev_per_angstrom, torch.zeros(2, 3))
    expected = torch.tensor([-1.0, -2.0, -3.0, -0.4 * 2**0.5, -0.5 * 2**0.5, -0.6 * 2**0.5])
    assert torch.allclose(record.stress_kelvin_gpa, expected)
    assert torch.allclose(record.stress_kelvin_gpa.double(), matpes_stress_kbar_to_kelvin_gpa(row["stress"]))
    cohesive = parse_matpes_row(row, energy_target="cohesive_energy_per_atom")
    assert cohesive.energy_per_atom_ev == -2.5


def test_matpes_split_groups_functionals_by_material_identity() -> None:
    material_ids = [f"matpes-{index}" for index in range(1000)]
    first = [matpes_iid_split(value, seed=5705) for value in material_ids]
    second = [matpes_iid_split(value, seed=5705) for value in material_ids]
    assert first == second
    assert {split: first.count(split) for split in set(first)} == {
        "train": 884,
        "calibration": 56,
        "test": 60,
    }
    assert matpes_iid_split("shared-id", seed=5705) == matpes_iid_split(
        "shared-id", seed=5705
    )


def test_functional_normalization_preserves_force_and_stress_covariance() -> None:
    rotation = _rotation()
    stress = torch.tensor([[[2.0, 0.3, -0.2], [0.3, 1.0, 0.4], [-0.2, 0.4, 3.0]]])
    force = torch.tensor([[1.0, 2.0, -3.0], [-2.0, 0.5, 1.0]])
    target = PhysicalTargets(
        energy_per_atom=torch.tensor([-4.0]),
        forces=force,
        stress_kelvin=symmetric_cartesian_to_kelvin(stress),
        teacher_features=torch.zeros(1, 2),
        energy_mask=torch.ones(1, dtype=torch.bool),
        force_mask=torch.ones(2, dtype=torch.bool),
        stress_mask=torch.ones(1, dtype=torch.bool),
        teacher_mask=torch.zeros(1, dtype=torch.bool),
    )
    rotated = PhysicalTargets(
        energy_per_atom=target.energy_per_atom,
        forces=force @ rotation.T,
        stress_kelvin=symmetric_cartesian_to_kelvin(rotation @ stress @ rotation.T),
        teacher_features=target.teacher_features,
        energy_mask=target.energy_mask,
        force_mask=target.force_mask,
        stress_mask=target.stress_mask,
        teacher_mask=target.teacher_mask,
    )
    normalizer = FunctionalPhysicalNormalizer(
        energy_location=torch.tensor([-5.0]),
        energy_scale=torch.tensor([2.0]),
        force_scale=torch.tensor([4.0]),
        stress_isotropic_location=torch.tensor([1.5]),
        stress_scale=torch.tensor([3.0]),
    )
    batch = torch.zeros(2, dtype=torch.long)
    normalized = normalizer.normalize(target, torch.zeros(1, dtype=torch.long), batch)
    rotated_normalized = normalizer.normalize(rotated, torch.zeros(1, dtype=torch.long), batch)
    assert torch.allclose(rotated_normalized.forces, normalized.forces @ rotation.T)
    assert torch.allclose(
        kelvin_to_symmetric_cartesian(rotated_normalized.stress_kelvin),
        rotation @ kelvin_to_symmetric_cartesian(normalized.stress_kelvin) @ rotation.T,
        atol=1e-6,
    )


def test_matpes_collation_packs_graphs_and_masks_without_ids() -> None:
    base = {
        "matpes_id": "pbe-id",
        "functional": "PBE",
        "nsites": 1,
        "structure": {
            "lattice": {"matrix": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]},
            "sites": [{"species": [{"element": "Si", "occu": 1.0}], "abc": [0.0, 0.0, 0.0]}],
        },
        "energy": -4.0,
        "forces": [[0.1, 0.2, 0.3]],
        "stress": None,
    }
    second = dict(base)
    second.update(matpes_id="r2scan-id", functional="r2SCAN", energy=None)
    records = [parse_matpes_row(base), parse_matpes_row(second)]
    packed = collate_matpes_records(
        records,
        functional_vocabulary={"PBE": 0, "r2SCAN": 1},
        teacher_dim=4,
    )
    assert packed.element_tokens.shape == (2,)
    assert packed.fractional_coordinates.shape == (2, 3)
    assert packed.lattice.shape == (2, 3, 3)
    assert packed.batch.tolist() == [0, 1]
    assert packed.functional_index.tolist() == [0, 1]
    assert packed.targets.energy_mask.tolist() == [True, False]
    assert packed.targets.force_mask.tolist() == [True, True]
    assert packed.targets.stress_mask.tolist() == [False, False]
    assert packed.targets.teacher_features.shape == (2, 4)
