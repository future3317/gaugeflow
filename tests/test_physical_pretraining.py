import torch

from gaugeflow.production.matpes_data import (
    matpes_stress_kbar_to_kelvin_gpa,
    parse_matpes_row,
)
from gaugeflow.production.physical_pretraining import (
    CartesianPhysicalHeads,
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
