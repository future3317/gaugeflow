import json
from dataclasses import replace
from pathlib import Path

import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.matpes_data import (
    collate_matpes_records,
    fit_functional_physical_normalizer,
    fit_functional_physical_normalizer_from_batches,
    matpes_iid_split,
    matpes_stress_kbar_to_kelvin_gpa,
    parse_matpes_row,
)
from gaugeflow.production.physical_pretraining import (
    CartesianPhysicalHeads,
    FunctionalPhysicalNormalizer,
    PhysicalLossDenominators,
    PhysicalLossOutput,
    PhysicalPredictions,
    PhysicalRepresentationModel,
    PhysicalTargets,
    kelvin_to_symmetric_cartesian,
    load_functional_physical_normalizer,
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


def test_clean_physical_backbone_is_covariant_and_skips_generation_heads() -> None:
    torch.manual_seed(8)
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        radial_cutoff=4.0,
        edge_dim=8,
        angular_channels=2,
        edge_refresh_rank=4,
    ).eval()
    elements = torch.tensor([10, 16])
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]])
    lattice = torch.diag(torch.tensor([5.0, 6.0, 7.0])).unsqueeze(0)
    batch = torch.zeros(2, dtype=torch.long)
    rotation = _rotation()
    carrier_calls = 0

    def count_carrier_calls(_module: object, _inputs: object, _output: object) -> None:
        nonlocal carrier_calls
        carrier_calls += 1

    handle = model.coordinate_carrier.register_forward_hook(count_carrier_calls)
    with torch.no_grad():
        reference = model.forward_physical_features(elements, coordinates, lattice, batch)
        transformed = model.forward_physical_features(
            elements,
            coordinates,
            lattice @ rotation.T,
            batch,
        )
    handle.remove()
    assert carrier_calls == 0
    assert torch.allclose(reference.node_scalar, transformed.node_scalar, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        transformed.node_vectors,
        reference.node_vectors @ rotation.T,
        atol=3e-5,
        rtol=3e-5,
    )


def test_physical_transfer_gradients_reach_heads_and_shared_backbone() -> None:
    torch.manual_seed(9)
    backbone = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        radial_cutoff=4.0,
        edge_dim=8,
        angular_channels=2,
        edge_refresh_rank=4,
    )
    model = PhysicalRepresentationModel(backbone, teacher_dim=3)
    elements = torch.tensor([10, 16])
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]])
    lattice = torch.diag(torch.tensor([5.0, 6.0, 7.0])).unsqueeze(0)
    batch = torch.zeros(2, dtype=torch.long)
    prediction = model(elements, coordinates, lattice, batch, torch.zeros(1, dtype=torch.long))
    alternate = model(elements, coordinates, lattice, batch, torch.ones(1, dtype=torch.long))
    assert not torch.allclose(prediction.energy_per_atom, alternate.energy_per_atom)
    loss = (
        prediction.energy_per_atom.square().mean()
        + prediction.forces.square().mean()
        + prediction.stress_kelvin.square().mean()
        + prediction.teacher_features.square().mean()
    )
    loss.backward()
    assert backbone.blocks[0].scalar_update[0].weight.grad is not None
    assert float(backbone.blocks[0].scalar_update[0].weight.grad.norm()) > 0.0
    assert model.heads.energy_head[0].weight.grad is not None
    assert float(model.heads.energy_head[0].weight.grad.norm()) > 0.0


def test_clean_physical_backbone_keeps_geometry_fp32_under_cuda_bf16() -> None:
    if not torch.cuda.is_available():
        return
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        radial_cutoff=4.0,
        edge_dim=8,
        angular_channels=2,
        edge_refresh_rank=4,
    ).cuda()
    elements = torch.tensor([10, 16], device="cuda")
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]],
        device="cuda",
    )
    lattice = torch.diag(torch.tensor([5.0, 6.0, 7.0], device="cuda")).unsqueeze(0)
    batch = torch.zeros(2, dtype=torch.long, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = model.forward_physical_features(elements, coordinates, lattice, batch)
    assert features.node_scalar.dtype == torch.float32
    assert features.node_vectors.dtype == torch.float32
    assert torch.isfinite(features.node_scalar).all()
    assert torch.isfinite(features.node_vectors).all()


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
        teacher_features=torch.zeros(4, 3),
    )
    target = PhysicalTargets(
        energy_per_atom=torch.tensor([1000.0, 2.0]),
        forces=torch.tensor([[1.0] * 3] + [[2.0] * 3] * 3),
        stress_kelvin=torch.full((2, 6), 1000.0),
        teacher_features=torch.full((4, 3), 1000.0),
        energy_mask=torch.tensor([False, True]),
        force_mask=torch.ones(4, dtype=torch.bool),
        stress_mask=torch.zeros(2, dtype=torch.bool),
        teacher_mask=torch.zeros(4, dtype=torch.bool),
    )
    output = physical_multitask_loss(prediction, target, batch)
    assert torch.allclose(output.energy_loss, torch.tensor(4.0))
    assert torch.allclose(output.force_loss, torch.tensor(2.5))
    assert output.stress_loss == 0.0 and output.feature_loss == 0.0
    assert torch.allclose(output.loss, torch.tensor(6.5))


def test_physical_loss_global_denominators_sum_disjoint_rank_contributions() -> None:
    denominators = PhysicalLossDenominators(energy=2, force=2, stress=2, feature=1)

    def shard(error: float, *, teacher: bool) -> PhysicalLossOutput:
        batch = torch.zeros(1, dtype=torch.long)
        prediction = PhysicalPredictions(
            energy_per_atom=torch.tensor([error]),
            forces=torch.full((1, 3), error),
            stress_kelvin=torch.full((1, 6), error),
            teacher_features=torch.tensor([[1.0, 0.0]]),
        )
        target = PhysicalTargets(
            energy_per_atom=torch.zeros(1),
            forces=torch.zeros(1, 3),
            stress_kelvin=torch.zeros(1, 6),
            teacher_features=torch.tensor([[0.0, 1.0]]),
            energy_mask=torch.ones(1, dtype=torch.bool),
            force_mask=torch.ones(1, dtype=torch.bool),
            stress_mask=torch.ones(1, dtype=torch.bool),
            teacher_mask=torch.tensor([teacher]),
        )
        return physical_multitask_loss(
            prediction,
            target,
            batch,
            denominators=denominators,
        )

    first = shard(1.0, teacher=True)
    second = shard(3.0, teacher=False)
    assert torch.allclose(first.energy_loss + second.energy_loss, torch.tensor(5.0))
    assert torch.allclose(first.force_loss + second.force_loss, torch.tensor(5.0))
    assert torch.allclose(first.stress_loss + second.stress_loss, torch.tensor(5.0))
    assert torch.allclose(first.feature_loss + second.feature_loss, torch.tensor(1.0))


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
        teacher_features=torch.zeros(2, 2),
        energy_mask=torch.ones(1, dtype=torch.bool),
        force_mask=torch.ones(2, dtype=torch.bool),
        stress_mask=torch.ones(1, dtype=torch.bool),
        teacher_mask=torch.zeros(2, dtype=torch.bool),
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


def test_physical_normalizer_loader_verifies_index_provenance(tmp_path: Path) -> None:
    index_manifest = tmp_path / "manifest.json"
    index_manifest.write_text('{"qualified": true}\n', encoding="utf-8")
    normalizer_path = tmp_path / "normalizer.json"
    payload = {
        "schema": "gaugeflow.matpes_physical_normalizer.v1",
        "qualified": True,
        "index_manifest": str(index_manifest),
        "index_manifest_sha256": sha256_file(index_manifest),
        "functional_vocabulary": {"PBE": 0, "r2SCAN": 1},
        "energy_location": [-2.0, -3.0],
        "energy_scale": [1.0, 2.0],
        "force_scale": [3.0, 4.0],
        "stress_isotropic_location": [5.0, 6.0],
        "stress_scale": [7.0, 8.0],
    }
    normalizer_path.write_text(json.dumps(payload), encoding="utf-8")
    normalizer, vocabulary = load_functional_physical_normalizer(normalizer_path)
    assert vocabulary == {"PBE": 0, "r2SCAN": 1}
    assert normalizer.energy_location.tolist() == [-2.0, -3.0]
    index_manifest.write_text('{"qualified": false}\n', encoding="utf-8")
    try:
        load_functional_physical_normalizer(normalizer_path)
    except ValueError:
        pass
    else:
        raise AssertionError("stale physical normalization provenance was accepted")


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


def test_matpes_collation_preserves_type_matched_per_atom_teacher_features() -> None:
    row = {
        "matpes_id": "teacher-row",
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
        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "stress": [0.0] * 6,
    }
    feature = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    record = replace(parse_matpes_row(row), teacher_features=feature)
    packed = collate_matpes_records(
        [record], functional_vocabulary={"PBE": 0}, teacher_dim=3
    )
    assert torch.equal(packed.targets.teacher_features, feature)
    assert packed.targets.teacher_mask.tolist() == [True, True]


def test_matpes_train_statistics_are_functional_and_streaming() -> None:
    row = {
        "matpes_id": "base",
        "functional": "PBE",
        "nsites": 1,
        "structure": {
            "lattice": {"matrix": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]},
            "sites": [{"species": [{"element": "Si", "occu": 1.0}], "abc": [0.0, 0.0, 0.0]}],
        },
        "energy": -4.0,
        "forces": [[1.0, 0.0, 0.0]],
        "stress": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
    }
    base = parse_matpes_row(row)
    records = [
        base,
        replace(
            base,
            material_id="pbe-2",
            energy_per_atom_ev=torch.tensor(-2.0),
            forces_ev_per_angstrom=torch.tensor([[2.0, 0.0, 0.0]]),
        ),
        replace(base, material_id="r2-1", functional="r2SCAN", energy_per_atom_ev=torch.tensor(-10.0)),
        replace(base, material_id="r2-2", functional="r2SCAN", energy_per_atom_ev=torch.tensor(-8.0)),
    ]
    vocabulary = {"PBE": 0, "r2SCAN": 1}
    normalizer = fit_functional_physical_normalizer(records, functional_vocabulary=vocabulary)
    batched_normalizer = fit_functional_physical_normalizer_from_batches(
        [
            collate_matpes_records(records[:2], functional_vocabulary=vocabulary, teacher_dim=2),
            collate_matpes_records(records[2:], functional_vocabulary=vocabulary, teacher_dim=2),
        ],
        functional_vocabulary=vocabulary,
    )
    for field in (
        "energy_location",
        "energy_scale",
        "force_scale",
        "stress_isotropic_location",
        "stress_scale",
    ):
        assert torch.allclose(getattr(normalizer, field), getattr(batched_normalizer, field))
    assert torch.allclose(normalizer.energy_location, torch.tensor([-3.0, -9.0]))
    assert torch.allclose(normalizer.energy_scale, torch.ones(2))
    packed = collate_matpes_records(records, functional_vocabulary=vocabulary, teacher_dim=2)
    normalized = normalizer.normalize(packed.targets, packed.functional_index, packed.batch)
    assert torch.allclose(normalized.energy_per_atom, torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    assert bool(torch.isfinite(normalized.forces).all())
    assert bool(torch.isfinite(normalized.stress_kelvin).all())
