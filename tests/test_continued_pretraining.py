from __future__ import annotations

import torch

from gaugeflow.production.continued_pretraining import (
    ContinuedPretrainingWeights,
    accumulate_continued_pretraining_step,
    collate_structure_records,
    pack_structure_batch,
    structure_replay_loss,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.matpes_data import MatPESPhysicalRecord, collate_matpes_records
from gaugeflow.production.physical_pretraining import (
    FunctionalPhysicalNormalizer,
    PhysicalRepresentationModel,
)
from gaugeflow.production.physical_training import (
    PhysicalTransferTrainer,
    PhysicalTransferTrainingConfig,
)


def _record(material_id: str, functional: str, nodes: int) -> MatPESPhysicalRecord:
    return MatPESPhysicalRecord(
        material_id=material_id,
        functional=functional,
        element_tokens=torch.arange(nodes, dtype=torch.long),
        fractional_coordinates=torch.arange(nodes * 3, dtype=torch.float32).reshape(nodes, 3)
        / (nodes * 3),
        lattice=torch.eye(3) * 4.0,
        energy_per_atom_ev=torch.tensor(123.0),
        forces_ev_per_angstrom=torch.full((nodes, 3), 456.0),
        stress_kelvin_gpa=torch.full((6,), 789.0),
        energy_present=True,
        forces_present=True,
        stress_present=True,
    )


def _diffusion() -> TensorFreeHybridDiffusion:
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        radial_cutoff=4.0,
        edge_dim=8,
        angular_channels=2,
        edge_refresh_rank=4,
    )
    raw_basis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    basis = torch.linalg.qr(raw_basis, mode="reduced").Q
    standardizer = P1LatticeStandardizer(
        volume_residual_mean=0.0,
        volume_residual_std=1.0,
        shape_mean=torch.zeros(6, dtype=torch.float64),
        shape_basis_columns=basis,
        shape_scales=torch.ones(5, dtype=torch.float64),
    )
    return TensorFreeHybridDiffusion(model, standardizer)


def test_structure_collator_removes_dataset_metadata_and_targets() -> None:
    packed = collate_structure_records([_record("secret-a", "pbe", 2), _record("secret-b", "scan", 3)])
    assert packed.element_tokens.tolist() == [0, 1, 0, 1, 2]
    assert packed.batch.tolist() == [0, 0, 1, 1, 1]
    assert packed.node_counts.tolist() == [2, 3]
    assert not hasattr(packed, "material_id")
    assert not hasattr(packed, "functional")
    assert not hasattr(packed, "energy_per_atom_ev")
    assert torch.equal(
        pack_structure_batch(
            packed.element_tokens,
            packed.fractional_coordinates,
            packed.lattice,
            packed.batch,
        ).node_counts,
        packed.node_counts,
    )


def test_structure_replay_loss_is_finite_and_backpropagates() -> None:
    diffusion = _diffusion()
    packed = collate_structure_records([_record("audit-only", "pbesol", 3)])
    loss = structure_replay_loss(
        diffusion,
        packed,
        generator=torch.Generator().manual_seed(7),
        precision="fp32",
    )
    loss.backward()
    gradients = [parameter.grad for parameter in diffusion.parameters() if parameter.requires_grad]
    assert torch.isfinite(loss)
    assert any(gradient is not None and bool((gradient != 0.0).any()) for gradient in gradients)
    assert all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_three_stream_objective_accumulates_all_losses_before_one_update() -> None:
    diffusion = _diffusion()
    model = PhysicalRepresentationModel(
        diffusion.denoiser,
        teacher_dim=3,
        functional_count=3,
    )
    trainer = PhysicalTransferTrainer(
        model,
        diffusion,
        PhysicalTransferTrainingConfig(precision="fp32"),
    )
    records = [_record("physical-audit", "pbe", 3)]
    physical = collate_matpes_records(
        records,
        functional_vocabulary={"pbe": 0, "pbesol": 1, "scan": 2},
        teacher_dim=3,
    )
    normalizer = FunctionalPhysicalNormalizer(
        energy_location=torch.zeros(3),
        energy_scale=torch.ones(3),
        force_scale=torch.ones(3),
        stress_isotropic_location=torch.zeros(3),
        stress_scale=torch.ones(3),
    )
    structure = collate_structure_records(records)
    losses = accumulate_continued_pretraining_step(
        trainer,
        structure,
        physical,
        structure,
        normalizer,
        ContinuedPretrainingWeights(0.4, 0.3, 0.3),
        lemat_rank_fraction=1.0,
        alex_rank_fraction=1.0,
        generator=torch.Generator().manual_seed(19),
    )
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert torch.isfinite(losses.lemat_structure)
    assert torch.isfinite(losses.matpes_physical.loss)
    assert torch.isfinite(losses.alex_structure)
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert float(trainer.finish_optimization_step()) > 0.0
