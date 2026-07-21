from pathlib import Path

import torch

from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.physical_checkpointing import (
    load_physical_checkpoint,
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
    save_physical_checkpoint,
)
from gaugeflow.production.physical_pretraining import PhysicalRepresentationModel
from gaugeflow.production.physical_training import (
    PhysicalTransferTrainer,
    PhysicalTransferTrainingConfig,
)
from gaugeflow.production.training import ExponentialMovingAverage


def _objects() -> tuple[PhysicalRepresentationModel, PhysicalTransferTrainer]:
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
    diffusion = TensorFreeHybridDiffusion(backbone, standardizer)
    return model, PhysicalTransferTrainer(model, diffusion, PhysicalTransferTrainingConfig())


def test_physical_checkpoint_round_trip_and_hash_validation(tmp_path: Path) -> None:
    torch.manual_seed(12)
    model, trainer = _objects()
    path = tmp_path / "physical.pt"
    runtime = [{"rank": 0, "generator": torch.arange(4, dtype=torch.uint8)}]
    metadata = {"protocol": "test", "seed": 12}
    save_physical_checkpoint(
        path,
        model=model,
        trainer=trainer,
        rank_runtime_states=runtime,
        metadata=metadata,
    )
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    restored_runtime, restored_metadata = load_physical_checkpoint(
        path,
        model=model,
        trainer=trainer,
        map_location="cpu",
    )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
    assert torch.equal(restored_runtime[0]["generator"], runtime[0]["generator"])
    assert restored_metadata == metadata == read_physical_checkpoint_metadata(path)

    evaluation_model, _ = _objects()
    evaluation_ema = ExponentialMovingAverage(evaluation_model, 0.999)
    step, evaluation_metadata = load_physical_ema_for_evaluation(
        path,
        model=evaluation_model,
        ema=evaluation_ema,
        map_location="cpu",
    )
    assert step == 0 and evaluation_metadata == metadata
    assert all(
        torch.equal(value, evaluation_model.state_dict()[name])
        for name, value in original.items()
    )
