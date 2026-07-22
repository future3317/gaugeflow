from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from gaugeflow.production.continued_checkpointing import (
    build_continued_pretraining_objects,
    load_stage_b_continuation_start,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.physical_checkpointing import save_physical_checkpoint
from gaugeflow.production.physical_pretraining import PhysicalRepresentationModel
from gaugeflow.production.physical_training import (
    PhysicalTransferTrainer,
    PhysicalTransferTrainingConfig,
)


def _metadata(model_config: dict[str, object], standardizer: P1LatticeStandardizer) -> dict:
    return {
        "protocol": "stage_b_physical_representation_v1_1",
        "model_config": model_config,
        "a1_training_config": {
            "coordinate_sigma_min": 0.01,
            "coordinate_sigma_max": 1.0,
            "minimum_time": 1.0e-3,
            "maximum_time": 0.999,
            "categorical_path": "absorbing_mask",
            "composition_conditioning": False,
        },
        "lattice_standardization": standardizer.as_mapping(),
        "physical_training_config": dataclasses.asdict(
            PhysicalTransferTrainingConfig(precision="fp32")
        ),
        "functional_vocabulary": {"PBE": 0, "r2SCAN": 1},
        "teacher_feature_dim": 3,
    }


def _checkpoint(tmp_path: Path) -> tuple[Path, dict, dict[str, torch.Tensor]]:
    model_config: dict[str, object] = {
        "hidden_dim": 16,
        "vector_dim": 4,
        "layers": 1,
        "radial_dim": 4,
        "radial_cutoff": 4.0,
        "edge_dim": 8,
        "angular_channels": 2,
        "edge_refresh_rank": 4,
    }
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
    standardizer = P1LatticeStandardizer(
        volume_residual_mean=0.0,
        volume_residual_std=1.0,
        shape_mean=torch.zeros(6, dtype=torch.float64),
        shape_basis_columns=torch.linalg.qr(raw_basis, mode="reduced").Q,
        shape_scales=torch.ones(5, dtype=torch.float64),
    )
    backbone = HybridCrystalDenoiser(**model_config)
    model = PhysicalRepresentationModel(backbone, teacher_dim=3, functional_count=2)
    trainer = PhysicalTransferTrainer(
        model,
        TensorFreeHybridDiffusion(backbone, standardizer),
        PhysicalTransferTrainingConfig(precision="fp32"),
    )
    metadata = _metadata(model_config, standardizer)
    checkpoint = tmp_path / "stage_b.pt"
    save_physical_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        rank_runtime_states=[{"rank": 0}],
        metadata=metadata,
    )
    return checkpoint, metadata, {name: value.clone() for name, value in model.state_dict().items()}


def test_continued_objects_preserve_stage_b_schema_and_owner_policy(tmp_path: Path) -> None:
    checkpoint, metadata, expected = _checkpoint(tmp_path)
    restored = load_stage_b_continuation_start(checkpoint, device=torch.device("cpu"))
    assert restored.metadata == metadata
    assert restored.functional_vocabulary == {"PBE": 0, "r2SCAN": 1}
    assert restored.trainer.optimizer is not None and restored.trainer.ema is not None
    assert all(
        torch.equal(value, restored.model.state_dict()[name])
        for name, value in expected.items()
    )
    replica = build_continued_pretraining_objects(
        metadata,
        device=torch.device("cpu"),
        optimizer_owner=False,
    )
    assert replica.trainer.optimizer is None and replica.trainer.ema is None
