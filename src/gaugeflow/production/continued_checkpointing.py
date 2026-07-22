"""Reconstruct the exact Stage-B state used to start Stage-C continuation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .equivariant_denoiser import HybridCrystalDenoiser
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .lattice_standardization import P1LatticeStandardizer
from .physical_checkpointing import load_physical_checkpoint, read_physical_checkpoint_metadata
from .physical_pretraining import PhysicalRepresentationModel
from .physical_training import PhysicalTransferTrainer, PhysicalTransferTrainingConfig


@dataclass(frozen=True)
class ContinuedPretrainingObjects:
    model: PhysicalRepresentationModel
    diffusion: TensorFreeHybridDiffusion
    trainer: PhysicalTransferTrainer
    functional_vocabulary: dict[str, int]
    metadata: dict[str, Any]


def _mapping(metadata: dict[str, Any], name: str) -> dict[str, Any]:
    value = metadata.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Stage-B checkpoint metadata lacks {name}")
    return dict(value)


def build_continued_pretraining_objects(
    metadata: dict[str, Any],
    *,
    device: torch.device,
    optimizer_owner: bool,
) -> ContinuedPretrainingObjects:
    """Build the exact model/optimizer schema declared by a Stage-B checkpoint."""

    if metadata.get("protocol") != "stage_b_physical_representation_v1_1":
        raise ValueError("continued pretraining requires a Stage-B checkpoint")
    model_config = _mapping(metadata, "model_config")
    a1_training = _mapping(metadata, "a1_training_config")
    standardization = _mapping(metadata, "lattice_standardization")
    physical_config = _mapping(metadata, "physical_training_config")
    vocabulary_value = _mapping(metadata, "functional_vocabulary")
    vocabulary = {str(key): int(value) for key, value in vocabulary_value.items()}
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise ValueError("Stage-B functional vocabulary is not contiguous")
    teacher_dim = metadata.get("teacher_feature_dim")
    if isinstance(teacher_dim, bool) or not isinstance(teacher_dim, int) or teacher_dim < 1:
        raise ValueError("Stage-B teacher feature dimension is invalid")
    backbone = HybridCrystalDenoiser(**model_config).to(device)
    model = PhysicalRepresentationModel(
        backbone,
        teacher_dim=teacher_dim,
        functional_count=len(vocabulary),
    ).to(device)
    diffusion = TensorFreeHybridDiffusion(
        backbone,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(a1_training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(a1_training["coordinate_sigma_max"]),
        minimum_time=float(a1_training["minimum_time"]),
        maximum_time=float(a1_training["maximum_time"]),
        categorical_path=str(a1_training["categorical_path"]),
        composition_conditioning=bool(a1_training["composition_conditioning"]),
    )
    config = PhysicalTransferTrainingConfig(**physical_config)
    trainer = PhysicalTransferTrainer(
        model,
        diffusion,
        config,
        optimizer_owner=optimizer_owner,
    )
    return ContinuedPretrainingObjects(model, diffusion, trainer, vocabulary, metadata)


def load_stage_b_continuation_start(
    checkpoint: Path,
    *,
    device: torch.device,
) -> ContinuedPretrainingObjects:
    """Restore all Stage-B learned state on the sole optimizer-owning rank."""

    metadata = read_physical_checkpoint_metadata(checkpoint)
    objects = build_continued_pretraining_objects(
        metadata,
        device=device,
        optimizer_owner=True,
    )
    _, loaded_metadata = load_physical_checkpoint(
        checkpoint,
        model=objects.model,
        trainer=objects.trainer,
        map_location=device,
    )
    if loaded_metadata != metadata:
        raise AssertionError("Stage-B checkpoint metadata changed while loading")
    return objects
