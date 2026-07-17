"""Construction of a hash-bound tensor-free EMA inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .blueprint import EmpiricalNodeCountPrior
from .checkpointing import load_production_checkpoint, read_production_checkpoint_metadata
from .equivariant_denoiser import HybridCrystalDenoiser
from .lattice_standardization import P1LatticeStandardizer
from .training import ExponentialMovingAverage


@dataclass(frozen=True)
class TensorFreeEmaRuntime:
    model: HybridCrystalDenoiser
    lattice_standardizer: P1LatticeStandardizer
    training_config: dict[str, Any]
    node_count_prior: EmpiricalNodeCountPrior


def load_tensor_free_ema_runtime(
    checkpoint: Path,
    device: torch.device,
    *,
    protocol_name: str,
    protocol_sha256: str,
) -> TensorFreeEmaRuntime:
    """Load one EMA checkpoint only when its complete protocol identity matches."""
    metadata = read_production_checkpoint_metadata(checkpoint)
    if (
        metadata.get("protocol") != protocol_name
        or metadata.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("checkpoint does not match the frozen H1a P1 protocol")
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    standardization = metadata.get("lattice_standardization")
    if (
        not isinstance(model_config, dict)
        or not isinstance(training_config, dict)
        or not isinstance(standardization, dict)
    ):
        raise ValueError("tensor-free checkpoint metadata is incomplete")
    model = HybridCrystalDenoiser(**model_config).to(device)
    ema = ExponentialMovingAverage(model, float(training_config["ema_decay"]))
    _, node_prior, _ = load_production_checkpoint(
        checkpoint, model=model, ema=ema, map_location=device
    )
    ema.copy_to(model)
    model.eval()
    return TensorFreeEmaRuntime(
        model=model,
        lattice_standardizer=P1LatticeStandardizer.from_mapping(standardization),
        training_config=training_config,
        node_count_prior=node_prior,
    )
