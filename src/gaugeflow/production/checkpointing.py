"""Recoverable tensor-only checkpoints for production training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from gaugeflow.file_utils import canonical_json_hash, sha256_file

from .blueprint import EmpiricalNodeCountPrior
from .training import ExponentialMovingAverage

PRODUCTION_CHECKPOINT_SCHEMA = 1


def read_production_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Verify a checkpoint pair and return its JSON-only configuration."""
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("production checkpoint requires weights and JSON sidecar")
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = description.get("metadata")
    if (
        description.get("schema") != PRODUCTION_CHECKPOINT_SCHEMA
        or description.get("weights_file") != path.name
        or description.get("weights_sha256") != sha256_file(path)
        or not isinstance(metadata, dict)
        or description.get("metadata_sha256") != canonical_json_hash(metadata)
    ):
        raise ValueError("production checkpoint sidecar failed schema/hash validation")
    return metadata


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint value is not tensor/JSON safe: {type(value).__name__}")


def save_production_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    training_step: int,
    node_count_prior: EmpiricalNodeCountPrior,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically save model, EMA, optimizer and CPU/CUDA RNG state."""
    if training_step < 0:
        raise ValueError("training step must be nonnegative")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema": PRODUCTION_CHECKPOINT_SCHEMA,
        "model": _cpu_tree(model.state_dict()),
        "ema": _cpu_tree(ema.state_dict()),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "training_step": int(training_step),
        "node_count_prior": _cpu_tree(node_count_prior.state_dict()),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    json_metadata = dict(metadata)
    sidecar_payload = {
        "schema": PRODUCTION_CHECKPOINT_SCHEMA,
        "weights_file": path.name,
        "weights_sha256": sha256_file(path),
        "metadata": json_metadata,
        "metadata_sha256": canonical_json_hash(json_metadata),
    }
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar_temporary.replace(sidecar)
    return sidecar


def load_production_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> tuple[int, EmpiricalNodeCountPrior, dict[str, Any]]:
    metadata = read_production_checkpoint_metadata(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != PRODUCTION_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported production checkpoint schema")
    model.load_state_dict(payload["model"], strict=True)
    if ema is not None:
        ema.load_state_dict(payload["ema"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        torch.set_rng_state(payload["cpu_rng_state"].cpu())
        cuda_state = payload["cuda_rng_state"]
        if torch.cuda.is_available() and cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
    prior = EmpiricalNodeCountPrior.from_state_dict(payload["node_count_prior"])
    return int(payload["training_step"]), prior, metadata
