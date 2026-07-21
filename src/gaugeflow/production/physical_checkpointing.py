"""Hash-verified one-owner checkpoints for Stage-B physical transfer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gaugeflow.file_utils import canonical_json_hash, sha256_file

from .physical_pretraining import PhysicalRepresentationModel
from .physical_training import PhysicalTransferTrainer

PHYSICAL_CHECKPOINT_SCHEMA = 1


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
    raise TypeError(f"physical checkpoint value is not tensor/JSON safe: {type(value).__name__}")


def read_physical_checkpoint_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("physical checkpoint requires weights and JSON sidecar")
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = description.get("metadata")
    if (
        description.get("schema") != PHYSICAL_CHECKPOINT_SCHEMA
        or description.get("weights_file") != path.name
        or description.get("weights_sha256") != sha256_file(path)
        or not isinstance(metadata, dict)
        or description.get("metadata_sha256") != canonical_json_hash(metadata)
    ):
        raise ValueError("physical checkpoint sidecar failed schema/hash validation")
    return metadata


def save_physical_checkpoint(
    path: Path,
    *,
    model: PhysicalRepresentationModel,
    trainer: PhysicalTransferTrainer,
    rank_runtime_states: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    if trainer.optimizer is None or trainer.ema is None:
        raise RuntimeError("only the optimizer owner can save a physical checkpoint")
    if not rank_runtime_states:
        raise ValueError("physical checkpoint requires every rank runtime state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": PHYSICAL_CHECKPOINT_SCHEMA,
            "model": _cpu_tree(model.state_dict()),
            "trainer": _cpu_tree(trainer.state_dict()),
            "rank_runtime_states": _cpu_tree(list(rank_runtime_states)),
        },
        temporary,
    )
    temporary.replace(path)
    description = {
        "schema": PHYSICAL_CHECKPOINT_SCHEMA,
        "weights_file": path.name,
        "weights_sha256": sha256_file(path),
        "metadata": dict(metadata),
        "metadata_sha256": canonical_json_hash(metadata),
    }
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(
        json.dumps(description, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_temporary.replace(sidecar)
    return sidecar


def load_physical_checkpoint(
    path: Path,
    *,
    model: PhysicalRepresentationModel,
    trainer: PhysicalTransferTrainer,
    map_location: str | torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = read_physical_checkpoint_metadata(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != PHYSICAL_CHECKPOINT_SCHEMA:
        raise ValueError("physical checkpoint schema mismatch")
    model.load_state_dict(payload["model"], strict=True)
    trainer.load_state_dict(payload["trainer"])
    runtime = payload.get("rank_runtime_states")
    if not isinstance(runtime, list) or not runtime or not all(isinstance(item, dict) for item in runtime):
        raise ValueError("physical checkpoint rank runtime state is invalid")
    return runtime, metadata
