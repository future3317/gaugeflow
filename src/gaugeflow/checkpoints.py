"""Safe tensor-only checkpoints with a separate JSON configuration sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .file_utils import canonical_json_hash, sha256_file

SAFE_CHECKPOINT_SCHEMA = 2


def json_safe(value: Any) -> Any:
    """Convert configuration values to the JSON-only public checkpoint schema."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint metadata is not JSON-safe: {type(value).__name__}")


def save_safe_checkpoint(
    path: Path,
    *,
    model_state: Mapping[str, torch.Tensor],
    isotypic_scales: torch.Tensor,
    training_step: int,
    metadata: Mapping[str, Any],
) -> Path:
    """Write tensor weights and JSON metadata as two independently checked files."""
    if training_step < 0 or not torch.isfinite(isotypic_scales).all():
        raise ValueError("checkpoint step and scales must be valid")
    if not model_state or any(not isinstance(value, torch.Tensor) for value in model_state.values()):
        raise ValueError("model_state must be a non-empty tensor mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_payload = {
        "schema": SAFE_CHECKPOINT_SCHEMA,
        "model": dict(model_state),
        "isotypic_scales": isotypic_scales.detach().cpu(),
        "training_step": int(training_step),
    }
    torch.save(tensor_payload, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    json_metadata = json_safe(metadata)
    payload = {
        "schema": SAFE_CHECKPOINT_SCHEMA,
        "weights_file": path.name,
        "weights_sha256": sha256_file(path),
        "metadata": json_metadata,
        "metadata_sha256": canonical_json_hash(json_metadata),
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def load_safe_checkpoint(path: Path, *, map_location: str | torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only the registered tensor schema; arbitrary pickle globals stay disabled."""
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("safe checkpoint requires both weights and JSON sidecar")
    metadata_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata_payload.get("schema") != SAFE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint metadata schema")
    if metadata_payload.get("weights_file") != path.name:
        raise ValueError("checkpoint sidecar names a different weights file")
    if metadata_payload.get("weights_sha256") != sha256_file(path):
        raise ValueError("checkpoint weights hash mismatch")
    metadata = metadata_payload.get("metadata")
    if not isinstance(metadata, dict) or metadata_payload.get("metadata_sha256") != canonical_json_hash(metadata):
        raise ValueError("checkpoint metadata hash mismatch")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SAFE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported tensor checkpoint schema")
    model = payload.get("model")
    scales = payload.get("isotypic_scales")
    if not isinstance(model, dict) or not model or any(not isinstance(value, torch.Tensor) for value in model.values()):
        raise ValueError("checkpoint model state is not a tensor mapping")
    if not isinstance(scales, torch.Tensor) or not torch.isfinite(scales).all():
        raise ValueError("checkpoint isotypic scales are invalid")
    return payload, metadata
