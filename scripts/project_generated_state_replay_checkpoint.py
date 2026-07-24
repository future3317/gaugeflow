"""Project a generated-state replay checkpoint onto predeclared parameter blocks.

This is a zero-training trust projection diagnostic.  It writes a checkpoint
with the same schema as generated-state replay correctness training so existing
evaluators can consume it, but it does not run optimization and does not
qualify Stage-E.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

try:
    from evaluate_generated_state_replay_correctness import _load_stage_c_base_backbone
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as scripts.*
    from scripts.evaluate_generated_state_replay_correctness import _load_stage_c_base_backbone

from gaugeflow.file_utils import sha256_file

ELEMENT_PREFIXES = ("element_embedding.", "element_head.", "composition_head.")
COORDINATE_PREFIXES = (
    "coordinate_control_gate.",
    "coordinate_edge_encoder.",
    "coordinate_edge_residual.",
    "coordinate_carrier.",
    "coordinate_carrier_mixer.",
)
LATTICE_HEAD_PREFIXES = ("volume_head.", "shape_head.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--element-alpha", type=float, default=0.0)
    parser.add_argument("--coordinate-alpha", type=float, default=1.0)
    parser.add_argument("--lattice-head-alpha", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _alpha_for_key(
    key: str,
    *,
    element_alpha: float,
    coordinate_alpha: float,
    lattice_head_alpha: float,
) -> float:
    if key.startswith(ELEMENT_PREFIXES):
        return element_alpha
    if key.startswith(COORDINATE_PREFIXES):
        return coordinate_alpha
    if key.startswith(LATTICE_HEAD_PREFIXES):
        return lattice_head_alpha
    return 0.0


def _validate_alpha(name: str, value: float) -> None:
    if not torch.isfinite(torch.tensor(value)).item():
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0,1]")


def _project_state_dict(
    base_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    *,
    element_alpha: float,
    coordinate_alpha: float,
    lattice_head_alpha: float,
) -> dict[str, torch.Tensor]:
    if set(base_state) != set(candidate_state):
        raise ValueError("base and candidate state dictionaries have different keys")
    projected: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        candidate_value = candidate_state[key]
        if base_value.shape != candidate_value.shape:
            raise ValueError(f"state shape mismatch for {key}")
        alpha = _alpha_for_key(
            key,
            element_alpha=element_alpha,
            coordinate_alpha=coordinate_alpha,
            lattice_head_alpha=lattice_head_alpha,
        )
        if base_value.is_floating_point():
            projected[key] = base_value + alpha * (candidate_value.to(base_value.dtype) - base_value)
        else:
            projected[key] = candidate_value.clone() if alpha == 1.0 else base_value.clone()
    return projected


def _parameter_update_norm(
    base_state: dict[str, torch.Tensor],
    projected_state: dict[str, torch.Tensor],
) -> float:
    squared = 0.0
    for key, base_value in base_state.items():
        projected_value = projected_state[key]
        if base_value.is_floating_point():
            delta = projected_value.float() - base_value.float()
            squared += float((delta * delta).sum())
    return squared**0.5


def _load_base_state(checkpoint: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    base_backbone, _, _ = _load_stage_c_base_backbone(checkpoint, device=device)
    return {key: value.detach().cpu().clone() for key, value in base_backbone.state_dict().items()}


def _read_candidate_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "gaugeflow.generated_state_replay_correctness_training.v1"
    ):
        raise ValueError("candidate is not a generated-state replay correctness checkpoint")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("candidate checkpoint lacks model state")
    ema = payload.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
        raise ValueError("candidate checkpoint lacks EMA shadow state")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError("candidate checkpoint lacks summary")
    return payload


def project_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("element_alpha", "coordinate_alpha", "lattice_head_alpha"):
        _validate_alpha(name, float(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty output directory: {args.output_dir}")
    device = torch.device(args.device)
    base_state = _load_base_state(args.base_checkpoint, device=device)
    candidate = _read_candidate_checkpoint(args.candidate_checkpoint)
    model = _project_state_dict(
        base_state,
        candidate["model"],
        element_alpha=float(args.element_alpha),
        coordinate_alpha=float(args.coordinate_alpha),
        lattice_head_alpha=float(args.lattice_head_alpha),
    )
    ema_shadow = _project_state_dict(
        base_state,
        candidate["ema"]["shadow"],
        element_alpha=float(args.element_alpha),
        coordinate_alpha=float(args.coordinate_alpha),
        lattice_head_alpha=float(args.lattice_head_alpha),
    )
    summary = copy.deepcopy(candidate["summary"])
    projection = {
        "schema": "gaugeflow.generated_state_replay_parameter_block_projection.v1",
        "element_alpha": float(args.element_alpha),
        "coordinate_alpha": float(args.coordinate_alpha),
        "lattice_head_alpha": float(args.lattice_head_alpha),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint),
        "boundary": "zero-training trust projection; not a trained checkpoint or Stage-E pass",
    }
    summary.update(
        {
            "status": "diagnostic_parameter_block_projection",
            "parameter_block_projection": projection,
            "final_parameter_update_norm": _parameter_update_norm(base_state, model),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint_step_00000000.pt"
    payload = {
        "schema": "gaugeflow.generated_state_replay_correctness_training.v1",
        "model": model,
        "ema": {"decay": candidate["ema"]["decay"], "shadow": ema_shadow},
        "optimizer": candidate["optimizer"],
        "trainer_step": candidate["trainer_step"],
        "summary": summary,
    }
    torch.save(payload, checkpoint_path)
    summary["checkpoint"] = str(checkpoint_path)
    summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(project_checkpoint(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
