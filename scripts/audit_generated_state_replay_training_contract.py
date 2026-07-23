"""Audit that generated-state replay entries enter the product loss correctly.

This is a correctness audit, not a training run and not a quality Gate.  It
loads a replay cache, reconstructs the frozen tensor-free backbone, evaluates
the current hybrid denoising objective role by role, and verifies that active
terminal branches receive finite nonzero gradients.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import load_production_checkpoint, read_production_checkpoint_metadata
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.generated_state_replay import (
    GeneratedCarrierRole,
    GeneratedStateReplayEntry,
    load_generated_state_replay_cache,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.training import ExponentialMovingAverage

_ROLES: tuple[GeneratedCarrierRole, ...] = (
    "clean_clean",
    "generated_assignment",
    "generated_lattice",
    "generated_joint",
)


@dataclass(frozen=True)
class PackedReplayRole:
    role: GeneratedCarrierRole
    source_structure_ids: tuple[str, ...]
    node_counts: torch.Tensor
    composition_counts: torch.Tensor
    assignment_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    batch: torch.Tensor
    blueprint: ParentBlueprintBatch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--expected-sampler-commit", default=None)
    parser.add_argument("--expected-sampler-protocol-sha256", default=None)
    parser.add_argument(
        "--forbidden-source-ids",
        type=Path,
        default=None,
        help="Optional JSON list or newline-delimited source IDs that must not appear in the replay cache.",
    )
    parser.add_argument(
        "--clean-retention-max-ratio",
        type=float,
        default=10.0,
        help="Fail if clean_clean total loss exceeds this multiple of the largest generated-role total loss.",
    )
    parser.add_argument(
        "--max-graphs-per-role-batch",
        type=int,
        default=0,
        help="If positive, split each role into graph chunks of this size and accumulate graph-weighted losses.",
    )
    return parser.parse_args()


def _read_forbidden_source_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        value: Any = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("forbidden source ID JSON must be a list of strings")
        return set(value)
    return {line.strip() for line in text.splitlines() if line.strip()}


def _load_diffusion(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[TensorFreeHybridDiffusion, dict[str, Any], dict[str, Any]]:
    try:
        metadata = read_production_checkpoint_metadata(checkpoint)
    except ValueError:
        metadata = read_physical_checkpoint_metadata(checkpoint)
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    standardization = metadata.get("lattice_standardization")
    if isinstance(model_config, dict) and isinstance(training_config, dict) and isinstance(standardization, dict):
        denoiser = HybridCrystalDenoiser(**model_config).to(device)
        ema = ExponentialMovingAverage(denoiser, float(training_config["ema_decay"]))
        load_production_checkpoint(checkpoint, model=denoiser, ema=ema, map_location=device)
        ema.copy_to(denoiser)
        diffusion = TensorFreeHybridDiffusion(
            denoiser,
            P1LatticeStandardizer.from_mapping(standardization),
            coordinate_sigma_min=float(training_config["coordinate_sigma_min"]),
            coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
            minimum_time=float(training_config["minimum_time"]),
            maximum_time=float(training_config["maximum_time"]),
            categorical_path=str(training_config["categorical_path"]),
            composition_conditioning=bool(training_config["composition_conditioning"]),
        )
        return diffusion, dict(training_config), metadata
    stage_b_metadata = metadata.get("stage_b_metadata")
    if not isinstance(stage_b_metadata, dict):
        raise ValueError("checkpoint metadata does not describe a production or Stage-C backbone")
    physical_config = stage_b_metadata.get("physical_training_config")
    if not isinstance(physical_config, dict):
        raise ValueError("Stage-C checkpoint metadata lacks physical training config")
    objects = build_continued_pretraining_objects(
        stage_b_metadata,
        device=device,
        optimizer_owner=False,
    )
    ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
    load_physical_ema_for_evaluation(checkpoint, model=objects.model, ema=ema, map_location=device)
    training = stage_b_metadata.get("a1_training_config")
    if not isinstance(training, dict):
        raise ValueError("Stage-C checkpoint metadata lacks A1 training config")
    return objects.diffusion, dict(training), metadata


def _pack_role_entries(
    role: GeneratedCarrierRole,
    entries: list[GeneratedStateReplayEntry],
    *,
    device: torch.device,
) -> PackedReplayRole:
    if not entries:
        raise ValueError(f"replay cache has no entries for role {role}")
    node_counts = torch.cat([entry.node_count.to(dtype=torch.long) for entry in entries], dim=0)
    batch = torch.repeat_interleave(torch.arange(node_counts.numel(), dtype=torch.long), node_counts)
    assignment_tokens = torch.cat([entry.assignment_tokens.to(dtype=torch.long) for entry in entries], dim=0)
    if bool((assignment_tokens >= 118).any()):
        raise ValueError(f"role {role} contains masked or non-chemical endpoint assignment tokens")
    composition_counts = torch.cat([entry.composition_counts.to(dtype=torch.long) for entry in entries], dim=0)
    observed = torch.bincount(
        batch * 118 + assignment_tokens,
        minlength=node_counts.numel() * 118,
    ).reshape(node_counts.numel(), 118)
    if not torch.equal(observed, composition_counts):
        raise ValueError(f"role {role} assignment tokens do not realize declared composition counts")
    lattice = torch.cat([entry.lattice.to(dtype=torch.float32) for entry in entries], dim=0)
    fractional = torch.cat([entry.fractional_coordinates.to(dtype=torch.float32) for entry in entries], dim=0)
    blueprint = ParentBlueprintBatch.from_node_counts(
        node_counts,
        dtype=lattice.dtype,
        device=device,
    )
    packed = PackedReplayRole(
        role=role,
        source_structure_ids=tuple(entry.key.source_structure_id for entry in entries),
        node_counts=node_counts.to(device=device),
        composition_counts=composition_counts.to(device=device),
        assignment_tokens=assignment_tokens.to(device=device),
        fractional_coordinates=fractional.to(device=device),
        lattice=lattice.to(device=device),
        batch=blueprint.batch,
        blueprint=blueprint,
    )
    if not torch.equal(packed.batch.cpu(), batch):
        raise AssertionError("packed replay batch changed graph membership")
    return packed


def _slice_packed_role(packed: PackedReplayRole, start: int, stop: int) -> PackedReplayRole:
    graph_count = int(packed.node_counts.numel())
    if start < 0 or stop <= start or stop > graph_count:
        raise ValueError(f"invalid graph slice {start}:{stop} for {graph_count} graphs")
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=packed.node_counts.device),
            torch.cumsum(packed.node_counts, dim=0),
        )
    )
    node_start = int(offsets[start].detach().cpu())
    node_stop = int(offsets[stop].detach().cpu())
    node_counts = packed.node_counts[start:stop]
    blueprint = ParentBlueprintBatch.from_node_counts(
        node_counts.detach().cpu(),
        dtype=packed.lattice.dtype,
        device=packed.lattice.device,
    )
    return PackedReplayRole(
        role=packed.role,
        source_structure_ids=packed.source_structure_ids[start:stop],
        node_counts=node_counts,
        composition_counts=packed.composition_counts[start:stop],
        assignment_tokens=packed.assignment_tokens[node_start:node_stop],
        fractional_coordinates=packed.fractional_coordinates[node_start:node_stop],
        lattice=packed.lattice[start:stop],
        batch=blueprint.batch,
        blueprint=blueprint,
    )


def _iter_role_chunks(packed: PackedReplayRole, max_graphs_per_batch: int) -> list[PackedReplayRole]:
    graph_count = int(packed.node_counts.numel())
    if max_graphs_per_batch <= 0 or max_graphs_per_batch >= graph_count:
        return [packed]
    return [
        _slice_packed_role(packed, start, min(start + max_graphs_per_batch, graph_count))
        for start in range(0, graph_count, max_graphs_per_batch)
    ]


def _parameter_group(name: str) -> str | None:
    if name.startswith(("element_embedding.", "element_head.", "composition_head.")):
        return "element"
    if name.startswith(("volume_head.", "shape_head.", "lattice_residual_adapter.")):
        return "lattice"
    if name.startswith(
        (
            "coordinate_control_gate.",
            "coordinate_edge_encoder.",
            "coordinate_edge_residual.",
            "coordinate_carrier.",
            "coordinate_carrier_mixer.",
        )
    ):
        return "coordinate"
    return None


def _gradient_group_norms(model: torch.nn.Module) -> dict[str, float]:
    squared = {"element": 0.0, "lattice": 0.0, "coordinate": 0.0}
    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        if group is None or parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        squared[group] += float((grad * grad).sum().cpu())
    return {key: value**0.5 for key, value in squared.items()}


def _finite_float(value: torch.Tensor) -> float:
    if not torch.isfinite(value.detach()).all():
        raise FloatingPointError("audit metric is non-finite")
    return float(value.detach().cpu())


def _check_composition_counts(packed: PackedReplayRole, output: Any) -> None:
    if output.noisy.composition_counts is None:
        raise RuntimeError("composition-conditioned replay audit did not pass composition counts")
    if not torch.equal(output.noisy.composition_counts.detach().cpu(), packed.composition_counts.detach().cpu()):
        raise RuntimeError(f"role {packed.role} used composition counts different from replay entry")


def _audit_role(
    diffusion: TensorFreeHybridDiffusion,
    packed: PackedReplayRole,
    *,
    seed: int,
    loss_weight: float,
    precision: str,
    max_graphs_per_batch: int,
) -> dict[str, Any]:
    diffusion.train()
    diffusion.denoiser.zero_grad(set_to_none=True)
    generator = torch.Generator(device=packed.lattice.device).manual_seed(seed)
    use_bf16 = precision == "bf16" and packed.lattice.device.type == "cuda"
    metric_sums = {
        "loss": 0.0,
        "element_loss": 0.0,
        "composition_loss": 0.0,
        "coordinate_loss": 0.0,
        "volume_loss": 0.0,
        "shape_loss": 0.0,
        "masked_fraction": 0.0,
    }
    weighted_loss = 0.0
    total_graphs = int(packed.node_counts.numel())
    for chunk in _iter_role_chunks(packed, max_graphs_per_batch):
        chunk_graphs = int(chunk.node_counts.numel())
        chunk_fraction = chunk_graphs / total_graphs
        with torch.autocast(
            device_type=packed.lattice.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = diffusion(
                chunk.assignment_tokens,
                chunk.fractional_coordinates,
                chunk.lattice,
                chunk.batch,
                chunk.blueprint.shape_projector,
                chunk.blueprint.fractional_to_cartesian,
                generator=generator,
            )
            optimization_loss = output.loss * loss_weight * chunk_fraction
        _check_composition_counts(chunk, output)
        if not torch.isfinite(optimization_loss.detach()):
            raise FloatingPointError(f"role {packed.role} optimization loss is non-finite")
        optimization_loss.backward()
        weighted_loss += _finite_float(optimization_loss)
        metric_sums["loss"] += _finite_float(output.loss) * chunk_graphs
        metric_sums["element_loss"] += _finite_float(output.element_loss) * chunk_graphs
        metric_sums["composition_loss"] += _finite_float(output.composition_loss) * chunk_graphs
        metric_sums["coordinate_loss"] += _finite_float(output.coordinate_loss) * chunk_graphs
        metric_sums["volume_loss"] += _finite_float(output.volume_loss) * chunk_graphs
        metric_sums["shape_loss"] += _finite_float(output.shape_loss) * chunk_graphs
        metric_sums["masked_fraction"] += _finite_float(output.masked_fraction) * chunk_graphs
    gradient_norms = _gradient_group_norms(diffusion.denoiser)
    nonzero = {key: value > 0.0 for key, value in gradient_norms.items()}
    return {
        "role": packed.role,
        "source_structure_ids": list(packed.source_structure_ids),
        "graph_count": total_graphs,
        "node_count": int(packed.node_counts.sum().detach().cpu()),
        "loss": metric_sums["loss"] / total_graphs,
        "element_loss": metric_sums["element_loss"] / total_graphs,
        "composition_loss": metric_sums["composition_loss"] / total_graphs,
        "coordinate_loss": metric_sums["coordinate_loss"] / total_graphs,
        "volume_loss": metric_sums["volume_loss"] / total_graphs,
        "shape_loss": metric_sums["shape_loss"] / total_graphs,
        "masked_fraction": metric_sums["masked_fraction"] / total_graphs,
        "loss_weight": float(loss_weight),
        "weighted_loss": weighted_loss,
        "gradient_norms": gradient_norms,
        "nonzero_gradient_groups": nonzero,
        "max_graphs_per_role_batch": int(max_graphs_per_batch),
        "composition_count_sums": [int(value) for value in packed.composition_counts.sum(dim=1).detach().cpu()],
    }


def main() -> None:
    args = _parse_args()
    if args.loss_weight <= 0.0:
        raise ValueError("loss weight must be positive")
    if args.clean_retention_max_ratio <= 0.0:
        raise ValueError("clean retention max ratio must be positive")
    if args.max_graphs_per_role_batch < 0:
        raise ValueError("max graphs per role batch must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    expected_base_sha = sha256_file(args.base_checkpoint)
    forbidden_source_ids = _read_forbidden_source_ids(args.forbidden_source_ids)
    entries, manifest = load_generated_state_replay_cache(
        args.cache_dir,
        expected_base_checkpoint_sha256=expected_base_sha,
        expected_sampler_commit=args.expected_sampler_commit,
        expected_sampler_protocol_sha256=args.expected_sampler_protocol_sha256,
        forbidden_source_ids=forbidden_source_ids,
    )
    diffusion, training_config, checkpoint_metadata = _load_diffusion(args.base_checkpoint, device=device)
    if training_config.get("categorical_path") != "orderless_reveal" or not bool(
        training_config.get("composition_conditioning")
    ):
        raise ValueError("replay training audit requires orderless composition-conditioned product training")
    precision = str(training_config.get("precision", "fp32"))
    by_role: dict[GeneratedCarrierRole, list[GeneratedStateReplayEntry]] = {
        role: [entry for entry in entries if entry.key.role == role] for role in _ROLES
    }
    role_reports = []
    for role_index, role in enumerate(_ROLES):
        packed = _pack_role_entries(role, by_role[role], device=device)
        role_reports.append(
            _audit_role(
                diffusion,
                packed,
                seed=args.seed + 1009 * role_index,
                loss_weight=args.loss_weight,
                precision=precision,
                max_graphs_per_batch=args.max_graphs_per_role_batch,
            )
        )
    losses = {str(report["role"]): float(report["loss"]) for report in role_reports}
    generated_losses = [losses[role] for role in _ROLES if role != "clean_clean"]
    clean_ratio = losses["clean_clean"] / max(max(generated_losses), 1.0e-12)
    clean_retention_not_exploded = clean_ratio <= args.clean_retention_max_ratio
    all_gradients_nonzero = all(
        all(bool(value) for value in report["nonzero_gradient_groups"].values()) for report in role_reports
    )
    status = "passed" if clean_retention_not_exploded and all_gradients_nonzero else "failed"
    report = {
        "status": status,
        "cache_dir": str(args.cache_dir),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": expected_base_sha,
        "manifest_sha256": manifest.canonical_sha256(),
        "entry_count": len(entries),
        "roles": role_reports,
        "precision": precision,
        "training_categorical_path": training_config.get("categorical_path"),
        "training_composition_conditioning": bool(training_config.get("composition_conditioning")),
        "clean_retention_loss_ratio_to_max_generated": clean_ratio,
        "clean_retention_max_ratio": args.clean_retention_max_ratio,
        "clean_retention_not_exploded": clean_retention_not_exploded,
        "all_role_terminal_gradient_groups_nonzero": all_gradients_nonzero,
        "max_graphs_per_role_batch": int(args.max_graphs_per_role_batch),
        "forbidden_source_id_check": {
            "executed": forbidden_source_ids is not None,
            "count": 0 if forbidden_source_ids is None else len(forbidden_source_ids),
        },
        "checkpoint_protocol": checkpoint_metadata.get("protocol", "unknown"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "passed":
        raise RuntimeError("generated-state replay training contract audit failed")


if __name__ == "__main__":
    main()
