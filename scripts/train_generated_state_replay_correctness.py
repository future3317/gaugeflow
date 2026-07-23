"""Run a bounded generated-state replay correctness training smoke.

This is a training-interface correctness runner, not a generated-quality Gate.
It consumes a provenance-checked generated-state replay cache, initializes the
frozen Stage-C product backbone, accumulates one equal-weight microbatch per
carrier role, and verifies that optimizer steps really update the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from audit_generated_state_replay_training_contract import (
        _ROLES,
        PackedReplayRole,
        _finite_float,
        _gradient_group_norms,
        _check_composition_counts,
        _iter_role_chunks,
        _load_diffusion,
        _pack_role_entries,
        _parameter_group,
        _read_forbidden_source_ids,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as scripts.*
    from scripts.audit_generated_state_replay_training_contract import (
        _ROLES,
        PackedReplayRole,
        _finite_float,
        _gradient_group_norms,
        _check_composition_counts,
        _iter_role_chunks,
        _load_diffusion,
        _pack_role_entries,
        _parameter_group,
        _read_forbidden_source_ids,
    )

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.generated_state_replay import (
    GeneratedCarrierRole,
    GeneratedStateReplayEntry,
    load_generated_state_replay_cache,
)
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
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
        help="Fail if clean_clean loss exceeds this multiple of the largest generated-role loss.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Write a final training checkpoint under output-dir. Keep this under server /runs, not git.",
    )
    parser.add_argument(
        "--max-graphs-per-role-batch",
        type=int,
        default=0,
        help="If positive, split each role into graph chunks of this size and accumulate graph-weighted losses.",
    )
    return parser.parse_args()


def _training_config_from_checkpoint(
    training_config: dict[str, Any],
    *,
    learning_rate: float | None,
    weight_decay: float | None,
    gradient_clip_norm: float | None,
) -> ProductionTrainingConfig:
    config = ProductionTrainingConfig(
        learning_rate=float(training_config.get("learning_rate", ProductionTrainingConfig.learning_rate)),
        weight_decay=float(training_config.get("weight_decay", ProductionTrainingConfig.weight_decay)),
        gradient_clip_norm=float(
            training_config.get("gradient_clip_norm", ProductionTrainingConfig.gradient_clip_norm)
        ),
        ema_decay=float(training_config.get("ema_decay", ProductionTrainingConfig.ema_decay)),
        coordinate_sigma_min=float(
            training_config.get("coordinate_sigma_min", ProductionTrainingConfig.coordinate_sigma_min)
        ),
        coordinate_sigma_max=float(
            training_config.get("coordinate_sigma_max", ProductionTrainingConfig.coordinate_sigma_max)
        ),
        minimum_time=float(training_config.get("minimum_time", ProductionTrainingConfig.minimum_time)),
        maximum_time=float(training_config.get("maximum_time", ProductionTrainingConfig.maximum_time)),
        precision=str(training_config.get("precision", ProductionTrainingConfig.precision)),
        objective="joint",
        coordinate_clean_side_information=False,
        modality_time_mode=str(training_config.get("modality_time_mode", "shared")),
        categorical_path=str(training_config.get("categorical_path", "orderless_reveal")),
        composition_loss_weight=float(
            training_config.get("composition_loss_weight", ProductionTrainingConfig.composition_loss_weight)
        ),
        composition_conditioning=bool(training_config.get("composition_conditioning", True)),
    )
    if learning_rate is not None:
        config = ProductionTrainingConfig(**{**config.__dict__, "learning_rate": learning_rate})
    if weight_decay is not None:
        config = ProductionTrainingConfig(**{**config.__dict__, "weight_decay": weight_decay})
    if gradient_clip_norm is not None:
        config = ProductionTrainingConfig(**{**config.__dict__, "gradient_clip_norm": gradient_clip_norm})
    config.validate()
    if config.categorical_path != "orderless_reveal" or not config.composition_conditioning:
        raise ValueError("generated-state correctness training requires orderless composition conditioning")
    return config


def _grouped_entries(
    entries: list[GeneratedStateReplayEntry],
) -> dict[GeneratedCarrierRole, list[GeneratedStateReplayEntry]]:
    grouped = {role: [entry for entry in entries if entry.key.role == role] for role in _ROLES}
    missing = [role for role, role_entries in grouped.items() if not role_entries]
    if missing:
        raise ValueError(f"replay cache is missing carrier roles: {missing}")
    return grouped


def _role_weight(role_count: int) -> float:
    if role_count < 1:
        raise ValueError("role count must be positive")
    return 1.0 / role_count


def _clone_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _parameter_update_norm(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        previous = reference.get(name)
        if previous is None:
            raise ValueError(f"reference parameter snapshot lacks {name}")
        delta = parameter.detach().cpu().float() - previous.float()
        squared += float((delta * delta).sum())
    return squared**0.5


def _clone_current_group_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    gradients: dict[str, torch.Tensor | None] = {}
    for name, parameter in model.named_parameters():
        if _parameter_group(name) is None:
            continue
        gradients[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    return gradients


def _gradient_delta_norms(
    model: torch.nn.Module,
    before: dict[str, torch.Tensor | None],
) -> dict[str, float]:
    squared = {"element": 0.0, "lattice": 0.0, "coordinate": 0.0}
    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        if group is None or parameter.grad is None:
            continue
        previous = before.get(name)
        delta = parameter.grad.detach()
        if previous is not None:
            delta = delta - previous
        delta = delta.float()
        squared[group] += float((delta * delta).sum().detach().cpu())
    return {key: value**0.5 for key, value in squared.items()}


def _role_loss_report(
    role: GeneratedCarrierRole,
    metrics: dict[str, float],
    *,
    gradient_delta_norms: dict[str, float],
    graph_count: int,
    loss_weight: float,
    max_graphs_per_role_batch: int,
) -> dict[str, Any]:
    return {
        "role": role,
        "graph_count": graph_count,
        "loss_weight": loss_weight,
        "loss": metrics["loss"],
        "weighted_loss": metrics["loss"] * loss_weight,
        "element_loss": metrics["element_loss"],
        "composition_loss": metrics["composition_loss"],
        "coordinate_loss": metrics["coordinate_loss"],
        "volume_loss": metrics["volume_loss"],
        "shape_loss": metrics["shape_loss"],
        "masked_fraction": metrics["masked_fraction"],
        "gradient_delta_norms": gradient_delta_norms,
        "nonzero_gradient_delta_groups": {key: value > 0.0 for key, value in gradient_delta_norms.items()},
        "max_graphs_per_role_batch": int(max_graphs_per_role_batch),
    }


def _accumulate_role_step(
    trainer: ProductionTrainer,
    packed: PackedReplayRole,
    *,
    role_weight: float,
    generator: torch.Generator,
    max_graphs_per_role_batch: int,
) -> dict[str, Any]:
    total_graphs = int(packed.node_counts.numel())
    before_gradients = _clone_current_group_gradients(trainer.diffusion.denoiser)
    metric_sums = {
        "loss": 0.0,
        "element_loss": 0.0,
        "composition_loss": 0.0,
        "coordinate_loss": 0.0,
        "volume_loss": 0.0,
        "shape_loss": 0.0,
        "masked_fraction": 0.0,
    }
    for chunk in _iter_role_chunks(packed, max_graphs_per_role_batch):
        chunk_graphs = int(chunk.node_counts.numel())
        chunk_weight = role_weight * chunk_graphs / total_graphs
        output = trainer.accumulate_hybrid_step(
            chunk.assignment_tokens,
            chunk.fractional_coordinates,
            chunk.lattice,
            chunk.batch,
            chunk.blueprint,
            loss_weight=chunk_weight,
            generator=generator,
        )
        _check_composition_counts(chunk, output)
        metric_sums["loss"] += _finite_float(output.loss) * chunk_graphs
        metric_sums["element_loss"] += _finite_float(output.element_loss) * chunk_graphs
        metric_sums["composition_loss"] += _finite_float(output.composition_loss) * chunk_graphs
        metric_sums["coordinate_loss"] += _finite_float(output.coordinate_loss) * chunk_graphs
        metric_sums["volume_loss"] += _finite_float(output.volume_loss) * chunk_graphs
        metric_sums["shape_loss"] += _finite_float(output.shape_loss) * chunk_graphs
        metric_sums["masked_fraction"] += _finite_float(output.masked_fraction) * chunk_graphs
    metrics = {key: value / total_graphs for key, value in metric_sums.items()}
    gradient_deltas = _gradient_delta_norms(trainer.diffusion.denoiser, before_gradients)
    return _role_loss_report(
        packed.role,
        metrics,
        gradient_delta_norms=gradient_deltas,
        graph_count=total_graphs,
        loss_weight=role_weight,
        max_graphs_per_role_batch=max_graphs_per_role_batch,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.clean_retention_max_ratio <= 0.0:
        raise ValueError("clean retention max ratio must be positive")
    if args.max_graphs_per_role_batch < 0:
        raise ValueError("max graphs per role batch must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty output directory: {args.output_dir}")
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
    diffusion, raw_training_config, checkpoint_metadata = _load_diffusion(args.base_checkpoint, device=device)
    training_config = _training_config_from_checkpoint(
        raw_training_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    grouped = _grouped_entries(entries)
    packed_roles: dict[GeneratedCarrierRole, PackedReplayRole] = {
        role: _pack_role_entries(role, role_entries, device=device) for role, role_entries in grouped.items()
    }
    role_weight = _role_weight(len(_ROLES))
    trainer = ProductionTrainer(diffusion, training_config)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial_parameters = _clone_named_parameters(diffusion.denoiser)
    first_step_parameters = _clone_named_parameters(diffusion.denoiser)
    first_step_update_norm: float | None = None
    metrics_path = args.output_dir / "training_metrics.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_role_reports: list[dict[str, Any]] = []
    clean_retention_ratios: list[float] = []
    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(args.steps):
            trainer.begin_optimization_step()
            role_reports: list[dict[str, Any]] = []
            losses_by_role: dict[str, float] = {}
            for role in _ROLES:
                packed = packed_roles[role]
                report = _accumulate_role_step(
                    trainer,
                    packed,
                    role_weight=role_weight,
                    generator=generator,
                    max_graphs_per_role_batch=args.max_graphs_per_role_batch,
                )
                role_reports.append(report)
                losses_by_role[str(role)] = float(report["loss"])
            accumulated_gradient_norms = _gradient_group_norms(diffusion.denoiser)
            gradient_norm = trainer.finish_optimization_step()
            if first_step_update_norm is None:
                first_step_update_norm = _parameter_update_norm(diffusion.denoiser, first_step_parameters)
            generated_losses = [losses_by_role[str(role)] for role in _ROLES if role != "clean_clean"]
            clean_ratio = losses_by_role["clean_clean"] / max(max(generated_losses), 1.0e-12)
            clean_retention_ratios.append(clean_ratio)
            step_report = {
                "step": step + 1,
                "role_reports": role_reports,
                "loss_by_role": losses_by_role,
                "clean_retention_loss_ratio_to_max_generated": clean_ratio,
                "accumulated_gradient_norm": _finite_float(gradient_norm),
                "accumulated_terminal_gradient_group_norms": accumulated_gradient_norms,
                "parameter_update_norm_from_initial": _parameter_update_norm(
                    diffusion.denoiser,
                    initial_parameters,
                ),
            }
            if device.type == "cuda":
                step_report["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024.0**2
            handle.write(json.dumps(step_report, sort_keys=True) + "\n")
            handle.flush()
            final_role_reports = role_reports
    final_update_norm = _parameter_update_norm(diffusion.denoiser, initial_parameters)
    all_role_gradients_nonzero = all(
        all(bool(value) for value in report["nonzero_gradient_delta_groups"].values())
        for report in final_role_reports
    )
    clean_retention_not_exploded = max(clean_retention_ratios) <= args.clean_retention_max_ratio
    parameters_updated = final_update_norm > 0.0 and (first_step_update_norm or 0.0) > 0.0
    status = (
        "passed"
        if all_role_gradients_nonzero and clean_retention_not_exploded and parameters_updated
        else "failed"
    )
    summary = {
        "status": status,
        "schema": "gaugeflow.generated_state_replay_correctness_training.v1",
        "cache_dir": str(args.cache_dir),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": expected_base_sha,
        "manifest_sha256": manifest.canonical_sha256(),
        "entry_count": len(entries),
        "roles": list(_ROLES),
        "source_structure_ids_by_role": {
            str(role): list(packed.source_structure_ids) for role, packed in packed_roles.items()
        },
        "steps": args.steps,
        "seed": args.seed,
        "training_config": training_config.__dict__,
        "role_weight": role_weight,
        "max_graphs_per_role_batch": int(args.max_graphs_per_role_batch),
        "final_role_reports": final_role_reports,
        "clean_retention_loss_ratio_max": max(clean_retention_ratios),
        "clean_retention_max_ratio": args.clean_retention_max_ratio,
        "clean_retention_not_exploded": clean_retention_not_exploded,
        "all_final_role_terminal_gradient_groups_nonzero": all_role_gradients_nonzero,
        "first_step_parameter_update_norm": first_step_update_norm,
        "final_parameter_update_norm": final_update_norm,
        "parameters_updated": parameters_updated,
        "metrics_jsonl": str(metrics_path),
        "forbidden_source_id_check": {
            "executed": forbidden_source_ids is not None,
            "count": 0 if forbidden_source_ids is None else len(forbidden_source_ids),
        },
        "checkpoint_protocol": checkpoint_metadata.get("protocol", "unknown"),
    }
    if args.save_checkpoint:
        checkpoint_path = args.output_dir / f"checkpoint_step_{args.steps:08d}.pt"
        torch.save(
            {
                "schema": "gaugeflow.generated_state_replay_correctness_training.v1",
                "model": _cpu_state_dict(diffusion.denoiser),
                "ema": trainer.ema.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "trainer_step": trainer.step,
                "summary": summary,
            },
            checkpoint_path,
        )
        summary["checkpoint"] = str(checkpoint_path)
        summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
    _write_json(args.output_dir / "training_summary.json", summary)
    checkpoint_metadata = {key: summary[key] for key in summary if key != "final_role_reports"}
    _write_json(args.output_dir / "checkpoint_metadata.json", checkpoint_metadata)
    if status != "passed":
        raise RuntimeError("generated-state replay correctness training smoke failed")
    return summary


def main() -> None:
    summary = _run_training(_parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
