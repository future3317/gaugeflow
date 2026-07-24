"""Run the first A-v2 clean plus generated-state replay training smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from audit_generated_state_replay_training_contract import (
    _ROLES,
    _gradient_group_norms,
    _pack_role_entries,
    _read_forbidden_source_ids,
)
from torch.utils.data import DataLoader
from train_generated_state_replay_correctness import (
    _accumulate_role_step,
    _clone_current_group_gradients,
    _clone_named_parameters,
    _gradient_delta_norms,
    _grouped_entries,
    _parameter_update_norm,
    _write_json,
)

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.generated_state_replay import load_generated_state_replay_cache
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig

_CHECKPOINT_SCHEMA = "gaugeflow.base_v2_generated_state_training_checkpoint.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--replay-cache-dir", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--forbidden-source-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        help="Save periodic exact-resume checkpoints; by default only step 0 and the final step are saved.",
    )
    parser.add_argument(
        "--stop-step",
        type=int,
        help="Stop this invocation before the protocol step count; used for interrupted-resume smokes.",
    )
    return parser.parse_args()


def _model_config(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_dim": int(specification["hidden_dim"]),
        "vector_dim": int(specification["vector_dim"]),
        "layers": int(specification["layers"]),
        "radial_dim": int(specification["radial_dim"]),
        "radial_cutoff": float(specification["radial_cutoff_angstrom"]),
        "atlas_residual_circle_samples": 8,
        "edge_dim": int(specification["edge_dim"]),
        "angular_channels": int(specification["angular_channels"]),
        "edge_refresh_rank": int(specification["edge_refresh_rank"]),
        "modality_time_conditioning": str(specification["modality_time_conditioning"]),
    }


def _training_config(training: dict[str, Any]) -> ProductionTrainingConfig:
    config = ProductionTrainingConfig(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        ema_decay=float(training["ema_decay"]),
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
        precision=str(training["precision"]),
        objective="joint",
        modality_time_mode="shared",
        categorical_path=str(training["categorical_path"]),
        composition_conditioning=bool(training["composition_conditioning"]),
    )
    config.validate()
    return config


def _move_clean_batch(host: Any, *, device: torch.device) -> tuple[torch.Tensor, ...]:
    moved = host.to(device, non_blocking=True)
    graph_count = int(moved.num_graphs)
    node_counts = torch.bincount(moved.batch, minlength=graph_count)
    blueprint = ParentBlueprintBatch.from_node_counts(
        node_counts,
        dtype=moved.frac_coords.dtype,
        device=device,
    )
    return moved.atom_types, moved.frac_coords, moved.lattice, moved.batch, blueprint


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"A-v2 checkpoint value is not tensor/JSON safe: {type(value).__name__}")


def _optimizer_execution_backend(optimizer: torch.optim.Optimizer) -> list[tuple[Any, Any]]:
    return [(group.get("fused"), group.get("foreach")) for group in optimizer.param_groups]


def _restore_optimizer_execution_backend(
    optimizer: torch.optim.Optimizer,
    backend: list[tuple[Any, Any]],
) -> None:
    if len(backend) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer group count changed during restore")
    for group, (fused, foreach) in zip(optimizer.param_groups, backend, strict=True):
        group["fused"] = fused
        group["foreach"] = foreach


def _read_training_checkpoint_description(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("A-v2 training checkpoint requires weights and JSON sidecar")
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = description.get("metadata")
    if (
        description.get("schema") != _CHECKPOINT_SCHEMA
        or description.get("weights_file") != path.name
        or description.get("weights_sha256") != sha256_file(path)
        or not isinstance(metadata, dict)
        or description.get("metadata_sha256") != canonical_json_hash(metadata)
    ):
        raise ValueError("A-v2 training checkpoint sidecar failed schema/hash validation")
    return description


def _save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    trainer: ProductionTrainer,
    trainer_step: int,
    metadata: dict[str, Any],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    if trainer_step < 0:
        raise ValueError("checkpoint step must be nonnegative")
    if trainer.optimizer is None or trainer.ema is None:
        raise RuntimeError("A-v2 checkpoint requires optimizer and EMA state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "model": _cpu_tree(model.state_dict()),
        "ema": _cpu_tree(trainer.ema.state_dict()),
        "optimizer": _cpu_tree(trainer.optimizer.state_dict()),
        "trainer_step": int(trainer_step),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "runtime_state": _cpu_tree(runtime_state),
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    description = {
        "schema": _CHECKPOINT_SCHEMA,
        "weights_file": path.name,
        "weights_sha256": sha256_file(path),
        "metadata": dict(metadata),
        "metadata_sha256": canonical_json_hash(metadata),
    }
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar_temporary.replace(sidecar)
    return {
        "step": int(trainer_step),
        "path": str(path),
        "sha256": description["weights_sha256"],
        "sidecar": str(sidecar),
    }


def _load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    trainer: ProductionTrainer,
    map_location: str | torch.device,
    restore_rng: bool = True,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    description = _read_training_checkpoint_description(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("unsupported A-v2 training checkpoint schema")
    model.load_state_dict(payload["model"], strict=True)
    trainer.ema.load_state_dict(payload["ema"])
    backend = _optimizer_execution_backend(trainer.optimizer)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    _restore_optimizer_execution_backend(trainer.optimizer, backend)
    if restore_rng:
        torch.set_rng_state(payload["cpu_rng_state"].cpu())
        cuda_state = payload["cuda_rng_state"]
        if torch.cuda.is_available() and cuda_state:
            if not isinstance(cuda_state, list) or not all(
                isinstance(state, torch.Tensor) and state.dtype == torch.uint8 for state in cuda_state
            ):
                raise ValueError("checkpoint CUDA RNG state is invalid")
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    runtime_state = payload.get("runtime_state")
    if not isinstance(runtime_state, dict):
        raise ValueError("A-v2 checkpoint lacks exact-resume runtime state")
    required = {"epoch_loader_generator_state", "batches_consumed_in_epoch", "device_generator_state"}
    if not required.issubset(runtime_state):
        raise ValueError("A-v2 checkpoint runtime state is incomplete")
    if (
        not isinstance(runtime_state["batches_consumed_in_epoch"], int)
        or runtime_state["batches_consumed_in_epoch"] < 0
    ):
        raise ValueError("A-v2 checkpoint has an invalid epoch cursor")
    for name in ("epoch_loader_generator_state", "device_generator_state"):
        if not isinstance(runtime_state[name], torch.Tensor) or runtime_state[name].dtype != torch.uint8:
            raise ValueError(f"A-v2 checkpoint {name} is not a generator state")
    return int(payload["trainer_step"]), description["metadata"], runtime_state


def _accumulate_clean_batches(
    trainer: ProductionTrainer,
    batches: list[Any],
    *,
    clean_loss_weight: float,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    total_graphs = sum(int(batch.num_graphs) for batch in batches)
    before_gradients = _clone_current_group_gradients(trainer.diffusion.denoiser)
    losses = {
        "loss": 0.0,
        "element_loss": 0.0,
        "composition_loss": 0.0,
        "coordinate_loss": 0.0,
        "volume_loss": 0.0,
        "shape_loss": 0.0,
        "masked_fraction": 0.0,
    }
    for host in batches:
        graph_count = int(host.num_graphs)
        clean_elements, clean_frac, clean_lattice, batch, blueprint = _move_clean_batch(host, device=device)
        output = trainer.accumulate_hybrid_step(
            clean_elements,
            clean_frac,
            clean_lattice,
            batch,
            blueprint,
            loss_weight=clean_loss_weight * graph_count / total_graphs,
            generator=generator,
        )
        losses["loss"] += float(output.loss.detach().cpu()) * graph_count
        losses["element_loss"] += float(output.element_loss.detach().cpu()) * graph_count
        losses["composition_loss"] += float(output.composition_loss.detach().cpu()) * graph_count
        losses["coordinate_loss"] += float(output.coordinate_loss.detach().cpu()) * graph_count
        losses["volume_loss"] += float(output.volume_loss.detach().cpu()) * graph_count
        losses["shape_loss"] += float(output.shape_loss.detach().cpu()) * graph_count
        losses["masked_fraction"] += float(output.masked_fraction.detach().cpu()) * graph_count
    metrics = {key: value / total_graphs for key, value in losses.items()}
    gradient_deltas = _gradient_delta_norms(trainer.diffusion.denoiser, before_gradients)
    return {
        "graph_count": total_graphs,
        "loss_weight": clean_loss_weight,
        **metrics,
        "gradient_delta_norms": gradient_deltas,
        "nonzero_gradient_delta_groups": {key: value > 0.0 for key, value in gradient_deltas.items()},
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "gaugeflow_base_v2_generated_state_smoke_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen A-v2 smoke protocol")
    source = protocol["source"]
    training = protocol["training"]
    candidates = protocol["model_candidates"]
    if args.candidate not in candidates:
        raise ValueError("unknown A-v2 smoke candidate")
    if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty output directory: {args.output_dir}")
    if args.resume is not None and args.resume.parent.resolve() != args.output_dir.resolve():
        raise ValueError("A-v2 resume checkpoint must live directly inside the output directory")
    if args.checkpoint_every_steps is not None and args.checkpoint_every_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    target_step = int(training["steps"]) if args.stop_step is None else int(args.stop_step)
    if target_step < 1 or target_step > int(training["steps"]):
        raise ValueError("A-v2 stop step must lie in [1, protocol training steps]")
    if sha256_file(args.alex_cache / "manifest.json") != str(source["alex_cache_manifest_sha256"]):
        raise ValueError("Alex cache manifest hash mismatch")
    if canonical_json_hash(load_json_object(args.lattice_standardization)) != str(
        source["lattice_standardization_canonical_sha256"]
    ):
        raise ValueError("lattice standardization hash mismatch")
    if sha256_file(args.forbidden_source_ids) != str(source["forbidden_source_ids_sha256"]):
        raise ValueError("forbidden-source ID file hash mismatch")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = HybridCrystalDenoiser(**_model_config(candidates[args.candidate])).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(candidates[args.candidate]["parameter_count"]):
        raise ValueError("candidate parameter count changed")
    config = _training_config(training)
    diffusion = TensorFreeHybridDiffusion(
        model,
        P1LatticeStandardizer.from_mapping(load_json_object(args.lattice_standardization)),
        coordinate_sigma_min=config.coordinate_sigma_min,
        coordinate_sigma_max=config.coordinate_sigma_max,
        minimum_time=config.minimum_time,
        maximum_time=config.maximum_time,
        categorical_path=config.categorical_path,
        composition_conditioning=config.composition_conditioning,
    )
    trainer = ProductionTrainer(diffusion, config)

    forbidden_source_ids = _read_forbidden_source_ids(args.forbidden_source_ids)
    entries, manifest = load_generated_state_replay_cache(
        args.replay_cache_dir,
        expected_sampler_commit=str(source["replay_sampler_commit"]),
        expected_sampler_protocol_sha256=str(source["replay_sampler_protocol_sha256"]),
        forbidden_source_ids=forbidden_source_ids,
    )
    if manifest.canonical_sha256() != str(source["replay_manifest_sha256"]):
        raise ValueError("replay manifest hash mismatch")
    packed_roles = {
        role: _pack_role_entries(role, role_entries, device=device)
        for role, role_entries in _grouped_entries(entries).items()
    }

    dataset = PackedAlexP1Dataset(args.alex_cache, "train")
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["clean_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_packed_alex,
        generator=loader_generator,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    iterator = iter(loader)
    clean_accumulation = int(training["clean_gradient_accumulation_steps"])
    clean_loss_weight = float(training["clean_loss_weight"])
    replay_loss_weight = float(training["replay_loss_weight"])
    if clean_accumulation < 1 or not 0.0 < clean_loss_weight < 1.0:
        raise ValueError("invalid clean accumulation or loss weight")
    if not 0.0 < replay_loss_weight < 1.0 or abs(clean_loss_weight + replay_loss_weight - 1.0) > 1.0e-12:
        raise ValueError("clean and replay loss weights must sum to one")

    generator = torch.Generator(device=device).manual_seed(seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = canonical_json_hash(protocol)
    checkpoint_metadata = {
        "schema": _CHECKPOINT_SCHEMA,
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha256,
        "candidate": args.candidate,
        "parameter_count": parameter_count,
        "training": training,
        "replay_manifest_sha256": manifest.canonical_sha256(),
        "entry_count": len(entries),
        "forbidden_source_id_count": len(forbidden_source_ids or set()),
        "boundary": protocol["boundary"],
    }
    resume_step = 0
    resume_runtime: dict[str, Any] | None = None
    resume_metadata: dict[str, Any] | None = None
    if args.resume is not None:
        resume_step, resume_metadata, resume_runtime = _load_training_checkpoint(
            args.resume,
            model=model,
            trainer=trainer,
            map_location=device,
            restore_rng=True,
        )
        if resume_metadata.get("protocol_sha256") != protocol_sha256:
            raise ValueError("A-v2 resume checkpoint protocol hash mismatch")
        if resume_metadata.get("candidate") != args.candidate:
            raise ValueError("A-v2 resume checkpoint candidate mismatch")
        trainer.step = resume_step
        generator.set_state(resume_runtime["device_generator_state"].cpu())
    initial_parameters = _clone_named_parameters(model)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    if args.resume is not None and not metrics_path.is_file():
        raise FileNotFoundError("resume requested but training_metrics.jsonl is missing")
    if resume_runtime is None:
        epoch_loader_generator_state = loader_generator.get_state().clone()
        iterator = iter(loader)
        batches_consumed_in_epoch = 0
    else:
        epoch_loader_generator_state = resume_runtime["epoch_loader_generator_state"].cpu()
        batches_consumed_in_epoch = int(resume_runtime["batches_consumed_in_epoch"])
        if batches_consumed_in_epoch > len(loader):
            raise ValueError("resume data cursor lies beyond the frozen epoch")
        loader_generator.set_state(epoch_loader_generator_state)
        iterator = iter(loader)
        for _ in range(batches_consumed_in_epoch):
            next(iterator)

    def exact_resume_state() -> dict[str, Any]:
        return {
            "epoch_loader_generator_state": epoch_loader_generator_state.clone(),
            "batches_consumed_in_epoch": batches_consumed_in_epoch,
            "device_generator_state": generator.get_state(),
        }

    checkpoints: list[dict[str, Any]] = []

    def save_checkpoint(step: int) -> None:
        checkpoints.append(
            _save_training_checkpoint(
                args.output_dir / f"checkpoint_step_{step:08d}.pt",
                model=model,
                trainer=trainer,
                trainer_step=step,
                metadata=checkpoint_metadata,
                runtime_state=exact_resume_state(),
            )
        )

    if args.resume is None:
        save_checkpoint(0)

    def next_host_batch() -> Any:
        nonlocal iterator, epoch_loader_generator_state, batches_consumed_in_epoch
        try:
            value = next(iterator)
        except StopIteration:
            epoch_loader_generator_state = loader_generator.get_state().clone()
            iterator = iter(loader)
            batches_consumed_in_epoch = 0
            value = next(iterator)
        batches_consumed_in_epoch += 1
        return value

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_clean_report: dict[str, Any] = {}
    final_replay_reports: list[dict[str, Any]] = []
    if trainer.step >= target_step:
        raise ValueError("A-v2 resume checkpoint is already at or beyond the requested stop step")
    with metrics_path.open("a" if args.resume is not None else "w", encoding="utf-8") as handle:
        while trainer.step < target_step:
            trainer.begin_optimization_step()
            clean_batches = [next_host_batch() for _ in range(clean_accumulation)]
            clean_report = _accumulate_clean_batches(
                trainer,
                clean_batches,
                clean_loss_weight=clean_loss_weight,
                generator=generator,
                device=device,
            )
            replay_reports = [
                _accumulate_role_step(
                    trainer,
                    packed_roles[role],
                    role_weight=replay_loss_weight / len(_ROLES),
                    generator=generator,
                    max_graphs_per_role_batch=int(training["max_graphs_per_role_batch"]),
                )
                for role in _ROLES
            ]
            terminal_gradient_norms = _gradient_group_norms(model)
            gradient_norm = trainer.finish_optimization_step()
            step = trainer.step
            step_report = {
                "step": step,
                "clean_report": clean_report,
                "replay_role_reports": replay_reports,
                "terminal_gradient_group_norms": terminal_gradient_norms,
                "gradient_norm": float(gradient_norm.cpu()),
                "parameter_update_norm_from_initial": _parameter_update_norm(model, initial_parameters),
            }
            if device.type == "cuda":
                step_report["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024.0**2
            handle.write(json.dumps(step_report, sort_keys=True) + "\n")
            handle.flush()
            final_clean_report = clean_report
            final_replay_reports = replay_reports
            if args.checkpoint_every_steps is not None and step % int(args.checkpoint_every_steps) == 0:
                save_checkpoint(step)

    if not checkpoints or checkpoints[-1]["step"] != trainer.step:
        save_checkpoint(trainer.step)

    final_update_norm = _parameter_update_norm(model, initial_parameters)
    acceptance = protocol["acceptance"]
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024.0**2 if device.type == "cuda" else 0.0
    clean_groups_nonzero = all(
        bool(value) for value in final_clean_report["nonzero_gradient_delta_groups"].values()
    )
    replay_groups_nonzero = all(
        all(bool(value) for value in report["nonzero_gradient_delta_groups"].values())
        for report in final_replay_reports
    )
    checks = {
        "parameters_updated": final_update_norm > 0.0,
        "clean_terminal_gradient_groups_nonzero": clean_groups_nonzero,
        "all_replay_role_terminal_gradient_groups_nonzero": replay_groups_nonzero,
        "memory": peak_memory <= float(acceptance["peak_cuda_memory_mib_max"]),
    }
    summary = {
        "schema": "gaugeflow.base_v2_generated_state_smoke.v1",
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha256,
        "candidate": args.candidate,
        "parameter_count": parameter_count,
        "training": training,
        "requested_stop_step": target_step,
        "replay_manifest_sha256": manifest.canonical_sha256(),
        "entry_count": len(entries),
        "forbidden_source_id_check": {"executed": True, "count": len(forbidden_source_ids or set())},
        "final_clean_report": final_clean_report,
        "final_replay_role_reports": final_replay_reports,
        "final_terminal_gradient_group_norms": _gradient_group_norms(model),
        "final_parameter_update_norm": final_update_norm,
        "peak_cuda_memory_mib": peak_memory,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "metrics_jsonl": str(metrics_path),
        "checkpoints": checkpoints,
        "resumed_from": None
        if args.resume is None
        else {"path": str(args.resume), "step": resume_step, "metadata": resume_metadata},
        "short_run_selector_contract": protocol.get("short_run_selector"),
        "boundary": protocol["boundary"],
    }
    _write_json(args.output_dir / "training_summary.json", summary)
    if summary["status"] != "passed":
        raise RuntimeError("A-v2 generated-state smoke failed")
    return summary


def main() -> None:
    print(json.dumps(_run(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
