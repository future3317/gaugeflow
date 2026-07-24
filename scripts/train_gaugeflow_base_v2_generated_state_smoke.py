"""Run the first A-v2 clean plus generated-state replay training smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from audit_generated_state_replay_training_contract import (
    _ROLES,
    _gradient_group_norms,
    _pack_role_entries,
    _read_forbidden_source_ids,
)
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty output directory: {args.output_dir}")
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
    loader = DataLoader(
        dataset,
        batch_size=int(training["clean_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_packed_alex,
        generator=torch.Generator().manual_seed(seed),
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

    initial_parameters = _clone_named_parameters(model)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_clean_report: dict[str, Any] = {}
    final_replay_reports: list[dict[str, Any]] = []
    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(int(training["steps"])):
            trainer.begin_optimization_step()
            clean_batches = [next(iterator) for _ in range(clean_accumulation)]
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
            step_report = {
                "step": step + 1,
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

    final_update_norm = _parameter_update_norm(model, initial_parameters)
    acceptance = protocol["acceptance"]
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024.0**2 if device.type == "cuda" else 0.0
    clean_groups_nonzero = all(bool(value) for value in final_clean_report["nonzero_gradient_delta_groups"].values())
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
        "protocol_sha256": canonical_json_hash(protocol),
        "candidate": args.candidate,
        "parameter_count": parameter_count,
        "training": training,
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
