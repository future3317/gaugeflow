"""Evaluate a generated-state replay correctness checkpoint.

This diagnostic compares the frozen Stage-C base and a generated-state replay
correctness checkpoint on two bounded surfaces:

1. replay-cache per-role denoising losses with fixed noise seeds;
2. tensor-free free-generation retention with the existing A1 metric code.

It is not a Stage-E pass, not a capacity competition, and not a tensor
conditioning evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from audit_generated_state_replay_training_contract import (
    _ROLES,
    PackedReplayRole,
    _finite_float,
    _pack_role_entries,
    _read_forbidden_source_ids,
)
from evaluate_gaugeflow_base_a1 import reference_statistics
from evaluate_physical_representation import evaluate_generation_retention

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.checkpointing import load_production_checkpoint
from gaugeflow.production.composition_runtime import load_qualified_composition_model
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-cache-dir", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-protocol", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--validation-graphs", type=int, default=128)
    parser.add_argument("--free-samples", type=int, default=128)
    parser.add_argument("--reverse-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-checkpoint-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-sampler-commit", default=None)
    parser.add_argument("--expected-sampler-protocol-sha256", default=None)
    parser.add_argument("--forbidden-source-ids", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _training_config(metadata: dict[str, Any]) -> dict[str, Any]:
    stage_b = metadata.get("stage_b_metadata")
    if not isinstance(stage_b, dict):
        raise ValueError("Stage-C checkpoint lacks Stage-B metadata")
    training = stage_b.get("a1_training_config")
    if not isinstance(training, dict):
        raise ValueError("Stage-C checkpoint lacks A1 training config")
    return dict(training)


def _standardization(metadata: dict[str, Any]) -> dict[str, Any]:
    stage_b = metadata.get("stage_b_metadata")
    if not isinstance(stage_b, dict):
        raise ValueError("Stage-C checkpoint lacks Stage-B metadata")
    standardization = stage_b.get("lattice_standardization")
    if not isinstance(standardization, dict):
        raise ValueError("Stage-C checkpoint lacks lattice standardization")
    return dict(standardization)


def _load_stage_c_base_backbone(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[HybridCrystalDenoiser, dict[str, Any], dict[str, Any]]:
    metadata = read_physical_checkpoint_metadata(checkpoint)
    stage_b = metadata.get("stage_b_metadata")
    if not isinstance(stage_b, dict):
        raise ValueError("base checkpoint is not a Stage-C physical checkpoint")
    objects = build_continued_pretraining_objects(stage_b, device=device, optimizer_owner=False)
    physical_config = stage_b.get("physical_training_config")
    if not isinstance(physical_config, dict):
        raise ValueError("Stage-C checkpoint lacks physical training config")
    ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
    load_physical_ema_for_evaluation(checkpoint, model=objects.model, ema=ema, map_location=device)
    objects.model.eval()
    return objects.model.backbone, dict(stage_b), metadata


def _load_correctness_backbone(
    checkpoint: Path,
    base_metadata: dict[str, Any],
    *,
    device: torch.device,
    use_ema: bool,
) -> tuple[HybridCrystalDenoiser, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "gaugeflow.generated_state_replay_correctness_training.v1"
    ):
        raise ValueError("checkpoint is not a generated-state replay correctness checkpoint")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("correctness checkpoint lacks summary")
    stage_b = base_metadata.get("stage_b_metadata")
    if not isinstance(stage_b, dict):
        raise ValueError("base checkpoint lacks Stage-B metadata")
    model_config = stage_b.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("base checkpoint lacks model config")
    model = HybridCrystalDenoiser(**model_config).to(device)
    if use_ema:
        training_config = summary.get("training_config")
        if not isinstance(training_config, dict):
            raise ValueError("correctness checkpoint lacks training config")
        ema = ExponentialMovingAverage(model, float(training_config["ema_decay"]))
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model)
    else:
        model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, summary


def _diffusion_for_backbone(
    backbone: HybridCrystalDenoiser,
    stage_b_metadata: dict[str, Any],
) -> TensorFreeHybridDiffusion:
    training = stage_b_metadata["a1_training_config"]
    return TensorFreeHybridDiffusion(
        backbone,
        P1LatticeStandardizer.from_mapping(stage_b_metadata["lattice_standardization"]),
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
        categorical_path=str(training["categorical_path"]),
        composition_conditioning=bool(training["composition_conditioning"]),
    )


def _grouped_entries(
    entries: list[GeneratedStateReplayEntry],
) -> dict[GeneratedCarrierRole, list[GeneratedStateReplayEntry]]:
    return {role: [entry for entry in entries if entry.key.role == role] for role in _ROLES}


@torch.no_grad()
def _evaluate_replay_losses(
    diffusion: TensorFreeHybridDiffusion,
    packed_roles: dict[GeneratedCarrierRole, PackedReplayRole],
    *,
    seed: int,
    precision: str,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for role_index, role in enumerate(_ROLES):
        packed = packed_roles[role]
        generator = torch.Generator(device=packed.lattice.device).manual_seed(seed + 1009 * role_index)
        with torch.autocast(
            device_type=packed.lattice.device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and packed.lattice.device.type == "cuda",
        ):
            output = diffusion(
                packed.assignment_tokens,
                packed.fractional_coordinates,
                packed.lattice,
                packed.batch,
                packed.blueprint.shape_projector,
                packed.blueprint.fractional_to_cartesian,
                generator=generator,
            )
        if output.noisy.composition_counts is None:
            raise RuntimeError("replay evaluation did not pass composition counts")
        if not torch.equal(output.noisy.composition_counts.detach().cpu(), packed.composition_counts.detach().cpu()):
            raise RuntimeError(f"role {role} used composition counts different from replay entry")
        reports[str(role)] = {
            "graph_count": int(packed.node_counts.numel()),
            "loss": _finite_float(output.loss),
            "element_loss": _finite_float(output.element_loss),
            "composition_loss": _finite_float(output.composition_loss),
            "coordinate_loss": _finite_float(output.coordinate_loss),
            "volume_loss": _finite_float(output.volume_loss),
            "shape_loss": _finite_float(output.shape_loss),
            "masked_fraction": _finite_float(output.masked_fraction),
        }
    return reports


def _loss_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for role in _ROLES:
        key = str(role)
        deltas[key] = {
            metric: float(candidate[key][metric]) - float(baseline[key][metric])
            for metric in ("loss", "element_loss", "coordinate_loss", "volume_loss", "shape_loss")
        }
    return deltas


def _generation_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    metrics = (
        "normalized_nearest_neighbor_wasserstein",
        "normalized_volume_wasserstein",
        "minimum_distance_fraction_at_0_5_angstrom",
        "exact_composition_fraction",
        "finite_positive_lattice_fraction",
        "element_marginal_jsd",
        "node_count_jsd",
        "formula_uniqueness_fraction",
        "sampling_failures",
        "terminal_masks",
    )
    return {
        metric: float(candidate.get(metric, float("nan"))) - float(baseline.get(metric, float("nan")))
        for metric in metrics
    }


def _evaluation_config(args: argparse.Namespace, a1_protocol: dict[str, Any]) -> dict[str, Any]:
    base = copy.deepcopy(a1_protocol["evaluation"])
    base["validation_graphs"] = int(args.validation_graphs)
    base["free_samples"] = int(args.free_samples)
    base["reverse_steps"] = int(args.reverse_steps)
    base["batch_size"] = int(args.batch_size)
    base["wasserstein_quantile_points"] = min(
        int(base["wasserstein_quantile_points"]),
        int(args.free_samples) + 1,
    )
    return base


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    base_sha = sha256_file(args.base_checkpoint)
    forbidden = _read_forbidden_source_ids(args.forbidden_source_ids)
    entries, manifest = load_generated_state_replay_cache(
        args.replay_cache_dir,
        expected_base_checkpoint_sha256=base_sha,
        expected_sampler_commit=args.expected_sampler_commit,
        expected_sampler_protocol_sha256=args.expected_sampler_protocol_sha256,
        forbidden_source_ids=forbidden,
    )
    base_backbone, stage_b_metadata, base_metadata = _load_stage_c_base_backbone(args.base_checkpoint, device=device)
    candidate_backbone, checkpoint_summary = _load_correctness_backbone(
        args.checkpoint,
        base_metadata,
        device=device,
        use_ema=bool(args.use_checkpoint_ema),
    )
    precision = str(_training_config(base_metadata).get("precision", "fp32"))
    grouped = _grouped_entries(entries)
    packed_roles = {
        role: _pack_role_entries(role, role_entries, device=device)
        for role, role_entries in grouped.items()
    }
    replay_base = _evaluate_replay_losses(
        _diffusion_for_backbone(base_backbone, stage_b_metadata),
        packed_roles,
        seed=int(args.seed),
        precision=precision,
    )
    replay_candidate = _evaluate_replay_losses(
        _diffusion_for_backbone(candidate_backbone, stage_b_metadata),
        packed_roles,
        seed=int(args.seed),
        precision=precision,
    )
    a1_protocol = _read_json(args.a1_protocol)
    evaluation = _evaluation_config(args, a1_protocol)
    alex = PackedAlexP1Dataset(args.alex_cache, "val")
    indices = torch.randperm(
        len(alex),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    reference = reference_statistics(
        alex,
        indices,
        batch_size=int(evaluation["batch_size"]),
        device=device,
    )
    node_model = HybridCrystalDenoiser(**stage_b_metadata["model_config"]).to(device)
    _, node_prior, _ = load_production_checkpoint(args.a1_checkpoint, model=node_model, map_location=device)
    del node_model
    composition = load_qualified_composition_model(
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
        expected_checkpoint_sha256=str(a1_protocol["composition_checkpoint_sha256"]),
    )
    generation_base = evaluate_generation_retention(
        base_backbone,
        node_prior,
        _standardization(base_metadata),
        _training_config(base_metadata),
        evaluation,
        reference,
        composition,
        device=device,
    )
    generation_candidate = evaluate_generation_retention(
        candidate_backbone,
        node_prior,
        _standardization(base_metadata),
        _training_config(base_metadata),
        evaluation,
        reference,
        composition,
        device=device,
    )
    output = {
        "schema": "gaugeflow.generated_state_replay_correctness_evaluation.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_ema_used": bool(args.use_checkpoint_ema),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": base_sha,
        "replay_cache_dir": str(args.replay_cache_dir),
        "replay_manifest_sha256": manifest.canonical_sha256(),
        "checkpoint_training_summary": {
            key: checkpoint_summary.get(key)
            for key in (
                "status",
                "steps",
                "clean_retention_loss_ratio_max",
                "final_parameter_update_norm",
            )
        },
        "forbidden_source_id_check": {
            "executed": forbidden is not None,
            "count": 0 if forbidden is None else len(forbidden),
        },
        "replay_role_losses": {
            "base": replay_base,
            "candidate": replay_candidate,
            "candidate_minus_base": _loss_deltas(replay_candidate, replay_base),
        },
        "free_generation": {
            "evaluation": evaluation,
            "validation_indices": indices.tolist(),
            "base": generation_base,
            "candidate": generation_candidate,
            "candidate_minus_base": _generation_delta(generation_candidate, generation_base),
        },
        "decision_boundary": (
            "diagnostic_only; this evaluates generated-state replay correctness and "
            "short free-generation retention, not Stage-E pass or capacity selection"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
