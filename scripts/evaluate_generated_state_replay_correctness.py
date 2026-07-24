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
from evaluate_gaugeflow_base_a1 import dense_token_counts, reference_statistics
from evaluate_physical_representation import evaluate_generation_retention
from train_gaugeflow_base_v2_generated_state_smoke import (
    _CHECKPOINT_SCHEMA as A_V2_CHECKPOINT_SCHEMA,
)
from train_gaugeflow_base_v2_generated_state_smoke import (
    _model_config as _a_v2_model_config,
)
from train_gaugeflow_base_v2_generated_state_smoke import (
    _read_training_checkpoint_description as _read_a_v2_checkpoint_description,
)

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import EmpiricalNodeCountPrior, ParentBlueprintBatch
from gaugeflow.production.checkpointing import load_production_checkpoint
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.generated_state_replay import (
    GeneratedCarrierRole,
    GeneratedStateReplayEntry,
    load_generated_state_replay_cache,
)
from gaugeflow.production.generation_metrics import (
    element_histogram,
    formula_keys,
    jensen_shannon,
    minimum_periodic_distances,
    quantile_wasserstein,
    robust_scale,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
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
    parser.add_argument(
        "--candidate-protocol",
        type=Path,
        default=None,
        help="Frozen A-v2 protocol JSON required when --checkpoint is an A-v2 training checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--validation-graphs", type=int, default=128)
    parser.add_argument("--free-samples", type=int, default=128)
    parser.add_argument("--reverse-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-checkpoint-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--record-free-generation-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-sample base/candidate free-generation retention records for paired diagnostics.",
    )
    parser.add_argument(
        "--paired-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap draws for optional paired free-generation retention diagnostics.",
    )
    parser.add_argument(
        "--paired-bootstrap-seed",
        type=int,
        default=6101,
        help="CPU RNG seed for optional paired free-generation retention bootstrap.",
    )
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
    candidate_protocol: dict[str, Any] | None,
) -> tuple[HybridCrystalDenoiser, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a generated-state replay correctness checkpoint")
    if payload.get("schema") == A_V2_CHECKPOINT_SCHEMA:
        if candidate_protocol is None:
            raise ValueError("A-v2 checkpoint evaluation requires --candidate-protocol")
        description = _read_a_v2_checkpoint_description(checkpoint)
        metadata = description["metadata"]
        candidate = str(metadata["candidate"])
        candidates = candidate_protocol.get("model_candidates")
        if not isinstance(candidates, dict) or candidate not in candidates:
            raise ValueError("A-v2 candidate protocol lacks checkpoint model spec")
        model = HybridCrystalDenoiser(**_a_v2_model_config(candidates[candidate])).to(device)
        if use_ema:
            ema = ExponentialMovingAverage(model, float(metadata["training"]["ema_decay"]))
            ema.load_state_dict(payload["ema"])
            ema.copy_to(model)
        else:
            model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return model, {
            "schema": A_V2_CHECKPOINT_SCHEMA,
            "checkpoint_metadata": metadata,
            "candidate_protocol": str(candidate_protocol["protocol"]),
        }
    if payload.get("schema") != "gaugeflow.generated_state_replay_correctness_training.v1":
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


def _diffusion_from_config(
    backbone: HybridCrystalDenoiser,
    standardization: dict[str, Any],
    training: dict[str, Any],
) -> TensorFreeHybridDiffusion:
    return TensorFreeHybridDiffusion(
        backbone,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
        categorical_path=str(training["categorical_path"]),
        composition_conditioning=bool(training["composition_conditioning"]),
    )


def _diffusion_for_backbone(
    backbone: HybridCrystalDenoiser,
    stage_b_metadata: dict[str, Any],
) -> TensorFreeHybridDiffusion:
    return _diffusion_from_config(
        backbone,
        dict(stage_b_metadata["lattice_standardization"]),
        dict(stage_b_metadata["a1_training_config"]),
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


def _reference_records(
    reference: dict[str, torch.Tensor],
    validation_indices: torch.Tensor,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row, source_index in enumerate(validation_indices.cpu().tolist()):
        records.append(
            {
                "row": row,
                "source_index": int(source_index),
                "node_count": int(reference["node_counts"][row]),
                "volume_per_atom": float(reference["volume_per_atom"][row]),
                "minimum_distance": float(reference["minimum_distance"][row]),
            }
        )
    return records


@torch.no_grad()
def _evaluate_generation_retention_with_records(
    backbone: HybridCrystalDenoiser,
    node_prior: EmpiricalNodeCountPrior,
    standardization: dict[str, Any],
    a1_training: dict[str, Any],
    a1_evaluation: dict[str, Any],
    reference: dict[str, torch.Tensor],
    composition_model: Any,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampler = TensorFreeReverseSampler(
        backbone,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(a1_training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(a1_training["coordinate_sigma_max"]),
        maximum_time=float(a1_training["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_model=composition_model,
    )
    sample_count = int(a1_evaluation["free_samples"])
    seed = int(a1_evaluation["sampling_seed"])
    counts = node_prior.sample(sample_count, generator=torch.Generator().manual_seed(seed))
    initialization_generator = torch.Generator(device=device).manual_seed(seed + 1)
    categorical_generator = torch.Generator(device=device).manual_seed(seed + 2)
    continuous_generator = torch.Generator(device=device).manual_seed(seed + 3)
    elements = torch.zeros(118, dtype=torch.float64)
    volumes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    formulas: list[str] = []
    records: list[dict[str, Any]] = []
    failures = terminal_masks = exact_composition = positive_lattice = 0
    batch_size = int(a1_evaluation["batch_size"])
    for start in range(0, sample_count, batch_size):
        selected_counts = counts[start : start + batch_size].to(device)
        blueprint = ParentBlueprintBatch.from_node_counts(selected_counts, device=device)
        try:
            generated = sampler.sample(
                blueprint,
                steps=int(a1_evaluation["reverse_steps"]),
                initialization_generator=initialization_generator,
                categorical_generator=categorical_generator,
                continuous_generator=continuous_generator,
                continuous_mode=str(a1_evaluation["continuous_mode"]),
                time_grid=str(a1_evaluation["time_grid"]),
            )
        except (SamplingFailure, RuntimeError, FloatingPointError, ValueError) as error:
            failures += int(selected_counts.numel())
            for offset, node_count in enumerate(selected_counts.detach().cpu().tolist()):
                records.append(
                    {
                        "sample_index": start + offset,
                        "node_count": int(node_count),
                        "sampling_failed": True,
                        "failure_type": type(error).__name__,
                    }
                )
            continue
        graph_count = selected_counts.numel()
        observed = dense_token_counts(generated.element_tokens, generated.batch, graph_count)
        exact = (observed == generated.composition_counts).all(dim=1)
        exact_composition += int(exact.sum())
        terminal_masks += int(generated.diagnostics.masked_count[-1])
        determinant = torch.linalg.det(generated.lattice)
        finite = torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (determinant > 0)
        positive_lattice += int(finite.sum())
        elements += element_histogram(generated.element_tokens.cpu())
        volume = determinant / selected_counts
        distance = minimum_periodic_distances(
            generated.fractional_coordinates,
            generated.lattice,
            generated.batch,
        )
        volumes.append(volume.cpu())
        distances.append(distance.cpu())
        batch_formulas = formula_keys(generated.element_tokens, generated.batch, graph_count)
        formulas.extend(batch_formulas)
        for offset in range(int(graph_count)):
            records.append(
                {
                    "sample_index": start + offset,
                    "node_count": int(selected_counts[offset].detach().cpu()),
                    "sampling_failed": False,
                    "exact_composition": bool(exact[offset].detach().cpu()),
                    "finite_positive_lattice": bool(finite[offset].detach().cpu()),
                    "volume_per_atom": float(volume[offset].detach().cpu()),
                    "minimum_distance": float(distance[offset].detach().cpu()),
                    "formula_key": batch_formulas[offset],
                }
            )
    if failures or not volumes or not distances:
        return {"samples": sample_count, "sampling_failures": failures}, records
    volume_all = torch.cat(volumes)
    distance_all = torch.cat(distances)
    count_classes = max(int(counts.max()), int(node_prior.support.max())) + 1
    count_histogram = torch.bincount(counts, minlength=count_classes)
    prior_histogram = torch.zeros(count_classes, dtype=torch.float64)
    prior_histogram[node_prior.support] = node_prior.probabilities
    points = int(a1_evaluation["wasserstein_quantile_points"])
    return (
        {
            "samples": sample_count,
            "sampling_failures": failures,
            "terminal_masks": terminal_masks,
            "exact_composition_fraction": exact_composition / sample_count,
            "finite_positive_lattice_fraction": positive_lattice / sample_count,
            "minimum_distance_fraction_at_0_5_angstrom": float((distance_all >= 0.5).double().mean()),
            "normalized_nearest_neighbor_wasserstein": quantile_wasserstein(
                distance_all, reference["minimum_distance"], points=points
            )
            / robust_scale(reference["minimum_distance"]),
            "normalized_volume_wasserstein": quantile_wasserstein(
                volume_all, reference["volume_per_atom"], points=points
            )
            / robust_scale(reference["volume_per_atom"]),
            "element_marginal_jsd": jensen_shannon(elements, reference["element_histogram"]),
            "node_count_jsd": jensen_shannon(count_histogram, prior_histogram),
            "formula_uniqueness_fraction": len(set(formulas)) / len(formulas),
        },
        records,
    )


def _successful_paired_values(
    base_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if len(base_records) != len(candidate_records):
        raise ValueError("base and candidate record counts differ")
    base_values: list[float] = []
    candidate_values: list[float] = []
    failed_pairs = 0
    for base, candidate in zip(base_records, candidate_records, strict=True):
        if int(base["sample_index"]) != int(candidate["sample_index"]):
            raise ValueError("base and candidate sample indices are not paired")
        if bool(base.get("sampling_failed")) or bool(candidate.get("sampling_failed")):
            failed_pairs += 1
            continue
        base_value = base.get(metric)
        candidate_value = candidate.get(metric)
        if not isinstance(base_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            raise ValueError(f"paired records lack metric {metric}")
        base_values.append(float(base_value))
        candidate_values.append(float(candidate_value))
    if not base_values:
        raise ValueError("paired bootstrap requires at least one successful pair")
    return (
        torch.tensor(base_values, dtype=torch.float64),
        torch.tensor(candidate_values, dtype=torch.float64),
        failed_pairs,
    )


def _paired_bootstrap_w1_delta(
    base_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    reference_values: torch.Tensor,
    *,
    metric: str,
    points: int,
    scale: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    base_values, candidate_values, failed_pairs = _successful_paired_values(
        base_records,
        candidate_records,
        metric,
    )
    generator = torch.Generator().manual_seed(seed)
    sample_count = int(base_values.numel())
    deltas = torch.empty(bootstrap_samples, dtype=torch.float64)
    for index in range(bootstrap_samples):
        sampled = torch.randint(sample_count, (sample_count,), generator=generator)
        base_w1 = quantile_wasserstein(base_values[sampled], reference_values, points=points) / scale
        candidate_w1 = quantile_wasserstein(candidate_values[sampled], reference_values, points=points) / scale
        deltas[index] = candidate_w1 - base_w1
    quantiles = torch.quantile(
        deltas,
        torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64),
    )
    return {
        "metric": metric,
        "bootstrap_samples": bootstrap_samples,
        "paired_successful_samples": sample_count,
        "failed_pairs": failed_pairs,
        "mean_delta": float(deltas.mean()),
        "p025_delta": float(quantiles[0]),
        "p50_delta": float(quantiles[1]),
        "p975_delta": float(quantiles[2]),
        "probability_delta_le_zero": float((deltas <= 0.0).double().mean()),
        "boundary": "paired generated-sample bootstrap against the fixed validation reference distribution",
    }


def _paired_free_generation_bootstrap(
    base_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    reference: dict[str, torch.Tensor],
    *,
    points: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "normalized_volume_wasserstein_delta": _paired_bootstrap_w1_delta(
            base_records,
            candidate_records,
            reference["volume_per_atom"],
            metric="volume_per_atom",
            points=points,
            scale=robust_scale(reference["volume_per_atom"]),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "normalized_nearest_neighbor_wasserstein_delta": _paired_bootstrap_w1_delta(
            base_records,
            candidate_records,
            reference["minimum_distance"],
            metric="minimum_distance",
            points=points,
            scale=robust_scale(reference["minimum_distance"]),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
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
    base_backbone, stage_b_metadata, base_metadata = _load_stage_c_base_backbone(
        args.base_checkpoint,
        device=device,
    )
    candidate_protocol = _read_json(args.candidate_protocol) if args.candidate_protocol is not None else None
    candidate_backbone, checkpoint_summary = _load_correctness_backbone(
        args.checkpoint,
        base_metadata,
        device=device,
        use_ema=bool(args.use_checkpoint_ema),
        candidate_protocol=candidate_protocol,
    )
    precision = str(_training_config(base_metadata).get("precision", "fp32"))
    candidate_training = _training_config(base_metadata)
    candidate_standardization = _standardization(base_metadata)
    if checkpoint_summary.get("schema") == A_V2_CHECKPOINT_SCHEMA:
        checkpoint_metadata = checkpoint_summary["checkpoint_metadata"]
        candidate_training = dict(checkpoint_metadata["training"])
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
        _diffusion_from_config(candidate_backbone, candidate_standardization, candidate_training),
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
    base_records: list[dict[str, Any]] | None = None
    candidate_records: list[dict[str, Any]] | None = None
    if args.record_free_generation_samples:
        generation_base, base_records = _evaluate_generation_retention_with_records(
            base_backbone,
            node_prior,
            _standardization(base_metadata),
            _training_config(base_metadata),
            evaluation,
            reference,
            composition,
            device=device,
        )
        generation_candidate, candidate_records = _evaluate_generation_retention_with_records(
            candidate_backbone,
            node_prior,
            candidate_standardization,
            candidate_training,
            evaluation,
            reference,
            composition,
            device=device,
        )
    else:
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
            candidate_standardization,
            candidate_training,
            evaluation,
            reference,
            composition,
            device=device,
        )
    free_generation: dict[str, Any] = {
        "evaluation": evaluation,
        "validation_indices": indices.tolist(),
        "base": generation_base,
        "candidate": generation_candidate,
        "candidate_minus_base": _generation_delta(generation_candidate, generation_base),
    }
    if base_records is not None and candidate_records is not None:
        free_generation["reference_records"] = _reference_records(reference, indices)
        free_generation["base_records"] = base_records
        free_generation["candidate_records"] = candidate_records
        free_generation["paired_bootstrap"] = _paired_free_generation_bootstrap(
            base_records,
            candidate_records,
            reference,
            points=int(evaluation["wasserstein_quantile_points"]),
            bootstrap_samples=int(args.paired_bootstrap_samples),
            seed=int(args.paired_bootstrap_seed),
        )
    output = {
        "schema": "gaugeflow.generated_state_replay_correctness_evaluation.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_ema_used": bool(args.use_checkpoint_ema),
        "candidate_protocol": None if args.candidate_protocol is None else str(args.candidate_protocol),
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
        "free_generation": free_generation,
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
