"""Evaluate frozen Stage-B physical learning and GaugeFlow-base retention."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from evaluate_gaugeflow_base_a1 import dense_token_counts, reference_statistics

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import EmpiricalNodeCountPrior, ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
)
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.generation_metrics import (
    element_histogram,
    formula_keys,
    jensen_shannon,
    minimum_periodic_distances,
    quantile_wasserstein,
    robust_scale,
)
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.matpes_data import collate_matpes_records
from gaugeflow.production.matpes_index import IndexedMatPESDataset
from gaugeflow.production.physical_checkpointing import load_physical_ema_for_evaluation
from gaugeflow.production.physical_evaluation import (
    finalize_physical_metrics,
    physical_metric_sums,
)
from gaugeflow.production.physical_pretraining import (
    PhysicalRepresentationModel,
    load_functional_physical_normalizer,
)
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.teacher_feature_cache import MatPESTeacherFeatureCache
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--a1-evaluation-protocol", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-run", type=Path, required=True)
    parser.add_argument("--matpes-index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def evaluate_physical_checkpoint(
    model: PhysicalRepresentationModel,
    dataset: IndexedMatPESDataset,
    normalizer: Any,
    vocabulary: dict[str, int],
    teacher_dim: int,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    statistics = torch.zeros((len(vocabulary), 9), dtype=torch.float64, device=device)
    for start in range(0, len(dataset), batch_size):
        records = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_matpes_records(
            records,
            functional_vocabulary=vocabulary,
            teacher_dim=teacher_dim,
        ).to(device)
        target = normalizer.normalize(batch.targets, batch.functional_index, batch.batch)
        prediction = model(
            batch.element_tokens,
            batch.fractional_coordinates,
            batch.lattice,
            batch.batch,
            batch.functional_index,
        )
        statistics += physical_metric_sums(
            prediction,
            target,
            batch.batch,
            batch.functional_index,
            len(vocabulary),
        )
    return finalize_physical_metrics(statistics.cpu(), vocabulary)


@torch.no_grad()
def evaluate_generation_retention(
    backbone: HybridCrystalDenoiser,
    node_prior: EmpiricalNodeCountPrior,
    standardization: dict[str, Any],
    a1_training: dict[str, Any],
    a1_evaluation: dict[str, Any],
    reference: dict[str, torch.Tensor],
    composition_model: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
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
        except (SamplingFailure, RuntimeError, FloatingPointError, ValueError):
            failures += int(selected_counts.numel())
            continue
        graph_count = selected_counts.numel()
        observed = dense_token_counts(generated.element_tokens, generated.batch, graph_count)
        exact_composition += int((observed == generated.composition_counts).all(dim=1).sum())
        terminal_masks += int(generated.diagnostics.masked_count[-1])
        determinant = torch.linalg.det(generated.lattice)
        positive_lattice += int((torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (determinant > 0)).sum())
        elements += element_histogram(generated.element_tokens.cpu())
        volumes.append((determinant / selected_counts).cpu())
        distances.append(
            minimum_periodic_distances(
                generated.fractional_coordinates,
                generated.lattice,
                generated.batch,
            ).cpu()
        )
        formulas.extend(formula_keys(generated.element_tokens, generated.batch, graph_count))
    if failures or not volumes or not distances:
        return {"samples": sample_count, "sampling_failures": failures}
    volume = torch.cat(volumes)
    distance = torch.cat(distances)
    count_classes = max(int(counts.max()), int(node_prior.support.max())) + 1
    count_histogram = torch.bincount(counts, minlength=count_classes)
    prior_histogram = torch.zeros(count_classes, dtype=torch.float64)
    prior_histogram[node_prior.support] = node_prior.probabilities
    points = int(a1_evaluation["wasserstein_quantile_points"])
    return {
        "samples": sample_count,
        "sampling_failures": failures,
        "terminal_masks": terminal_masks,
        "exact_composition_fraction": exact_composition / sample_count,
        "finite_positive_lattice_fraction": positive_lattice / sample_count,
        "minimum_distance_fraction_at_0_5_angstrom": float((distance >= 0.5).double().mean()),
        "normalized_nearest_neighbor_wasserstein": quantile_wasserstein(
            distance, reference["minimum_distance"], points=points
        )
        / robust_scale(reference["minimum_distance"]),
        "normalized_volume_wasserstein": quantile_wasserstein(
            volume, reference["volume_per_atom"], points=points
        )
        / robust_scale(reference["volume_per_atom"]),
        "element_marginal_jsd": jensen_shannon(elements, reference["element_histogram"]),
        "node_count_jsd": jensen_shannon(count_histogram, prior_histogram),
        "formula_uniqueness_fraction": len(set(formulas)) / len(formulas),
    }


def _head_losses(metrics: dict[str, Any]) -> dict[str, float]:
    losses = {
        name: float(metrics[key]) ** 2
        for name, key in (
            ("energy", "normalized_energy_rmse"),
            ("force", "normalized_force_rmse"),
            ("stress", "normalized_kelvin_stress_rmse"),
        )
        if metrics[key] is not None
    }
    if metrics["teacher_feature_cosine"] is not None:
        losses["feature"] = 1.0 - float(metrics["teacher_feature_cosine"])
    return losses


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    a1_protocol = load_json_object(args.a1_evaluation_protocol)
    if protocol.get("protocol") != "stage_b_physical_representation_v1":
        raise ValueError("unexpected Stage-B protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Stage-B evaluation requires CUDA")
    prerequisites = protocol["prerequisites"]
    implementation_paths = {
        "evaluator_sha256": Path(__file__),
        "physical_evaluation_sha256": Path(inspect.getsourcefile(physical_metric_sums) or ""),
    }
    for name, path in implementation_paths.items():
        if not path.is_file() or sha256_file(path) != prerequisites[name]:
            raise ValueError(f"Stage-B evaluation implementation hash mismatch: {name}")
    if sha256_file(args.a1_checkpoint) != prerequisites["a1_checkpoint_sha256"]:
        raise ValueError("Stage-B evaluator received the wrong A1 checkpoint")
    a1_metadata = read_production_checkpoint_metadata(args.a1_checkpoint)
    model_config = a1_metadata["model_config"]
    a1_training = a1_metadata["training_config"]
    standardization = a1_metadata["lattice_standardization"]
    backbone = HybridCrystalDenoiser(**model_config).to(device)
    _, node_prior, _ = load_production_checkpoint(
        args.a1_checkpoint,
        model=backbone,
        map_location=device,
    )
    normalizer, vocabulary = load_functional_physical_normalizer(args.normalizer)
    calibration = IndexedMatPESDataset(
        args.matpes_index,
        "calibration",
        teacher_feature_cache=args.teacher_cache,
    )
    if calibration.teacher_feature_cache is None:
        raise AssertionError("Stage-B calibration data lacks teacher features")
    feature_cache: MatPESTeacherFeatureCache = calibration.teacher_feature_cache
    model = PhysicalRepresentationModel(
        backbone,
        teacher_dim=feature_cache.feature_dim,
        functional_count=len(vocabulary),
    ).to(device)
    ema = ExponentialMovingAverage(model, float(protocol["training"]["ema_decay"]))
    protocol_sha256 = canonical_json_hash(protocol)
    checkpoints: dict[str, Any] = {}
    for step in protocol["training"]["checkpoint_steps"]:
        checkpoint = args.physical_run / f"checkpoint_step_{int(step):08d}.pt"
        observed_step, metadata = load_physical_ema_for_evaluation(
            checkpoint,
            model=model,
            ema=ema,
            map_location=device,
        )
        if observed_step != int(step) or metadata.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Stage-B checkpoint step or protocol identity mismatch")
        model.eval()
        checkpoints[str(step)] = evaluate_physical_checkpoint(
            model,
            calibration,
            normalizer,
            vocabulary,
            feature_cache.feature_dim,
            batch_size=64,
            device=device,
        )

    alex_validation = PackedAlexP1Dataset(args.alex_cache, "val")
    a1_evaluation = a1_protocol["evaluation"]
    validation_indices = torch.randperm(
        len(alex_validation),
        generator=torch.Generator().manual_seed(int(a1_evaluation["validation_seed"])),
    )[: int(a1_evaluation["validation_graphs"])]
    reference = reference_statistics(
        alex_validation,
        validation_indices,
        batch_size=int(a1_evaluation["batch_size"]),
        device=device,
    )
    composition_model = load_qualified_composition_model(
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
        expected_checkpoint_sha256=str(a1_protocol["composition_checkpoint_sha256"]),
    )
    generation = evaluate_generation_retention(
        model.backbone,
        node_prior,
        standardization,
        a1_training,
        a1_evaluation,
        reference,
        composition_model,
        device=device,
    )

    initial = checkpoints[str(protocol["training"]["checkpoint_steps"][0])]["aggregate"]
    final = checkpoints[str(protocol["training"]["checkpoint_steps"][-1])]["aggregate"]
    initial_heads = _head_losses(initial)
    final_heads = _head_losses(final)
    aggregate_head_ratios = {
        name: final_heads[name] / initial_heads[name] for name in initial_heads
    }
    per_functional_head_ratios = {}
    for functional in vocabulary:
        initial_functional = _head_losses(
            checkpoints[str(protocol["training"]["checkpoint_steps"][0])]["per_functional"][functional]
        )
        final_functional = _head_losses(
            checkpoints[str(protocol["training"]["checkpoint_steps"][-1])]["per_functional"][functional]
        )
        per_functional_head_ratios[functional] = {
            name: final_functional[name] / initial_functional[name]
            for name in initial_functional
        }
    all_head_ratios = list(aggregate_head_ratios.values()) + [
        ratio
        for functional in per_functional_head_ratios.values()
        for ratio in functional.values()
    ]
    acceptance = protocol["acceptance"]
    checks = {
        "physical_composite": final["equal_head_composite_loss"]
        / initial["equal_head_composite_loss"]
        <= acceptance["physical_learning"]["calibration_composite_loss_relative_to_step0_max"],
        "each_physical_head": max(all_head_ratios)
        <= acceptance["physical_learning"]["each_supervised_head_relative_to_step0_max"],
        "pbe_teacher_feature_cosine": checkpoints[
            str(protocol["training"]["checkpoint_steps"][-1])
        ]["per_functional"]["PBE"]["teacher_feature_cosine"]
        >= acceptance["physical_learning"]["pbe_teacher_feature_cosine_min"],
        "sampling_failures": generation.get("sampling_failures")
        == acceptance["core_safety"]["sampling_failures"],
        "exact_composition": generation.get("exact_composition_fraction")
        == acceptance["core_safety"]["exact_composition_fraction"],
        "positive_lattice": generation.get("finite_positive_lattice_fraction")
        == acceptance["core_safety"]["positive_lattice_fraction"],
        "minimum_distance": generation.get("minimum_distance_fraction_at_0_5_angstrom", 0.0)
        >= acceptance["core_safety"]["minimum_distance_fraction_at_0_5_angstrom_min"],
        "nearest_neighbor_retention": generation.get(
            "normalized_nearest_neighbor_wasserstein", float("inf")
        )
        <= acceptance["a1_retention"]["normalized_nearest_neighbor_wasserstein_max"],
        "volume_retention": generation.get("normalized_volume_wasserstein", float("inf"))
        <= acceptance["a1_retention"]["normalized_volume_wasserstein_max"],
        "element_retention": generation.get("element_marginal_jsd", float("inf"))
        <= acceptance["a1_retention"]["element_marginal_jsd_max"],
        "node_count_retention": generation.get("node_count_jsd", float("inf"))
        <= acceptance["a1_retention"]["node_count_jsd_max"],
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_canonical_sha256": protocol_sha256,
        "physical_checkpoints": checkpoints,
        "physical_head_loss_ratios": {
            "aggregate": aggregate_head_ratios,
            "per_functional": per_functional_head_ratios,
        },
        "generation_retention": generation,
        "generation_validation_indices": validation_indices.tolist(),
        "checks": checks,
        "qualified": qualified,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-B evaluation: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not qualified:
        raise RuntimeError("Stage-B physical representation failed its frozen Gate")


if __name__ == "__main__":
    main()
