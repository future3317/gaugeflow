"""Evaluate the frozen exact-count GaugeFlow-base A1 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
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
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def reference_statistics(
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    elements = torch.zeros(118, dtype=torch.float64)
    node_counts: list[torch.Tensor] = []
    volumes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for start in range(0, indices.numel(), batch_size):
        selected = dataset.select_model_batch(
            indices[start : start + batch_size],
            device=device,
        )
        graph_count = int(selected.lattice.shape[0])
        counts = torch.bincount(selected.batch, minlength=graph_count)
        elements += element_histogram(selected.atom_types.cpu())
        node_counts.append(counts.cpu())
        volumes.append((torch.linalg.det(selected.lattice) / counts).cpu())
        distances.append(
            minimum_periodic_distances(
                selected.fractional_coordinates,
                selected.lattice,
                selected.batch,
            ).cpu()
        )
    return {
        "element_histogram": elements,
        "node_counts": torch.cat(node_counts),
        "volume_per_atom": torch.cat(volumes),
        "minimum_distance": torch.cat(distances),
    }


def dense_token_counts(
    tokens: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    flat = batch * 118 + tokens
    return torch.bincount(flat, minlength=graph_count * 118).reshape(graph_count, 118)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    protocol: dict[str, Any],
    reference: dict[str, torch.Tensor],
    *,
    composition_checkpoint: Path,
    composition_protocol: Path,
    device: torch.device,
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    metadata = read_production_checkpoint_metadata(checkpoint)
    if (
        metadata.get("protocol") != protocol["training_protocol"]
        or metadata.get("protocol_sha256") != protocol["training_protocol_canonical_sha256"]
    ):
        raise ValueError("A1 checkpoint does not match the frozen training protocol")
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    standardization = metadata.get("lattice_standardization")
    if not all(isinstance(value, dict) for value in (model_config, training_config, standardization)):
        raise ValueError("A1 checkpoint metadata is incomplete")
    assert isinstance(model_config, dict)
    assert isinstance(training_config, dict)
    assert isinstance(standardization, dict)
    if (
        training_config.get("categorical_path") != "orderless_reveal"
        or not bool(training_config.get("composition_conditioning"))
    ):
        raise ValueError("checkpoint is not an exact-count A1 product model")
    model = HybridCrystalDenoiser(**model_config).to(device)
    ema = ExponentialMovingAverage(model, float(training_config["ema_decay"]))
    _, node_prior, _ = load_production_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        map_location=device,
    )
    ema.copy_to(model)
    model.eval()
    composition_model = load_qualified_composition_model(
        composition_checkpoint,
        composition_protocol,
        device=device,
        expected_checkpoint_sha256=str(protocol["composition_checkpoint_sha256"]),
    )
    sampler = TensorFreeReverseSampler(
        model,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
        maximum_time=float(training_config["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_model=composition_model,
    )
    sample_count = int(evaluation["free_samples"])
    count_generator = torch.Generator().manual_seed(int(evaluation["sampling_seed"]))
    counts = node_prior.sample(sample_count, generator=count_generator)
    initialization_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["sampling_seed"]) + 1
    )
    categorical_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["sampling_seed"]) + 2
    )
    continuous_generator = torch.Generator(device=device).manual_seed(
        int(evaluation["sampling_seed"]) + 3
    )
    elements = torch.zeros(118, dtype=torch.float64)
    volumes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    formulas: list[str] = []
    terminal_masks = 0
    failures = 0
    finite_positive = 0
    exact_composition = 0
    batch_size = int(evaluation["batch_size"])
    for start in range(0, sample_count, batch_size):
        selected_counts = counts[start : start + batch_size].to(device)
        blueprint = ParentBlueprintBatch.from_node_counts(selected_counts, device=device)
        try:
            generated = sampler.sample(
                blueprint,
                steps=int(evaluation["reverse_steps"]),
                initialization_generator=initialization_generator,
                categorical_generator=categorical_generator,
                continuous_generator=continuous_generator,
                continuous_mode=str(evaluation["continuous_mode"]),
                time_grid=str(evaluation["time_grid"]),
            )
        except (SamplingFailure, RuntimeError, FloatingPointError, ValueError):
            failures += int(selected_counts.numel())
            continue
        graph_count = int(selected_counts.numel())
        observed = dense_token_counts(generated.element_tokens, generated.batch, graph_count)
        exact_composition += int((observed == generated.composition_counts).all(dim=1).sum())
        terminal_masks += int(generated.diagnostics.masked_count[-1])
        determinant = torch.linalg.det(generated.lattice)
        finite = torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (determinant > 0.0)
        finite_positive += int(finite.sum())
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
    successful = sample_count - failures
    if successful < 1 or not volumes or not distances or not formulas:
        return {
            "checkpoint": str(checkpoint),
            "samples": sample_count,
            "sampling_failures": failures,
            "successful_samples": 0,
        }
    generated_volume = torch.cat(volumes)
    generated_distance = torch.cat(distances)
    count_classes = max(int(counts.max()), int(reference["node_counts"].max())) + 1
    points = int(evaluation["wasserstein_quantile_points"])
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "samples": sample_count,
        "successful_samples": successful,
        "sampling_failures": failures,
        "terminal_masks": terminal_masks,
        "exact_composition_fraction": exact_composition / sample_count,
        "finite_positive_lattice_fraction": finite_positive / sample_count,
        "minimum_distance_fraction_at_0_5_angstrom": float(
            (generated_distance >= 0.5).double().mean()
        ),
        "normalized_nearest_neighbor_wasserstein": quantile_wasserstein(
            generated_distance,
            reference["minimum_distance"],
            points=points,
        )
        / robust_scale(reference["minimum_distance"]),
        "normalized_volume_wasserstein": quantile_wasserstein(
            generated_volume,
            reference["volume_per_atom"],
            points=points,
        )
        / robust_scale(reference["volume_per_atom"]),
        "element_marginal_jsd": jensen_shannon(elements, reference["element_histogram"]),
        "node_count_jsd": jensen_shannon(
            torch.bincount(counts, minlength=count_classes),
            torch.bincount(reference["node_counts"], minlength=count_classes),
        ),
        "formula_uniqueness_fraction": len(set(formulas)) / len(formulas),
        "generated_minimum_distance_quantiles_angstrom": torch.quantile(
            generated_distance.double(),
            torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64),
        ).tolist(),
        "reference_minimum_distance_quantiles_angstrom": torch.quantile(
            reference["minimum_distance"].double(),
            torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64),
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("A1 evaluation protocol was not frozen")
    expected_runner = protocol.get("evaluator_sha256")
    if expected_runner is not None and sha256_file(Path(__file__)) != expected_runner:
        raise ValueError("A1 evaluator changed after protocol freeze")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    evaluation = protocol["evaluation"]
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    reference = reference_statistics(
        dataset,
        indices,
        batch_size=int(evaluation["batch_size"]),
        device=device,
    )
    checkpoint_results = {
        str(step): evaluate_checkpoint(
            args.run_root / f"checkpoint_step_{int(step):08d}.pt",
            protocol,
            reference,
            composition_checkpoint=args.composition_checkpoint,
            composition_protocol=args.composition_protocol,
            device=device,
        )
        for step in evaluation["checkpoints"]
    }
    final = checkpoint_results[str(evaluation["checkpoints"][-1])]
    acceptance = protocol["acceptance"]
    checks = {
        "finite_training_and_sampling": final.get("successful_samples", 0)
        == int(evaluation["free_samples"]),
        "exact_composition": final.get("exact_composition_fraction")
        == float(acceptance["exact_composition_fraction"]),
        "terminal_masks": final.get("terminal_masks") == int(acceptance["terminal_masks"]),
        "sampling_failures": final.get("sampling_failures")
        == int(acceptance["sampling_failures"]),
        "finite_positive_lattice": final.get("finite_positive_lattice_fraction")
        == float(acceptance["finite_positive_lattice_fraction"]),
        "minimum_distance": final.get("minimum_distance_fraction_at_0_5_angstrom", 0.0)
        >= float(acceptance["minimum_distance_fraction_at_0_5_angstrom_min"]),
        "nearest_neighbor_wasserstein": final.get(
            "normalized_nearest_neighbor_wasserstein", float("inf")
        )
        <= float(acceptance["normalized_nearest_neighbor_wasserstein_max"]),
        "volume_wasserstein": final.get("normalized_volume_wasserstein", float("inf"))
        <= float(acceptance["normalized_volume_wasserstein_max"]),
        "element_marginal": final.get("element_marginal_jsd", float("inf"))
        <= float(acceptance["element_marginal_jsd_max"]),
        "node_count": final.get("node_count_jsd", float("inf"))
        <= float(acceptance["node_count_jsd_max"]),
        "formula_uniqueness": final.get("formula_uniqueness_fraction", 0.0)
        >= float(acceptance["formula_uniqueness_fraction_min"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_canonical_sha256": canonical_json_hash(protocol),
        "validation_indices": indices.tolist(),
        "checkpoints": checkpoint_results,
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite A1 result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not qualified:
        raise RuntimeError("GaugeFlow-base A1 failed its frozen evaluation Gate")


if __name__ == "__main__":
    main()
