"""Evaluate one frozen three-seed H1a P1 tensor-free protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


@torch.no_grad()
def _validation_losses(
    checkpoint: Path,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    protocol_name: str,
    protocol_sha256: str,
    batch_size: int = 16,
) -> dict[str, float]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_fractional_sigma_max=float(
            runtime.training_config["coordinate_fractional_sigma_max"]
        ),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {name: 0.0 for name in ("total", "element", "coordinate", "volume", "shape")}
    graphs_seen = 0
    candidate_count = 0
    use_bf16 = (
        runtime.training_config.get("precision") == "bf16" and device.type == "cuda"
    )
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
        packed = Batch.from_data_list(
            [dataset[int(index)] for index in selected]
        ).to(device)
        graphs = int(packed.num_graphs)
        counts = torch.bincount(packed.batch, minlength=graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts, dtype=packed.frac_coords.dtype, device=device
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            output = diffusion(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                generator=generator,
            )
        values = {
            "total": output.loss,
            "element": output.element_loss,
            "coordinate": output.coordinate_loss,
            "volume": output.volume_loss,
            "shape": output.shape_loss,
        }
        for name, value in values.items():
            totals[name] += float(value.float().cpu()) * graphs
        candidate_count += int(output.prediction.gauge_atlas.effective_frame_count.sum())
        graphs_seen += graphs
    result = {name: value / graphs_seen for name, value in totals.items()}
    result["tensor_candidate_count"] = float(candidate_count)
    return result


@torch.no_grad()
def _sample_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device,
    samples: int,
    steps: int,
    seed: int,
    protocol_name: str,
    protocol_sha256: str,
    minimum_distance_threshold: float,
    batch_size: int = 8,
) -> dict[str, Any]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_fractional_sigma_max=float(
            runtime.training_config["coordinate_fractional_sigma_max"]
        ),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    count_generator = torch.Generator().manual_seed(seed)
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
    node_counts = runtime.node_count_prior.sample(
        samples, generator=count_generator, device=device
    )
    failures = 0
    masks = 0
    finite_positive = 0
    minimum_distances: list[float] = []
    volume_per_atom: list[float] = []
    for start in range(0, samples, batch_size):
        counts = node_counts[start : start + batch_size]
        blueprint = ParentBlueprintBatch.from_node_counts(
            counts, dtype=torch.float32, device=device
        )
        try:
            generated = sampler.sample(
                blueprint,
                steps=steps,
                generator=sample_generator,
                stochastic=True,
                time_grid="uniform_log_alpha",
            )
        except SamplingFailure:
            failures += counts.numel()
            continue
        masks += int(generated.diagnostics.masked_count[-1])
        determinant = torch.linalg.det(generated.lattice)
        valid = torch.isfinite(generated.lattice).all(dim=(-2, -1)) & (
            determinant > 0.0
        )
        finite_positive += int(valid.sum())
        volume_per_atom.extend((determinant / counts).cpu().tolist())
        edges = periodic_radius_multigraph(
            generated.fractional_coordinates,
            generated.lattice,
            generated.batch,
            cutoff=8.0,
        )
        for graph in range(counts.numel()):
            selected = generated.batch[edges.target] == graph
            minimum_distances.append(
                float(edges.distance[selected].min().cpu())
                if bool(selected.any())
                else math.inf
            )
    distance = torch.tensor(minimum_distances, dtype=torch.float64)
    volume = torch.tensor(volume_per_atom, dtype=torch.float64)
    return {
        "samples": samples,
        "sampling_failures": failures,
        "terminal_masks": masks,
        "finite_positive_lattices_fraction": finite_positive / samples,
        "minimum_distance_threshold_angstrom": minimum_distance_threshold,
        "minimum_distance_guardrail_fraction": float(
            (distance >= minimum_distance_threshold).double().mean()
        ),
        "minimum_distance_quantiles_angstrom": torch.quantile(
            distance, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        ).tolist(),
        "volume_per_atom_quantiles_angstrom3": torch.quantile(
            volume, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        ).tolist(),
    }


def _training_log_is_finite(path: Path, expected_step: int) -> bool:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return bool(records) and int(records[-1]["step"]) == expected_step and all(
        math.isfinite(float(value))
        for record in records
        for key, value in record.items()
        if key not in {"step"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("E:/DATA/T2C-Flow/processed/gaugeflow_h1a_v1/p1_structure_cache_v1"),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = load_json_object(arguments.protocol)
    protocol_name = str(protocol.get("protocol"))
    if not protocol_name.startswith("h1a_p1_"):
        raise ValueError("unexpected H1a P1 protocol")
    protocol_sha256 = canonical_json_hash(protocol)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    evaluation = protocol["fixed_evaluation"]
    training = protocol["training"]
    acceptance = protocol["acceptance"]
    distance_guardrail = acceptance.get("minimum_distance_guardrail")
    distance_threshold = (
        float(distance_guardrail["threshold_angstrom"])
        if isinstance(distance_guardrail, dict)
        else 0.5
    )
    dataset = PackedAlexP1Dataset(arguments.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    seed_results: dict[str, Any] = {}
    ratios: list[float] = []
    coordinate_ratios: list[float] = []
    for seed in training["seeds"]:
        run = arguments.run_root / f"seed_{seed}"
        validation_curve: dict[str, dict[str, float]] = {}
        for checkpoint_step in training["checkpoint_steps"]:
            checkpoint = run / f"checkpoint_step_{int(checkpoint_step):08d}.pt"
            validation_curve[str(checkpoint_step)] = _validation_losses(
                checkpoint,
                dataset,
                indices,
                device=device,
                seed=int(evaluation["validation_seed"]) + 1,
                protocol_name=protocol_name,
                protocol_sha256=protocol_sha256,
            )
        initial = validation_curve["0"]
        final = validation_curve[str(int(training["steps"]))]
        final_checkpoint = run / f"checkpoint_step_{int(training['steps']):08d}.pt"
        ratio = final["total"] / initial["total"]
        ratios.append(ratio)
        coordinate_ratios.append(final["coordinate"] / initial["coordinate"])
        samples = _sample_checkpoint(
            final_checkpoint,
            device=device,
            samples=int(evaluation["samples_per_seed"]),
            steps=int(evaluation["sampler_steps"]),
            seed=int(evaluation["sampling_seed"]) + int(seed),
            protocol_name=protocol_name,
            protocol_sha256=protocol_sha256,
            minimum_distance_threshold=distance_threshold,
        )
        seed_results[str(seed)] = {
            "initial_validation": initial,
            "final_validation": final,
            "validation_curve": validation_curve,
            "final_over_initial_total": ratio,
            "final_over_initial_coordinate": coordinate_ratios[-1],
            "training_log_finite": _training_log_is_finite(
                run / "training_metrics.jsonl", int(training["steps"])
            ),
            "sampling": samples,
        }
    checks = {
        "training_finite": all(
            bool(value["training_log_finite"]) for value in seed_results.values()
        ),
        "mean_validation_ratio": sum(ratios) / len(ratios)
        <= float(acceptance["mean_final_validation_total_over_initial_max"]),
        "all_seed_validation_ratio": max(ratios)
        <= float(acceptance["three_seed_final_validation_total_over_initial_max"]),
        "terminal_masks": sum(
            int(value["sampling"]["terminal_masks"])
            for value in seed_results.values()
        )
        == int(acceptance["terminal_masks"]),
        "sampling_failures": sum(
            int(value["sampling"]["sampling_failures"])
            for value in seed_results.values()
        )
        == int(acceptance["sampling_failures"]),
        "finite_positive_lattices": all(
            value["sampling"]["finite_positive_lattices_fraction"]
            >= float(acceptance["finite_positive_lattices_fraction"])
            for value in seed_results.values()
        ),
        "tensor_bypass": all(
            value["final_validation"]["tensor_candidate_count"]
            == float(acceptance["tensor_candidates_when_absent"])
            for value in seed_results.values()
        ),
    }
    coordinate_mean_bound = acceptance.get(
        "mean_final_coordinate_over_initial_max"
    )
    coordinate_seed_bound = acceptance.get(
        "three_seed_final_coordinate_over_initial_max"
    )
    if coordinate_mean_bound is not None and coordinate_seed_bound is not None:
        checks["mean_coordinate_ratio"] = sum(coordinate_ratios) / len(
            coordinate_ratios
        ) <= float(coordinate_mean_bound)
        checks["all_seed_coordinate_ratio"] = max(coordinate_ratios) <= float(
            coordinate_seed_bound
        )
    if isinstance(distance_guardrail, dict):
        fractions = [
            float(value["sampling"]["minimum_distance_guardrail_fraction"])
            for value in seed_results.values()
        ]
        checks["minimum_distance_each_seed"] = min(fractions) >= float(
            distance_guardrail["each_seed_fraction_min"]
        )
        checks["minimum_distance_aggregate"] = sum(fractions) / len(fractions) >= float(
            distance_guardrail["aggregate_fraction_min"]
        )
    qualified = all(checks.values())
    result = {
        "protocol": protocol_name,
        "protocol_sha256": protocol_sha256,
        "seed_results": seed_results,
        "mean_final_over_initial_total": sum(ratios) / len(ratios),
        "mean_final_over_initial_coordinate": sum(coordinate_ratios)
        / len(coordinate_ratios),
        "checks": checks,
        "qualified": qualified,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not qualified:
        raise RuntimeError("H1a P1 protocol failed its frozen acceptance checks")


if __name__ == "__main__":
    main()
