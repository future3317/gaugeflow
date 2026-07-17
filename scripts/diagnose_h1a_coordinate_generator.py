"""Time-resolved diagnosis of the failed H1a coordinate generator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import load_json_object
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.manifold import torus_logmap
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.reverse_sampler import TensorFreeReverseSampler
from gaugeflow.production.runtime import TensorFreeEmaRuntime, load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import graph_mean


def _fixed_indices(length: int, count: int, seed: int) -> torch.Tensor:
    if count < 1 or count > length:
        raise ValueError("diagnostic subset size is outside the dataset")
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed))[:count]


def _minimum_distances(
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    edges = periodic_radius_multigraph(
        fractional_coordinates, lattice, batch, cutoff=8.0
    )
    graphs = lattice.shape[0]
    if edges.target.numel() == 0:
        return lattice.new_full((graphs,), math.inf)
    edge_graph = batch[edges.target]
    return scatter(edges.distance, edge_graph, dim=0, dim_size=graphs, reduce="min")


def _quantiles(values: torch.Tensor) -> list[float]:
    probabilities = values.new_tensor([0.0, 0.01, 0.05, 0.5, 0.95, 1.0])
    return torch.quantile(values, probabilities).double().cpu().tolist()


def _translation_aligned_endpoint_rms(
    estimate: torch.Tensor,
    clean: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    graphs = lattice.shape[0]
    difference = torus_logmap(clean, estimate)
    phase = 2.0 * math.pi * difference
    circular_mean = torch.atan2(
        graph_mean(phase.sin(), batch, graphs),
        graph_mean(phase.cos(), batch, graphs),
    ) / (2.0 * math.pi)
    residual = torus_logmap(circular_mean[batch], difference)
    cartesian = torch.einsum("ni,nij->nj", residual, lattice[batch])
    return scatter(
        cartesian.square().sum(dim=-1),
        batch,
        dim=0,
        dim_size=graphs,
        reduce="mean",
    ).sqrt()


@torch.no_grad()
def _score_calibration(
    runtime: TensorFreeEmaRuntime,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    specification: dict[str, Any],
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_fractional_sigma_max=float(
            runtime.training_config["coordinate_fractional_sigma_max"]
        ),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    batch_size = int(specification["batch_size"])
    results: list[dict[str, float]] = []
    use_bf16 = (
        runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    )
    for time_value in specification["times"]:
        totals = {
            "nodes": 0.0,
            "squared_error": 0.0,
            "target_energy": 0.0,
            "prediction_energy": 0.0,
            "dot": 0.0,
            "endpoint_squared": 0.0,
            "oracle_endpoint_squared": 0.0,
            "graphs": 0.0,
        }
        generator = torch.Generator(device=device).manual_seed(
            int(specification["noise_seed"]) + round(float(time_value) * 1_000_000)
        )
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(
                device
            )
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts, dtype=packed.frac_coords.dtype, device=device
            )
            time = packed.lattice.new_full((graphs,), float(time_value))
            noisy = diffusion.noise_clean_batch(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time,
                generator=generator,
            )
            condition = time.new_zeros((graphs, 18))
            condition_present = torch.zeros(
                (graphs, 1), dtype=torch.bool, device=device
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                prediction = runtime.model(
                    noisy.element_tokens,
                    noisy.fractional_coordinates,
                    noisy.log_volume,
                    noisy.log_shape,
                    packed.batch,
                    time,
                    condition,
                    condition_present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                )
            predicted = prediction.coordinate_fractional_score.float()
            target = noisy.coordinate_score_target.float()
            error = predicted - target
            variance = diffusion.coordinate_schedule.variance(time)[packed.batch]
            clean = packed.frac_coords
            estimate = noisy.fractional_coordinates + variance.unsqueeze(-1) * predicted
            oracle_estimate = noisy.fractional_coordinates + variance.unsqueeze(-1) * target
            endpoint_rms = _translation_aligned_endpoint_rms(
                estimate, clean, packed.lattice, packed.batch
            )
            oracle_rms = _translation_aligned_endpoint_rms(
                oracle_estimate, clean, packed.lattice, packed.batch
            )
            totals["nodes"] += float(target.shape[0])
            totals["squared_error"] += float(error.square().sum())
            totals["target_energy"] += float(target.square().sum())
            totals["prediction_energy"] += float(predicted.square().sum())
            totals["dot"] += float((predicted * target).sum())
            totals["endpoint_squared"] += float(endpoint_rms.square().sum())
            totals["oracle_endpoint_squared"] += float(oracle_rms.square().sum())
            totals["graphs"] += graphs
        target_energy = totals["target_energy"]
        prediction_energy = totals["prediction_energy"]
        results.append(
            {
                "time": float(time_value),
                "score_mse_per_component": totals["squared_error"]
                / (3.0 * totals["nodes"]),
                "zero_score_mse_per_component": target_energy
                / (3.0 * totals["nodes"]),
                "score_explained_fraction": 1.0
                - totals["squared_error"] / target_energy,
                "prediction_to_target_norm": math.sqrt(
                    prediction_energy / target_energy
                ),
                "prediction_target_cosine": totals["dot"]
                / math.sqrt(prediction_energy * target_energy),
                "endpoint_rms_angstrom": math.sqrt(
                    totals["endpoint_squared"] / totals["graphs"]
                ),
                "oracle_endpoint_rms_angstrom": math.sqrt(
                    totals["oracle_endpoint_squared"] / totals["graphs"]
                ),
            }
        )
    return results


@torch.no_grad()
def _reference_geometry(
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    distances: list[torch.Tensor] = []
    for start in range(0, indices.numel(), 64):
        selected = indices[start : start + 64]
        packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(
            device
        )
        distances.append(
            _minimum_distances(packed.frac_coords, packed.lattice, packed.batch).cpu()
        )
    values = torch.cat(distances).double()
    return {
        "graphs": int(values.numel()),
        "threshold_angstrom": threshold,
        "fraction_at_least_threshold": float((values >= threshold).double().mean()),
        "quantiles_angstrom": _quantiles(values),
    }


@torch.no_grad()
def _sampler_sensitivity(
    runtime: TensorFreeEmaRuntime,
    specification: dict[str, Any],
    *,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    samples = int(specification["samples"])
    count_generator = torch.Generator().manual_seed(int(specification["sample_seed"]))
    counts = runtime.node_count_prior.sample(samples, generator=count_generator)
    sampler = TensorFreeReverseSampler(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_fractional_sigma_max=float(
            runtime.training_config["coordinate_fractional_sigma_max"]
        ),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    results: list[dict[str, Any]] = []
    for steps in specification["steps"]:
        distances: list[torch.Tensor] = []
        terminal_coordinate_steps: list[float] = []
        generator = torch.Generator(device=device).manual_seed(
            int(specification["sample_seed"]) + 1
        )
        for start in range(0, samples, 8):
            selected_counts = counts[start : start + 8].to(device)
            blueprint = ParentBlueprintBatch.from_node_counts(
                selected_counts, dtype=torch.float32, device=device
            )
            generated = sampler.sample(
                blueprint,
                steps=int(steps),
                generator=generator,
                stochastic=bool(specification["stochastic"]),
                time_grid=str(specification["time_grid"]),
            )
            distances.append(
                _minimum_distances(
                    generated.fractional_coordinates,
                    generated.lattice,
                    generated.batch,
                ).cpu()
            )
            terminal_coordinate_steps.append(
                float(generated.diagnostics.coordinate_step_rms[-1])
            )
        values = torch.cat(distances).double()
        results.append(
            {
                "steps": int(steps),
                "fraction_at_least_threshold": float(
                    (values >= threshold).double().mean()
                ),
                "minimum_distance_quantiles_angstrom": _quantiles(values),
                "mean_terminal_coordinate_step_rms": sum(terminal_coordinate_steps)
                / len(terminal_coordinate_steps),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = load_json_object(arguments.protocol)
    source_protocol = load_json_object(arguments.source_protocol)
    if (
        protocol.get("protocol") != "h1a_coordinate_diagnostic_v1"
        or source_protocol.get("protocol") != protocol["source_protocol"]
    ):
        raise ValueError("coordinate diagnostic protocol identity mismatch")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    validation = PackedAlexP1Dataset(arguments.cache_root, "val")
    train = PackedAlexP1Dataset(arguments.cache_root, "train")
    calibration_spec = protocol["score_calibration"]
    validation_indices = _fixed_indices(
        len(validation),
        int(calibration_spec["validation_graphs"]),
        int(calibration_spec["validation_seed"]),
    )
    reference_spec = protocol["reference_geometry"]
    train_indices = _fixed_indices(
        len(train),
        int(reference_spec["train_graphs"]),
        int(reference_spec["train_seed"]),
    )
    threshold = float(reference_spec["distance_threshold_angstrom"])
    runtimes: dict[int, TensorFreeEmaRuntime] = {}
    calibration: dict[str, list[dict[str, float]]] = {}
    for seed in protocol["seeds"]:
        checkpoint = (
            arguments.run_root
            / f"seed_{int(seed)}"
            / f"checkpoint_step_{int(protocol['source_checkpoint_step']):08d}.pt"
        )
        runtime = load_tensor_free_ema_runtime(
            checkpoint,
            device,
            protocol_name=str(protocol["source_protocol"]),
            protocol_sha256=str(protocol["source_protocol_sha256"]),
        )
        runtimes[int(seed)] = runtime
        calibration[str(seed)] = _score_calibration(
            runtime,
            validation,
            validation_indices,
            calibration_spec,
            device=device,
        )
    sensitivity_spec = protocol["sampler_sensitivity"]
    result = {
        "protocol": protocol["protocol"],
        "source_protocol": protocol["source_protocol"],
        "score_calibration": calibration,
        "train_reference_geometry": _reference_geometry(
            train, train_indices, device=device, threshold=threshold
        ),
        "sampler_sensitivity": _sampler_sensitivity(
            runtimes[int(sensitivity_spec["seed"])],
            sensitivity_spec,
            device=device,
            threshold=threshold,
        ),
        "decision_boundary": protocol["decision_boundary"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
