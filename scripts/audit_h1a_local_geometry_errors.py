"""Read-only local-error attribution for a frozen H1a coordinate checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from evaluate_h1a_coordinate_pretraining import _rollout_closure
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import fractional_tangent_to_cartesian


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _empty_totals() -> dict[str, float]:
    return {"count": 0.0, "error": 0.0, "target": 0.0, "prediction": 0.0, "dot": 0.0}


def _accumulate(
    totals: dict[str, float], prediction: torch.Tensor, target: torch.Tensor
) -> None:
    if prediction.numel() == 0:
        return
    totals["count"] += float(prediction.numel())
    totals["error"] += float((prediction - target).square().sum())
    totals["target"] += float(target.square().sum())
    totals["prediction"] += float(prediction.square().sum())
    totals["dot"] += float((prediction * target).sum())


def _finalize(totals: dict[str, float]) -> dict[str, float]:
    target = totals["target"]
    prediction = totals["prediction"]
    denominator = math.sqrt(max(target * prediction, 1.0e-30))
    return {
        "count": totals["count"],
        "mse": totals["error"] / max(totals["count"], 1.0),
        "explained_fraction": 1.0 - totals["error"] / max(target, 1.0e-30),
        "prediction_to_target_norm": math.sqrt(prediction / max(target, 1.0e-30)),
        "prediction_target_cosine": totals["dot"] / denominator,
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double()
    right = right.double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator.clamp_min(1.0e-30)).cpu())


def _partial_correlation(
    response: torch.Tensor, predictor: torch.Tensor, control: torch.Tensor
) -> float:
    response = response.double().log1p()
    predictor = predictor.double().log1p()
    control = control.double().log1p()
    design = torch.stack((torch.ones_like(control), control), dim=-1)
    response_residual = response - design @ torch.linalg.lstsq(design, response).solution
    predictor_residual = predictor - design @ torch.linalg.lstsq(design, predictor).solution
    return _correlation(response_residual, predictor_residual)


def _bucket_results(
    totals: list[dict[str, float]], boundaries: list[float]
) -> list[dict[str, Any]]:
    edges = [-math.inf, *boundaries, math.inf]
    return [
        {
            "lower": edges[index],
            "upper": edges[index + 1],
            **_finalize(value),
        }
        for index, value in enumerate(totals)
    ]


@torch.no_grad()
def _audit(
    runtime: Any,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    specification: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    distance_boundaries = [float(value) for value in specification["distance_bins_angstrom"]]
    degree_boundaries = [float(value) for value in specification["degree_bins"]]
    node_count_boundaries = [float(value) for value in specification["node_count_bins"]]
    pair_size = 119 * 119
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    output: dict[str, Any] = {}
    for time_value in specification["times"]:
        distance_totals = [_empty_totals() for _ in range(len(distance_boundaries) + 1)]
        degree_totals = [_empty_totals() for _ in range(len(degree_boundaries) + 1)]
        node_count_totals = [
            _empty_totals() for _ in range(len(node_count_boundaries) + 1)
        ]
        normalized_node_count_totals = [
            _empty_totals() for _ in range(len(node_count_boundaries) + 1)
        ]
        pair_count = torch.zeros(pair_size, dtype=torch.float64)
        pair_error = torch.zeros_like(pair_count)
        pair_target = torch.zeros_like(pair_count)
        pair_prediction = torch.zeros_like(pair_count)
        pair_dot = torch.zeros_like(pair_count)
        graph_values: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "mse",
                "volume_normalized_mse",
                "node_count",
                "mean_degree",
                "self_image_fraction",
                "volume_per_atom",
                "lattice_condition",
                "nearest_cross_site_distance",
            )
        }
        generator = torch.Generator(device=device).manual_seed(
            int(specification["noise_seed"]) + round(float(time_value) * 1_000_000)
        )
        batch_size = int(specification["batch_size"])
        for start in range(0, indices.numel(), batch_size):
            selected = indices[start : start + batch_size]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
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
            present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                prediction = runtime.model(
                    noisy.element_tokens,
                    noisy.fractional_coordinates,
                    noisy.log_volume,
                    noisy.log_shape,
                    packed.batch,
                    time,
                    condition,
                    present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                )
            lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
                blueprint.fractional_to_cartesian
            )
            target = fractional_tangent_to_cartesian(
                noisy.coordinate_scaled_score_target, lattice, packed.batch
            )
            predicted = prediction.coordinate_cartesian_scaled_score.float()
            node_error = (predicted - target).square().sum(dim=-1) / 3.0
            edges = periodic_radius_multigraph(
                noisy.fractional_coordinates, lattice, packed.batch, cutoff=8.0
            )
            degree = torch.bincount(edges.target, minlength=packed.num_nodes).float()
            self_image = (edges.source == edges.target) & edges.image_shift.ne(0).any(dim=-1)

            degree_bucket = torch.bucketize(degree, degree.new_tensor(degree_boundaries))
            for index, totals in enumerate(degree_totals):
                selected_node = degree_bucket == index
                _accumulate(totals, predicted[selected_node], target[selected_node])
            node_count_per_node = counts[packed.batch].float()
            node_count_bucket = torch.bucketize(
                node_count_per_node,
                node_count_per_node.new_tensor(node_count_boundaries),
            )
            for index, totals in enumerate(node_count_totals):
                selected_node = node_count_bucket == index
                _accumulate(totals, predicted[selected_node], target[selected_node])
            volume = torch.det(lattice).abs()
            cell_scale = volume.clamp_min(1.0e-12).pow(1.0 / 3.0)
            normalized_prediction = predicted / cell_scale[packed.batch, None]
            normalized_target = target / cell_scale[packed.batch, None]
            for index, totals in enumerate(normalized_node_count_totals):
                selected_node = node_count_bucket == index
                _accumulate(
                    totals,
                    normalized_prediction[selected_node],
                    normalized_target[selected_node],
                )

            cross = edges.source != edges.target
            cross_source = edges.source[cross]
            cross_target = edges.target[cross]
            direction = edges.direction[cross]
            relative_prediction = (
                (predicted[cross_target] - predicted[cross_source]) * direction
            ).sum(dim=-1)
            relative_target = ((target[cross_target] - target[cross_source]) * direction).sum(dim=-1)
            distance = edges.distance[cross]
            distance_bucket = torch.bucketize(
                distance, distance.new_tensor(distance_boundaries)
            )
            for index, totals in enumerate(distance_totals):
                selected_edge = distance_bucket == index
                _accumulate(
                    totals,
                    relative_prediction[selected_edge],
                    relative_target[selected_edge],
                )

            low = torch.minimum(packed.atom_types[cross_source], packed.atom_types[cross_target])
            high = torch.maximum(packed.atom_types[cross_source], packed.atom_types[cross_target])
            pair = low * 119 + high
            pair_count += torch.bincount(pair, minlength=pair_size).double().cpu()
            pair_error += torch.bincount(
                pair, weights=(relative_prediction - relative_target).square(), minlength=pair_size
            ).double().cpu()
            pair_target += torch.bincount(
                pair, weights=relative_target.square(), minlength=pair_size
            ).double().cpu()
            pair_prediction += torch.bincount(
                pair, weights=relative_prediction.square(), minlength=pair_size
            ).double().cpu()
            pair_dot += torch.bincount(
                pair, weights=relative_prediction * relative_target, minlength=pair_size
            ).double().cpu()

            edge_graph = packed.batch[edges.target]
            cross_graph = packed.batch[cross_target]
            graph_mse = scatter(
                node_error, packed.batch, dim=0, dim_size=graphs, reduce="mean"
            )
            graph_values["mse"].append(graph_mse.cpu())
            graph_values["volume_normalized_mse"].append(
                (graph_mse / cell_scale.square()).cpu()
            )
            graph_values["node_count"].append(counts.float().cpu())
            graph_values["mean_degree"].append(
                scatter(degree, packed.batch, dim=0, dim_size=graphs, reduce="mean").cpu()
            )
            graph_values["self_image_fraction"].append(
                scatter(
                    self_image.float(), edge_graph, dim=0, dim_size=graphs, reduce="mean"
                ).cpu()
            )
            graph_values["volume_per_atom"].append(
                (volume / counts).cpu()
            )
            graph_values["lattice_condition"].append(torch.linalg.cond(lattice).cpu())
            graph_values["nearest_cross_site_distance"].append(
                scatter(distance, cross_graph, dim=0, dim_size=graphs, reduce="min").cpu()
            )

        minimum_pair_count = int(specification["minimum_element_pair_edges"])
        pair_rows: list[dict[str, Any]] = []
        for key in torch.nonzero(pair_count >= minimum_pair_count).flatten().tolist():
            totals = {
                "count": float(pair_count[key]),
                "error": float(pair_error[key]),
                "target": float(pair_target[key]),
                "prediction": float(pair_prediction[key]),
                "dot": float(pair_dot[key]),
            }
            pair_rows.append(
                {
                    "elements": [key // 119, key % 119],
                    **_finalize(totals),
                }
            )
        pair_rows.sort(key=lambda value: float(value["mse"]), reverse=True)
        joined = {key: torch.cat(value) for key, value in graph_values.items()}
        output[str(time_value)] = {
            "distance_bins": _bucket_results(distance_totals, distance_boundaries),
            "degree_bins": _bucket_results(degree_totals, degree_boundaries),
            "node_count_bins": _bucket_results(
                node_count_totals, node_count_boundaries
            ),
            "volume_normalized_node_count_bins": _bucket_results(
                normalized_node_count_totals, node_count_boundaries
            ),
            "top_element_pairs_by_mse": pair_rows[: int(specification["reported_element_pairs"])],
            "graph_error_correlations": {
                key: _correlation(joined["mse"].log1p(), value.log1p())
                for key, value in joined.items()
                if key not in {"mse", "volume_normalized_mse"}
            },
            "volume_normalized_error_correlations": {
                key: _correlation(joined["volume_normalized_mse"].log1p(), value.log1p())
                for key, value in joined.items()
                if key not in {"mse", "volume_normalized_mse"}
            },
            "partial_graph_error_correlations_controlling_node_count": {
                key: _partial_correlation(
                    joined["mse"], value, joined["node_count"]
                )
                for key, value in joined.items()
                if key not in {"mse", "volume_normalized_mse", "node_count"}
            },
        }
    return output


def main() -> None:
    args = _arguments()
    protocol = load_json_object(args.protocol)
    source = load_json_object(args.source_protocol)
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("local-error audit protocol was not frozen before execution")
    if protocol.get("source_protocol") != source.get("protocol"):
        raise ValueError("local-error source protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["cache_manifest_sha256"]
    ):
        raise ValueError("local-error audit cache mismatch")
    device = torch.device(args.device)
    checkpoint = (
        args.run_root
        / f"seed_{int(protocol['seed'])}"
        / f"checkpoint_step_{int(protocol['source_checkpoint_step']):08d}.pt"
    )
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["source_protocol"]),
        protocol_sha256=str(protocol["source_protocol_sha256"]),
    )
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    specification = protocol["local_error"]
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(specification["selection_seed"]))
    )[: int(specification["graphs"])]
    local_error = _audit(runtime, dataset, indices, specification, device=device)
    rollout_specification = {
        "rollout_start_times": protocol["rollout"]["start_times"],
        "rollout_noise_seed": int(protocol["rollout"]["noise_seed"]),
        "rollout_steps": int(protocol["rollout"]["steps"]),
        "rollout_stochastic": bool(protocol["rollout"]["stochastic"]),
    }
    rollout_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(protocol["rollout"]["selection_seed"]))
    )[: int(protocol["rollout"]["graphs"])]
    rollout = _rollout_closure(
        checkpoint,
        dataset,
        rollout_indices,
        rollout_specification,
        device=device,
        protocol_name=str(protocol["source_protocol"]),
        protocol_sha256=str(protocol["source_protocol_sha256"]),
    )
    result = {
        "protocol": protocol["protocol"],
        "source_protocol": protocol["source_protocol"],
        "local_error": local_error,
        "rollout_by_start_time": rollout,
        "decision_rule": protocol["decision_rule"],
        "optimizer_steps": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
