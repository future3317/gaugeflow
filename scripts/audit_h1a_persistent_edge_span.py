"""Probe span, layer coupling and moment collisions in the active H1a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.factorized_angular_moments import (
    FactorizedCartesianAngularMoments,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import (
    fractional_tangent_to_cartesian,
    graph_mean,
    sorted_segment_sum,
)


def _environment_directions(name: str) -> torch.Tensor:
    if name == "tetrahedral":
        value = torch.tensor(
            [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
            dtype=torch.float64,
        )
    elif name == "octahedral":
        value = torch.tensor(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=torch.float64,
        )
    elif name == "cubic":
        value = torch.tensor(
            [
                [x, y, z]
                for x in (-1, 1)
                for y in (-1, 1)
                for z in (-1, 1)
            ],
            dtype=torch.float64,
        )
    elif name == "cuboctahedral":
        rows: set[tuple[int, int, int]] = set()
        for zero_axis in range(3):
            for first in (-1, 1):
                for second in (-1, 1):
                    row = [first, second]
                    row.insert(zero_axis, 0)
                    rows.add(tuple(row))
        value = torch.tensor(sorted(rows), dtype=torch.float64)
    elif name == "triangular_prism":
        height = 1.0 / math.sqrt(3.0)
        radius = math.sqrt(2.0 / 3.0)
        value = torch.tensor(
            [
                [
                    radius * math.cos(2.0 * math.pi * index / 3.0),
                    radius * math.sin(2.0 * math.pi * index / 3.0),
                    sign * height,
                ]
                for sign in (-1.0, 1.0)
                for index in range(3)
            ],
            dtype=torch.float64,
        )
    else:
        raise ValueError(f"unknown synthetic environment {name!r}")
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def _analytic_environment(name: str) -> dict[str, Any]:
    direction = _environment_directions(name)
    first = direction.mean(dim=0)
    covariance = torch.einsum("ni,nj->ij", direction, direction) / direction.shape[0]
    second = covariance - torch.eye(3, dtype=torch.float64) / 3.0
    probes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    probes = probes / torch.linalg.vector_norm(probes, dim=-1, keepdim=True)
    fourth = (direction @ probes.T).pow(4).mean(dim=0)
    return {
        "count": direction.shape[0],
        "first_moment_norm": float(torch.linalg.vector_norm(first)),
        "second_stf_norm": float(torch.linalg.matrix_norm(second)),
        "fourth_order_fingerprint": fourth.tolist(),
    }


class _FeatureCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self.block_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.final_carrier: torch.Tensor | None = None
        self.handles = []
        for block in model.blocks:  # type: ignore[attr-defined]
            self.handles.append(block.register_forward_hook(self._block_hook))
        self.handles.append(
            model.coordinate_carrier_mixer.register_forward_pre_hook(  # type: ignore[attr-defined]
                self._carrier_hook
            )
        )

    def _block_hook(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        self.block_outputs.append(output)

    def _carrier_hook(
        self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        self.final_carrier = inputs[0]

    def reset(self) -> None:
        self.block_outputs.clear()
        self.final_carrier = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _center_carrier(
    carrier: torch.Tensor, batch: torch.Tensor, graphs: int
) -> torch.Tensor:
    return carrier - graph_mean(carrier, batch, graphs)[batch]


def _graph_equal_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    graphs: int,
) -> float:
    node_error = (prediction - target).square().mean(dim=-1)
    graph_error = sorted_segment_sum(node_error, batch, graphs) / torch.bincount(
        batch, minlength=graphs
    ).clamp_min(1).to(node_error)
    return float(graph_error.mean())


def _linear_vector_probe(
    carrier: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    graph_limit: int,
) -> dict[str, float | int]:
    mask = batch < graph_limit
    selected_carrier = carrier[mask].detach().cpu().double()
    selected_target = target[mask].detach().cpu().double()
    selected_batch = batch[mask].detach().cpu()
    if selected_carrier.ndim != 3 or selected_carrier.shape[-1] != 3:
        raise ValueError("linear vector probe requires [nodes,channels,3]")
    rows = selected_carrier.permute(0, 2, 1).reshape(-1, selected_carrier.shape[1])
    response = selected_target.reshape(-1)
    counts = torch.bincount(selected_batch, minlength=graph_limit).clamp_min(1)
    row_weight = counts[selected_batch].double().rsqrt().repeat_interleave(3)
    weighted_rows = rows * row_weight[:, None]
    weighted_response = response * row_weight
    solution = torch.linalg.lstsq(
        weighted_rows, weighted_response.unsqueeze(-1), driver="gelsd"
    ).solution.squeeze(-1)
    prediction = torch.einsum("ncd,c->nd", selected_carrier, solution)
    mse = _graph_equal_mse(
        prediction, selected_target, selected_batch, graph_limit
    )
    energy = _graph_equal_mse(
        torch.zeros_like(selected_target),
        selected_target,
        selected_batch,
        graph_limit,
    )
    singular = torch.linalg.svdvals(weighted_rows)
    rank = int((singular > singular.max().clamp_min(1e-30) * 1e-10).sum())
    normalized = singular.square() / singular.square().sum().clamp_min(1e-30)
    effective_rank = float(
        torch.exp(-(normalized * normalized.clamp_min(1e-30).log()).sum())
    )
    return {
        "channels": selected_carrier.shape[1],
        "rank": rank,
        "effective_rank": effective_rank,
        "target_energy": energy,
        "probe_mse": mse,
        "explained_fraction": 1.0 - mse / max(energy, 1e-30),
        "coefficient_norm": float(torch.linalg.vector_norm(solution)),
    }


def _local_strata(
    direction: torch.Tensor,
    target_index: torch.Tensor,
    node_count: int,
    *,
    isotropic_gap: float,
    axial_gap: float,
) -> torch.Tensor:
    dyad = torch.einsum("ei,ej->eij", direction, direction)
    covariance = sorted_segment_sum(dyad, target_index, node_count)
    degree = torch.bincount(target_index, minlength=node_count).clamp_min(1)
    covariance = covariance / degree.to(covariance)[:, None, None]
    eigenvalues = torch.linalg.eigvalsh(covariance)
    gaps = eigenvalues[:, 1:] - eigenvalues[:, :-1]
    total_gap = eigenvalues[:, -1] - eigenvalues[:, 0]
    strata = torch.full((node_count,), 2, dtype=torch.long, device=direction.device)
    strata[gaps.min(dim=-1).values <= axial_gap] = 1
    strata[total_gap <= isotropic_gap] = 0
    return strata


def _stratified_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    strata: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    names = {0: "descriptor_isotropic", 1: "descriptor_axial", 2: "generic"}
    result: dict[str, dict[str, float | int]] = {}
    node_mse = (prediction - target).square().mean(dim=-1)
    target_energy = target.square().mean(dim=-1)
    for value, name in names.items():
        mask = strata == value
        count = int(mask.sum())
        result[name] = {
            "nodes": count,
            "mse": float(node_mse[mask].mean()) if count else 0.0,
            "target_energy": float(target_energy[mask].mean()) if count else 0.0,
        }
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_persistent_edge_causal_attribution_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen persistent-edge audit protocol")
    source = protocol["source"]
    specification = protocol["span_probe"]
    if sha256_file(args.cache_root / "manifest.json") != str(
        source["cache_manifest_sha256"]
    ):
        raise ValueError("persistent-edge span cache mismatch")
    if sha256_file(args.checkpoint) != str(source["checkpoint_sha256"]):
        raise ValueError("persistent-edge span checkpoint mismatch")
    source_protocol_path = Path("configs/gates") / f"{source['training_protocol']}.json"
    source_protocol = load_json_object(source_protocol_path)
    if canonical_json_hash(source_protocol) != str(source["training_protocol_sha256"]):
        raise ValueError("persistent-edge source protocol mismatch")

    device = torch.device(args.device)
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(source["training_protocol"]),
        protocol_sha256=str(source["training_protocol_sha256"]),
    )
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    dataset = PackedAlexP1Dataset(args.cache_root, str(specification["split"]))
    selected = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(specification["selection_seed"])),
    )[: int(specification["graphs"])]
    packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    capture = _FeatureCapture(runtime.model)
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    by_time: dict[str, Any] = {}
    try:
        for time_index, time_value in enumerate(specification["times"]):
            capture.reset()
            fixed_time = packed.lattice.new_full((graphs,), float(time_value))
            generator = torch.Generator(device=device).manual_seed(
                int(specification["noise_seed"]) + time_index
            )
            noisy = diffusion.noise_clean_batch(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=fixed_time,
                generator=generator,
            )
            condition = packed.lattice.new_zeros((graphs, 18))
            present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                prediction = runtime.model(
                    noisy.element_tokens,
                    noisy.fractional_coordinates,
                    noisy.log_volume,
                    noisy.log_shape,
                    packed.batch,
                    noisy.time,
                    condition,
                    present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                )
            if capture.final_carrier is None or len(capture.block_outputs) != len(
                runtime.model.blocks
            ):
                raise RuntimeError("feature hooks did not capture the complete backbone")
            lattice = LatticeVolumeShape(
                noisy.log_volume.float(), noisy.log_shape.float()
            ).lattice(blueprint.fractional_to_cartesian.float())
            cell_scale = torch.exp(noisy.log_volume.float() / 3.0)
            target = fractional_tangent_to_cartesian(
                noisy.coordinate_scaled_score_target.float(), lattice, packed.batch
            ) / cell_scale[packed.batch, None]
            current = prediction.coordinate_cartesian_scaled_score.float() / cell_scale[
                packed.batch, None
            ]
            edges = periodic_radius_multigraph(
                noisy.fractional_coordinates.float(),
                lattice,
                packed.batch,
                cutoff=runtime.model.radial.cutoff,
            )
            position = fractional_tangent_to_cartesian(
                noisy.fractional_coordinates.float(), lattice, packed.batch
            )
            features: dict[str, torch.Tensor] = {
                "final_coordinate_carrier": capture.final_carrier.float()
            }
            degree_scale = torch.bincount(
                edges.target, minlength=packed.num_nodes
            ).clamp_min(1).to(position).rsqrt()
            for layer, (nodes, vectors, edge_state) in enumerate(
                capture.block_outputs, start=1
            ):
                features[f"layer_{layer}_vector"] = _center_carrier(
                    vectors.float(), packed.batch, graphs
                )
                node_position = nodes.float()[:, :, None] * position[:, None, :]
                features[f"layer_{layer}_node_position"] = _center_carrier(
                    node_position, packed.batch, graphs
                )
                edge_direction = sorted_segment_sum(
                    edge_state.float()[:, :, None]
                    * edges.direction.float()[:, None, :],
                    edges.target,
                    packed.num_nodes,
                )
                edge_direction = edge_direction * degree_scale[:, None, None]
                features[f"layer_{layer}_edge_direction"] = _center_carrier(
                    edge_direction, packed.batch, graphs
                )
            probes: dict[str, dict[str, Any]] = {}
            for name, carrier in features.items():
                probes[name] = {
                    str(state_count): _linear_vector_probe(
                        carrier, target, packed.batch, int(state_count)
                    )
                    for state_count in specification["state_counts"]
                }
            strata = _local_strata(
                edges.direction.float(),
                edges.target,
                packed.num_nodes,
                isotropic_gap=float(specification["isotropic_gap_max"]),
                axial_gap=float(specification["axial_single_gap_max"]),
            )
            by_time[str(time_value)] = {
                "current_model_mse": _graph_equal_mse(
                    current, target, packed.batch, graphs
                ),
                "target_energy": _graph_equal_mse(
                    torch.zeros_like(target), target, packed.batch, graphs
                ),
                "linear_probes": probes,
                "local_symmetry_strata": _stratified_error(current, target, strata),
            }
            print(
                json.dumps(
                    {
                        "time": time_value,
                        "current_model_mse": by_time[str(time_value)][
                            "current_model_mse"
                        ],
                        "final_carrier_64": probes["final_coordinate_carrier"]["64"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        capture.close()

    synthetic = {
        name: _analytic_environment(name)
        for name in protocol["synthetic_environments"]["families"]
    }
    left_name, right_name = protocol["synthetic_environments"]["matched_pair"]
    left = torch.tensor(
        synthetic[left_name]["fourth_order_fingerprint"], dtype=torch.float64
    )
    right = torch.tensor(
        synthetic[right_name]["fourth_order_fingerprint"], dtype=torch.float64
    )
    matched_difference = float(torch.linalg.vector_norm(left - right))
    constant_state_outputs: dict[str, dict[str, float]] = {}
    for layer, block in enumerate(runtime.model.blocks, start=1):
        module: FactorizedCartesianAngularMoments = block.angular_moments
        constant_state_outputs[str(layer)] = {}
        generator = torch.Generator(device=device).manual_seed(9200 + layer)
        base_state = torch.randn(
            (1, module.edge_dim), device=device, generator=generator
        )
        for name in (left_name, right_name):
            direction = _environment_directions(name).to(device=device, dtype=torch.float32)
            output = module(
                base_state.expand(direction.shape[0], -1),
                torch.zeros(direction.shape[0], dtype=torch.long, device=device),
                direction,
                torch.ones((direction.shape[0], 1), device=device),
                1,
            )
            constant_state_outputs[str(layer)][name] = float(
                torch.linalg.vector_norm(output)
            )
    moment_tolerance = float(protocol["synthetic_environments"]["moment_tolerance"])
    synthetic_checks = {
        "matched_count": synthetic[left_name]["count"] == synthetic[right_name]["count"],
        "matched_first_moments": max(
            synthetic[left_name]["first_moment_norm"],
            synthetic[right_name]["first_moment_norm"],
        )
        <= moment_tolerance,
        "matched_second_moments": max(
            synthetic[left_name]["second_stf_norm"],
            synthetic[right_name]["second_stf_norm"],
        )
        <= moment_tolerance,
        "fourth_order_separated": matched_difference
        >= float(protocol["synthetic_environments"]["fourth_order_separation_min"]),
    }
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "checkpoint": str(args.checkpoint),
        "selected_validation_indices": selected.tolist(),
        "by_time": by_time,
        "synthetic_environments": synthetic,
        "matched_environment_pair": {
            "left": left_name,
            "right": right_name,
            "fourth_order_fingerprint_distance": matched_difference,
            "constant_edge_state_operator_output_norm": constant_state_outputs,
            "checks": synthetic_checks,
        },
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("persistent-edge span audit output already exists")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
