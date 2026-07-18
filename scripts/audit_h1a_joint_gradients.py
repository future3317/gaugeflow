"""Audit time-resolved H1a multi-head gradient scale and interference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import load_json_object
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.runtime import load_tensor_free_ema_runtime

_HEAD_PREFIXES = {
    "coordinate": (
        "coordinate_control_gate.",
        "coordinate_edge_encoder.",
        "coordinate_carrier.",
        "coordinate_carrier_head.",
    ),
    "element": ("element_head.",),
    "lattice": ("volume_head.", "shape_head."),
    "condition": ("gauge_atlas.", "geometry_query_encoder."),
}


def _parameter_group(name: str) -> str:
    for group, prefixes in _HEAD_PREFIXES.items():
        if name.startswith(prefixes):
            return group
    return "shared"


def _gradient_statistics(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor | None]]:
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = {group: 0.0 for group in (*_HEAD_PREFIXES, "shared", "all")}
    shared: dict[str, torch.Tensor | None] = {}
    for (name, _), gradient in zip(named_parameters, gradients, strict=True):
        shared[name] = gradient if _parameter_group(name) == "shared" else None
        if gradient is None:
            continue
        value = float(gradient.detach().float().square().sum())
        squared[_parameter_group(name)] += value
        squared["all"] += value
    return ({key: math.sqrt(value) for key, value in squared.items()}, shared)


def _shared_cosine(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> float:
    dot = 0.0
    left_energy = 0.0
    right_energy = 0.0
    for name in left:
        left_value = left[name]
        right_value = right[name]
        if left_value is None or right_value is None:
            continue
        left_float = left_value.detach().float()
        right_float = right_value.detach().float()
        dot += float((left_float * right_float).sum())
        left_energy += float(left_float.square().sum())
        right_energy += float(right_float.square().sum())
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return dot / math.sqrt(left_energy * right_energy)


def _mean_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {
            key: _mean_values([value[key] for value in values])
            for key in first
        }
    return sum(float(value) for value in values) / len(values)


def _mean_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty gradient audit")
    return {
        key: _mean_values([record[key] for record in records])
        for key in records[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = load_json_object(arguments.protocol)
    if protocol.get("protocol") != "h1a_joint_gradient_audit_v1":
        raise ValueError("unexpected H1a gradient-audit protocol")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    checkpoint = (
        arguments.run_root
        / f"seed_{int(protocol['seed'])}"
        / f"checkpoint_step_{int(protocol['source_checkpoint_step']):08d}.pt"
    )
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=str(protocol["source_protocol"]),
        protocol_sha256=str(protocol["source_protocol_sha256"]),
    )
    runtime.model.train()
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    data_spec = protocol["data"]
    dataset = PackedAlexP1Dataset(arguments.cache_root, str(data_spec["split"]))
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(data_spec["subset_seed"]))
    )[: int(data_spec["graphs"])]
    packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad
    ]
    time_results: list[dict[str, Any]] = []
    for time_index, time_value in enumerate(protocol["times"]):
        repeats: list[dict[str, Any]] = []
        for repeat in range(int(data_spec["noise_repeats"])):
            generator = torch.Generator(device=device).manual_seed(
                int(data_spec["noise_seed"]) + 1000 * time_index + repeat
            )
            time = packed.lattice.new_full((graphs,), float(time_value))
            output = diffusion(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time,
                generator=generator,
            )
            losses = {
                "coordinate": output.coordinate_loss,
                "element": output.element_loss,
                "volume": output.volume_loss,
                "shape": output.shape_loss,
            }
            gradient_norms: dict[str, dict[str, float]] = {}
            shared_gradients: dict[str, dict[str, torch.Tensor | None]] = {}
            for loss_index, (name, loss) in enumerate(losses.items()):
                norms, shared = _gradient_statistics(
                    loss,
                    named_parameters,
                    retain_graph=loss_index < len(losses) - 1,
                )
                gradient_norms[name] = norms
                shared_gradients[name] = shared
            coordinate_target = output.noisy.coordinate_scaled_score_target
            zero_coordinate_loss = float(coordinate_target.float().square().mean())
            repeats.append(
                {
                    "loss": {name: float(loss.detach()) for name, loss in losses.items()},
                    "gradient_norm": gradient_norms,
                    "coordinate_shared_cosine": {
                        name: _shared_cosine(shared_gradients["coordinate"], shared_gradients[name])
                        for name in ("element", "volume", "shape")
                    },
                    "coordinate_zero_baseline_loss": zero_coordinate_loss,
                    "masked_fraction": float(output.masked_fraction.detach()),
                }
            )
        time_results.append({"time": float(time_value), **_mean_records(repeats)})
    result = {
        "protocol": protocol["protocol"],
        "source_protocol": protocol["source_protocol"],
        "source_checkpoint_step": int(protocol["source_checkpoint_step"]),
        "graphs": graphs,
        "parameter_count_by_group": {
            group: sum(
                parameter.numel()
                for name, parameter in named_parameters
                if _parameter_group(name) == group
            )
            for group in (*_HEAD_PREFIXES, "shared")
        },
        "time_resolved": time_results,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
