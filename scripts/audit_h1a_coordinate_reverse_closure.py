"""Audit coordinate reverse closure from forward-noised real structures."""

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
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.reverse_sampler import quotient_coordinate_reverse_step
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.schedules import ExponentialTorusNoiseSchedule, standard_normal
from gaugeflow.production.state_projection import graph_mean, project_translation_state


def _minimum_distance(
    coordinates: torch.Tensor, lattice: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    edges = periodic_radius_multigraph(coordinates, lattice, batch, cutoff=8.0)
    return scatter(
        edges.distance,
        batch[edges.target],
        dim=0,
        dim_size=lattice.shape[0],
        reduce="min",
    )


def _endpoint_rms(
    estimate: torch.Tensor,
    clean: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    graphs = lattice.shape[0]
    difference = torus_logmap(clean, estimate)
    phase = 2.0 * math.pi * difference
    translation = torch.atan2(
        graph_mean(phase.sin(), batch, graphs),
        graph_mean(phase.cos(), batch, graphs),
    ) / (2.0 * math.pi)
    residual = torus_logmap(translation[batch], difference)
    cartesian = torch.einsum("ni,nij->nj", residual, lattice[batch])
    return scatter(
        cartesian.square().sum(-1), batch, dim=0, dim_size=graphs, reduce="mean"
    ).sqrt()


def _quantiles(values: torch.Tensor) -> list[float]:
    return torch.quantile(
        values.double(), torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64)
    ).tolist()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    protocol = load_json_object(arguments.protocol)
    if protocol.get("protocol") != "h1a_oracle_context_reverse_closure_v1":
        raise ValueError("unexpected coordinate closure protocol")
    device = torch.device(arguments.device)
    runtime = load_tensor_free_ema_runtime(
        arguments.run_root
        / f"seed_{int(protocol['seed'])}"
        / f"checkpoint_step_{int(protocol['source_checkpoint_step']):08d}.pt",
        device,
        protocol_name=str(protocol["source_protocol"]),
        protocol_sha256=str(protocol["source_protocol_sha256"]),
    )
    dataset = PackedAlexP1Dataset(arguments.cache_root, "val")
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(protocol["validation_seed"]))
    )[: int(protocol["validation_graphs"])]
    schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
    )
    results: list[dict[str, Any]] = []
    for time_index, start_time in enumerate(protocol["start_times"]):
        endpoint_values: list[torch.Tensor] = []
        initial_distances: list[torch.Tensor] = []
        final_distances: list[torch.Tensor] = []
        steps = max(1, round(float(start_time) * int(protocol["steps_per_unit_time"])))
        generator = torch.Generator(device=device).manual_seed(
            int(protocol["noise_seed"]) + time_index
        )
        for start in range(0, indices.numel(), int(protocol["batch_size"])):
            selected = indices[start : start + int(protocol["batch_size"])]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts, dtype=packed.frac_coords.dtype, device=device
            )
            clean = project_translation_state(packed.frac_coords, packed.batch, graphs)
            sigma = schedule.sigma(packed.lattice.new_full((graphs,), float(start_time)))
            noise = standard_normal(clean.shape, clean, generator)
            noise = project_translation_state(noise, packed.batch, graphs)
            coordinates = clean + sigma[packed.batch].unsqueeze(-1) * noise
            lattice_state = LatticeVolumeShape.from_lattice(
                packed.lattice, blueprint.fractional_to_cartesian
            )
            condition = packed.lattice.new_zeros((graphs, 18))
            present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
            times = torch.linspace(
                float(start_time), 0.0, steps + 1, dtype=packed.lattice.dtype, device=device
            )
            initial_distances.append(
                _minimum_distance(coordinates, packed.lattice, packed.batch).cpu()
            )
            for step in range(steps):
                time_from = times[step].expand(graphs)
                time_to = times[step + 1].expand(graphs)
                prediction = runtime.model(
                    packed.atom_types,
                    coordinates,
                    lattice_state.log_volume,
                    lattice_state.log_shape,
                    packed.batch,
                    time_from,
                    condition,
                    present,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                )
                coordinates = quotient_coordinate_reverse_step(
                    coordinates,
                    prediction.coordinate_fractional_scaled_score,
                    schedule.variance(time_from),
                    schedule.variance(time_to),
                    packed.batch,
                    graphs,
                    generator=generator,
                    stochastic=bool(protocol["stochastic"]),
                )
            endpoint_values.append(
                _endpoint_rms(coordinates, clean, packed.lattice, packed.batch).cpu()
            )
            final_distances.append(
                _minimum_distance(coordinates, packed.lattice, packed.batch).cpu()
            )
        endpoint = torch.cat(endpoint_values)
        initial_distance = torch.cat(initial_distances)
        final_distance = torch.cat(final_distances)
        results.append(
            {
                "start_time": float(start_time),
                "steps": steps,
                "endpoint_rms_mean_angstrom": float(endpoint.mean()),
                "endpoint_rms_quantiles_angstrom": _quantiles(endpoint),
                "initial_minimum_distance_quantiles_angstrom": _quantiles(initial_distance),
                "final_minimum_distance_quantiles_angstrom": _quantiles(final_distance),
            }
        )
    result = {
        "protocol": protocol["protocol"],
        "source_protocol": protocol["source_protocol"],
        "seed": int(protocol["seed"]),
        "graphs": int(indices.numel()),
        "results": results,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
