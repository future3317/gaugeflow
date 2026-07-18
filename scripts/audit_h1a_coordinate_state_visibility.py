"""Audit whether the coordinate DSM target is visible from an unlabeled state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.quotient_score import (
    factorized_translation_quotient_scaled_score,
)
from gaugeflow.production.schedules import ExponentialTorusNoiseSchedule
from gaugeflow.production.state_projection import project_translation_state


def type_preserving_cycle_permutation(
    elements: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """Cycle each repeated ``(graph, species)`` block without Python loops."""
    if elements.ndim != 1 or batch.shape != elements.shape:
        raise ValueError("elements and batch must be aligned rank-one tensors")
    key = batch * 119 + elements
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    count = order.numel()
    if count == 0:
        return order
    start = torch.ones(count, dtype=torch.bool, device=order.device)
    start[1:] = sorted_key[1:] != sorted_key[:-1]
    start_index = torch.where(start)[0]
    group = start.cumsum(0) - 1
    next_position = torch.arange(count, device=order.device) + 1
    end = torch.ones(count, dtype=torch.bool, device=order.device)
    end[:-1] = sorted_key[:-1] != sorted_key[1:]
    next_position[end] = start_index[group[end]]
    permutation = torch.empty_like(order)
    permutation[order] = order[next_position]
    if not torch.equal(elements[permutation], elements) or not torch.equal(
        batch[permutation], batch
    ):
        raise RuntimeError("constructed permutation is not graph/type preserving")
    return permutation


def _relative_target_metrics(
    left: torch.Tensor, right: torch.Tensor
) -> dict[str, float]:
    left_energy = float(left.square().sum())
    right_energy = float(right.square().sum())
    difference_energy = float((left - right).square().sum())
    denominator = max(0.5 * (left_energy + right_energy), torch.finfo(left.dtype).tiny)
    cosine_denominator = math.sqrt(max(left_energy * right_energy, 0.0))
    return {
        "target_left_mse_per_component": left_energy / left.numel(),
        "target_right_mse_per_component": right_energy / right.numel(),
        "target_difference_mse_per_component": difference_energy / left.numel(),
        "target_relative_difference": math.sqrt(difference_energy / denominator),
        "target_cosine": (
            float((left * right).sum()) / cosine_denominator
            if cosine_denominator > 0.0
            else 1.0
        ),
    }


@torch.no_grad()
def run_audit(
    protocol: dict[str, Any],
    source_protocol: dict[str, Any],
    dataset: PackedAlexP1Dataset,
    *,
    device: torch.device,
) -> dict[str, Any]:
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(protocol["validation_seed"])),
    )[: int(protocol["validation_graphs"])]
    training = source_protocol["training"]
    results: list[dict[str, float]] = []
    graph_total = 0
    nontrivial_graph_total = 0
    for time_offset, time_value in enumerate(protocol["times"]):
        left_targets: list[torch.Tensor] = []
        right_targets: list[torch.Tensor] = []
        time_graphs = 0
        time_nontrivial = 0
        generator = torch.Generator(device=device).manual_seed(
            int(protocol["noise_seed"]) + time_offset
        )
        for start in range(0, indices.numel(), int(protocol["batch_size"])):
            selected = indices[start : start + int(protocol["batch_size"])]
            packed = Batch.from_data_list(
                [dataset[int(index)] for index in selected]
            ).to(device)
            graphs = int(packed.num_graphs)
            schedule = ExponentialTorusNoiseSchedule(
                sigma_min=float(training["coordinate_sigma_min"]),
                sigma_max=float(training["coordinate_sigma_max"]),
            )
            time = packed.lattice.new_full((graphs,), float(time_value))
            clean = project_translation_state(
                packed.frac_coords, packed.batch, graphs
            )
            sigma = schedule.sigma(time)
            noise = torch.randn(
                clean.shape,
                dtype=clean.dtype,
                device=device,
                generator=generator,
            )
            noisy = clean + sigma[packed.batch, None] * noise
            permutation = type_preserving_cycle_permutation(
                packed.atom_types, packed.batch
            )
            changed_graph = torch.zeros(graphs, dtype=torch.bool, device=device)
            changed_node = permutation != torch.arange(
                permutation.numel(), device=device
            )
            changed_graph[packed.batch[changed_node]] = True
            keep = changed_graph[packed.batch]
            left = factorized_translation_quotient_scaled_score(
                noisy - clean, sigma, packed.batch, graphs
            )
            right = factorized_translation_quotient_scaled_score(
                noisy - clean[permutation], sigma, packed.batch, graphs
            )
            left_targets.append(left[keep].cpu())
            right_targets.append(right[keep].cpu())
            time_graphs += graphs
            time_nontrivial += int(changed_graph.sum())
        left_all = torch.cat(left_targets).double()
        right_all = torch.cat(right_targets).double()
        metrics = _relative_target_metrics(left_all, right_all)
        metrics.update(
            {
                "time": float(time_value),
                "graphs": float(time_graphs),
                "graphs_with_repeated_species": float(time_nontrivial),
                "repeated_species_graph_fraction": time_nontrivial / time_graphs,
            }
        )
        results.append(metrics)
        graph_total = time_graphs
        nontrivial_graph_total = time_nontrivial
    maximum = max(value["target_relative_difference"] for value in results)
    threshold = float(protocol["acceptance"]["target_relative_difference_max"])
    invariant = maximum <= threshold
    return {
        "protocol": protocol["protocol"],
        "source_protocol": protocol["source_protocol"],
        "representative": protocol["representative"],
        "visible_state_max_abs_difference": 0.0,
        "endpoint_set_difference": 0.0,
        "graphs": graph_total,
        "graphs_with_repeated_species": nontrivial_graph_total,
        "time_resolved": results,
        "maximum_target_relative_difference": maximum,
        "checks": {
            "visible_state_identity": True,
            "endpoint_set_identity": True,
            "translation_only_target_representative_invariance": invariant,
        },
        "qualified": invariant,
        "decision": (
            "row_label_nuisance_not_detected"
            if invariant
            else "translation_only_target_is_not_state_visible_repair_joint_quotient"
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    source_protocol = load_json_object(args.source_protocol)
    if (
        protocol.get("protocol") != "h1a_coordinate_state_visibility_audit_v1"
        or source_protocol.get("protocol") != protocol["source_protocol"]
        or canonical_json_hash(source_protocol) != protocol["source_protocol_sha256"]
    ):
        raise ValueError("coordinate state-visibility protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate state-visibility cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    result = run_audit(
        protocol,
        source_protocol,
        PackedAlexP1Dataset(args.cache_root, "val"),
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
