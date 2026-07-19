"""Audit whether trained H1a induced angular slots specialize or collapse."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from evaluate_h1a_p1_protocol import _validation_losses_for_runtime
from torch import nn
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.edge_query_angular_kernel import (
    InducedEdgeQueryAngularKernel,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _effective_rank(matrix: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(matrix.double())
    energy = singular.square()
    if float(energy.sum()) <= 1.0e-30:
        return 0.0
    probability = energy / energy.sum()
    return float(torch.exp(-(probability * probability.clamp_min(1.0e-30).log()).sum()))


def _assignment_variation(
    probability: torch.Tensor, group: torch.Tensor, groups: int
) -> float:
    global_mean = probability.mean(dim=0)
    mass = torch.bincount(group, minlength=groups).to(probability)
    sums = probability.new_zeros((groups, probability.shape[1]))
    sums.index_add_(0, group, probability)
    means = sums / mass.clamp_min(1.0)[:, None]
    weights = mass / mass.sum().clamp_min(1.0)
    return float(
        (weights[:, None] * (means - global_mean).square()).sum().sqrt().cpu()
    )


def _slot_metrics(
    module: InducedEdgeQueryAngularKernel,
    inputs: tuple[Any, ...],
    *,
    source: torch.Tensor,
    distance: torch.Tensor,
    element_tokens: torch.Tensor,
) -> dict[str, Any]:
    statistics = module.slot_statistics(
        inputs[0], inputs[1], inputs[2], inputs[3], int(inputs[4])
    )
    probability = statistics.probability.float()
    entropy = -(
        probability * probability.clamp_min(1.0e-30).log()
    ).sum(dim=-1)
    occupancy = probability.mean(dim=0)
    representation = torch.cat(
        (
            statistics.scalar.flatten(2),
            statistics.vector.flatten(2),
            statistics.stf2.flatten(2),
        ),
        dim=-1,
    ).float()
    slot_matrix = representation.permute(1, 0, 2).reshape(
        representation.shape[1], -1
    )
    normalized = torch.nn.functional.normalize(slot_matrix, dim=-1, eps=1.0e-12)
    cosine = normalized @ normalized.T
    off_diagonal = ~torch.eye(
        cosine.shape[0], dtype=torch.bool, device=cosine.device
    )
    centered = representation - representation.mean(dim=1, keepdim=True)
    centered_matrix = centered.permute(1, 0, 2).reshape(centered.shape[1], -1)
    direction = inputs[2].float()
    directional = torch.einsum("er,ed->rd", probability, direction)
    directional = directional / probability.sum(dim=0).clamp_min(1.0e-12)[:, None]
    shell_boundaries = distance.new_tensor([2.0, 3.0, 4.0, 6.0])
    distance_shell = torch.bucketize(distance, shell_boundaries)
    source_element = element_tokens[source].clamp(max=118)
    return {
        "normalized_assignment_entropy": float(
            (entropy.mean() / math.log(probability.shape[1])).cpu()
        ),
        "effective_slot_count": float(entropy.exp().mean().cpu()),
        "maximum_global_slot_mass": float(occupancy.max().cpu()),
        "occupancy": occupancy.cpu().tolist(),
        "slot_representation_effective_rank": _effective_rank(centered_matrix),
        "mean_absolute_inter_slot_cosine": float(
            cosine[off_diagonal].abs().mean().cpu()
        ),
        "element_assignment_variation": _assignment_variation(
            probability, source_element, 119
        ),
        "distance_shell_assignment_variation": _assignment_variation(
            probability, distance_shell, 5
        ),
        "mean_slot_directional_polarization": float(
            torch.linalg.vector_norm(directional, dim=-1).mean().cpu()
        ),
    }


class _ZeroAngularKernel(InducedEdgeQueryAngularKernel):
    """Diagnostic-only branch ablation; never serialized into production."""

    def __init__(self, reference: InducedEdgeQueryAngularKernel) -> None:
        super().__init__(
            reference.edge_dim,
            reference.channels,
            slots=reference.slots,
            slot_chunk=reference.slot_chunk,
        )

    def forward(
        self,
        edge_state: torch.Tensor,
        edge_target: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_envelope: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        return edge_state.new_zeros((edge_state.shape[0], self.output_dim))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    audit = protocol.get("slot_audit")
    if (
        protocol.get("status_before_run") != "frozen_not_run"
        or protocol.get("model", {}).get("angular_operator") != "induced_slots"
        or not isinstance(audit, dict)
    ):
        raise ValueError("slot audit requires a frozen induced-slot protocol")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("slot-audit cache manifest mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal slot audit requires CUDA")
    protocol_hash = canonical_json_hash(protocol)
    seed = int(protocol["training"]["seeds"][0])
    run = args.run_root / f"seed_{seed}"
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(audit["selection_seed"])),
    )[: int(audit["graphs"])]
    packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    checkpoint_results: dict[str, Any] = {}
    final_runtime = None
    for checkpoint_step in audit["checkpoints"]:
        checkpoint = run / f"checkpoint_step_{int(checkpoint_step):08d}.pt"
        runtime = load_tensor_free_ema_runtime(
            checkpoint,
            device,
            protocol_name=str(protocol["protocol"]),
            protocol_sha256=protocol_hash,
        )
        diffusion = TensorFreeHybridDiffusion(
            runtime.model,
            runtime.lattice_standardizer,
            coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
            coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
            minimum_time=float(runtime.training_config["minimum_time"]),
            maximum_time=float(runtime.training_config["maximum_time"]),
        )
        by_time: dict[str, Any] = {}
        for time_index, time_value in enumerate(audit["times"]):
            time = packed.frac_coords.new_full((graphs,), float(time_value))
            noisy = diffusion.noise_clean_batch(
                packed.atom_types,
                packed.frac_coords,
                packed.lattice,
                packed.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=time,
                generator=torch.Generator(device=device).manual_seed(
                    int(audit["noise_seed"]) + time_index
                ),
            )
            lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
                blueprint.fractional_to_cartesian
            )
            edges = periodic_radius_multigraph(
                noisy.fractional_coordinates,
                lattice,
                packed.batch,
                cutoff=float(protocol["model"]["radial_cutoff_angstrom"]),
            )
            captured: list[dict[str, Any]] = []
            handles: list[Any] = []
            for block in runtime.model.blocks:
                module = block.angular_moments
                if not isinstance(module, InducedEdgeQueryAngularKernel):
                    raise TypeError("checkpoint does not contain induced slots")

                def capture(
                    observed: nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    expected: InducedEdgeQueryAngularKernel = module,
                ) -> None:
                    if observed is not expected:
                        raise RuntimeError("slot hook observed the wrong module")
                    captured.append(
                        _slot_metrics(
                            expected,
                            inputs,
                            source=edges.source,
                            distance=edges.distance,
                            element_tokens=noisy.element_tokens,
                        )
                    )

                handles.append(module.register_forward_pre_hook(capture))
            condition = noisy.log_volume.new_zeros((graphs, 18))
            present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                runtime.model(
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
            for handle in handles:
                handle.remove()
            if len(captured) != len(runtime.model.blocks):
                raise RuntimeError("slot audit did not observe every message block")
            by_time[str(time_value)] = captured
        checkpoint_results[str(checkpoint_step)] = by_time
        if int(checkpoint_step) == int(protocol["training"]["steps"]):
            final_runtime = runtime

    if final_runtime is None:
        raise ValueError("slot audit omits the final checkpoint")
    validation_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(8501)
    )[: int(protocol["evaluation"]["validation_graphs"])]
    full_validation = _validation_losses_for_runtime(
        final_runtime,
        dataset,
        validation_indices,
        device=device,
        seed=int(protocol["evaluation"]["validation_noise_seed"]),
    )
    original_kernels: list[nn.Module] = []
    for block in final_runtime.model.blocks:
        original_kernels.append(block.angular_moments)
        if not isinstance(block.angular_moments, InducedEdgeQueryAngularKernel):
            raise TypeError("branch ablation requires an induced-slot checkpoint")
        block.angular_moments = _ZeroAngularKernel(block.angular_moments).to(device)
    ablated_validation = _validation_losses_for_runtime(
        final_runtime,
        dataset,
        validation_indices,
        device=device,
        seed=int(protocol["evaluation"]["validation_noise_seed"]),
    )
    for block, module in zip(final_runtime.model.blocks, original_kernels, strict=True):
        block.angular_moments = module

    final_records = [
        record
        for layers in checkpoint_results[str(protocol["training"]["steps"])].values()
        for record in layers
    ]
    checks = {
        "assignment_entropy": min(
            record["normalized_assignment_entropy"] for record in final_records
        )
        >= float(audit["normalized_assignment_entropy_min"]),
        "effective_slots": min(
            record["effective_slot_count"] for record in final_records
        )
        >= float(audit["effective_slot_count_min"]),
        "slot_mass": max(
            record["maximum_global_slot_mass"] for record in final_records
        )
        <= float(audit["maximum_global_slot_mass_max"]),
        "representation_rank": min(
            record["slot_representation_effective_rank"] for record in final_records
        )
        >= float(audit["slot_representation_effective_rank_min"]),
        "inter_slot_cosine": max(
            record["mean_absolute_inter_slot_cosine"] for record in final_records
        )
        <= float(audit["mean_absolute_inter_slot_cosine_max"]),
    }
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "checkpoints": checkpoint_results,
        "final_collapse_checks": checks,
        "slots_noncollapsed": all(checks.values()),
        "final_full_validation_coordinate": full_validation["coordinate"],
        "final_ablated_validation_coordinate": ablated_validation["coordinate"],
        "induced_branch_ablation_relative_change": (
            ablated_validation["coordinate"] / full_validation["coordinate"] - 1.0
        ),
        "optimizer_steps": 0,
        "tensor_candidates": full_validation["tensor_candidate_count"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
