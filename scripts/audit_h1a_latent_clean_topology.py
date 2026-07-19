"""Audit whether H1a is limited by a noise-corrupted coordination graph.

This is a frozen-checkpoint, zero-optimizer diagnostic. It defines a smooth
clean first-shell field on the production periodic multigraph, probes that
field from final noisy node/edge states, and measures the held-out coordinate
residual explained by an oracle clean-topology carrier. It never changes the
denoiser or authorizes a later Gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.geometry import PeriodicEdges, periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import TensorFreeEmaRuntime, load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import (
    fractional_tangent_to_cartesian,
    graph_mean,
    sorted_segment_sum,
)


@dataclass
class ScalarRidgeAccumulator:
    xtx: torch.Tensor
    xty: torch.Tensor
    xsum: torch.Tensor
    ysum: torch.Tensor
    y2sum: torch.Tensor
    weight: torch.Tensor

    @classmethod
    def create(cls, features: int) -> ScalarRidgeAccumulator:
        return cls(
            xtx=torch.zeros((features, features), dtype=torch.float64),
            xty=torch.zeros(features, dtype=torch.float64),
            xsum=torch.zeros(features, dtype=torch.float64),
            ysum=torch.zeros((), dtype=torch.float64),
            y2sum=torch.zeros((), dtype=torch.float64),
            weight=torch.zeros((), dtype=torch.float64),
        )

    def update(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        edge_graph: torch.Tensor,
        graphs: int,
    ) -> None:
        counts = torch.bincount(edge_graph, minlength=graphs).clamp_min(1).to(features)
        sample_weight = counts.reciprocal()[edge_graph]
        self.xtx += torch.einsum(
            "ef,eg,e->fg", features, features, sample_weight
        ).double().cpu()
        self.xty += torch.einsum("ef,e,e->f", features, target, sample_weight).double().cpu()
        self.xsum += torch.einsum("ef,e->f", features, sample_weight).double().cpu()
        self.ysum += torch.einsum("e,e->", target, sample_weight).double().cpu()
        self.y2sum += torch.einsum("e,e,e->", target, target, sample_weight).double().cpu()
        self.weight += sample_weight.sum().double().cpu()


@dataclass(frozen=True)
class StandardizedRidge:
    mean: torch.Tensor
    scale: torch.Tensor
    coefficient: torch.Tensor
    target_mean: float
    ridge: float
    rank: int
    condition_number: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.mean.to(features)) / self.scale.to(features)
        return standardized @ self.coefficient.to(features) + self.target_mean


@dataclass
class CarrierAccumulator:
    xtx: torch.Tensor
    xty: torch.Tensor

    @classmethod
    def create(cls, channels: int) -> CarrierAccumulator:
        return cls(
            xtx=torch.zeros((channels, channels), dtype=torch.float64),
            xty=torch.zeros(channels, dtype=torch.float64),
        )

    def update(
        self,
        carrier: torch.Tensor,
        residual: torch.Tensor,
        batch: torch.Tensor,
        graphs: int,
    ) -> None:
        counts = torch.bincount(batch, minlength=graphs).clamp_min(1).to(carrier)
        node_weight = counts.reciprocal()[batch]
        self.xtx += torch.einsum(
            "ncd,ned,n->ce", carrier, carrier, node_weight
        ).double().cpu()
        self.xty += torch.einsum(
            "ncd,nd,n->c", carrier, residual, node_weight
        ).double().cpu()


class FinalEdgeCapture:
    """Read-only hook for the final production message block."""

    def __init__(self, model: nn.Module) -> None:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) < 1:
            raise TypeError("topology audit requires the production block stack")
        self.values: dict[str, torch.Tensor] = {}
        self.handle = blocks[-1].register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        del module
        self.values = {
            "nodes": output[0].detach(),
            "edge_state": output[2].detach(),
            "source": inputs[2].detach(),
            "target": inputs[3].detach(),
            "direction": inputs[4].detach(),
            "radial": inputs[6].detach(),
        }

    def close(self) -> None:
        self.handle.remove()


def _segment_min(values: torch.Tensor, target: torch.Tensor, nodes: int) -> torch.Tensor:
    output = values.new_full((nodes,), torch.inf)
    output.scatter_reduce_(0, target, values, reduce="amin", include_self=True)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("periodic topology has a node without a neighbour")
    return output


def smooth_first_shell_probability(
    distance: torch.Tensor,
    target: torch.Tensor,
    nearest: torch.Tensor,
    *,
    multiplier: float,
    relative_width: float,
    cutoff: float,
) -> torch.Tensor:
    """Return a smooth adaptive first-shell field for directed image edges."""
    if multiplier <= 1.0 or relative_width <= 0.0 or cutoff <= 0.0:
        raise ValueError("invalid smooth coordination parameters")
    local_scale = nearest[target].clamp_min(1.0e-6)
    shell = torch.sigmoid(
        (multiplier * local_scale - distance) / (relative_width * local_scale)
    )
    inside = distance < cutoff
    envelope = torch.where(
        inside,
        0.5 * (torch.cos(math.pi * distance / cutoff) + 1.0),
        torch.zeros_like(distance),
    )
    return shell * envelope


def _clean_distance_on_edges(
    clean_coordinates: torch.Tensor,
    clean_lattice: torch.Tensor,
    batch: torch.Tensor,
    edges: PeriodicEdges,
) -> torch.Tensor:
    relative = (
        clean_coordinates[edges.target]
        - clean_coordinates[edges.source]
        + edges.image_shift.to(clean_coordinates)
    )
    displacement = torch.einsum(
        "ni,nij->nj", relative, clean_lattice[batch[edges.target]]
    )
    return torch.linalg.vector_norm(displacement, dim=-1)


def _edge_graph_sum(
    values: torch.Tensor, edge_graph: torch.Tensor, graphs: int
) -> torch.Tensor:
    output = values.new_zeros((graphs,) + values.shape[1:])
    output.index_add_(0, edge_graph, values)
    return output


def topology_fields(
    clean_coordinates: torch.Tensor,
    clean_lattice: torch.Tensor,
    noisy_edges: PeriodicEdges,
    batch: torch.Tensor,
    *,
    cutoff: float,
    multiplier: float,
    relative_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clean/noisy fields on noisy edges and clean-mass coverage."""
    nodes = clean_coordinates.shape[0]
    graphs = clean_lattice.shape[0]
    clean_edges = periodic_radius_multigraph(
        clean_coordinates, clean_lattice, batch, cutoff=cutoff
    )
    clean_nearest = _segment_min(clean_edges.distance, clean_edges.target, nodes)
    noisy_nearest = _segment_min(noisy_edges.distance, noisy_edges.target, nodes)
    clean_full = smooth_first_shell_probability(
        clean_edges.distance,
        clean_edges.target,
        clean_nearest,
        multiplier=multiplier,
        relative_width=relative_width,
        cutoff=cutoff,
    )
    clean_on_noisy = smooth_first_shell_probability(
        _clean_distance_on_edges(clean_coordinates, clean_lattice, batch, noisy_edges),
        noisy_edges.target,
        clean_nearest,
        multiplier=multiplier,
        relative_width=relative_width,
        cutoff=cutoff,
    )
    noisy_field = smooth_first_shell_probability(
        noisy_edges.distance,
        noisy_edges.target,
        noisy_nearest,
        multiplier=multiplier,
        relative_width=relative_width,
        cutoff=cutoff,
    )
    clean_graph = batch[clean_edges.target]
    noisy_graph = batch[noisy_edges.target]
    denominator = _edge_graph_sum(clean_full, clean_graph, graphs).clamp_min(1.0e-12)
    numerator = _edge_graph_sum(clean_on_noisy, noisy_graph, graphs)
    coverage = numerator / denominator
    if bool((coverage > 1.0002).any()):
        raise RuntimeError("noisy edge set duplicates clean coordination mass")
    return clean_on_noisy, noisy_field, coverage.clamp_max(1.0)


def fixed_cosine_projection(hidden: int, channels: int, reference: torch.Tensor) -> torch.Tensor:
    if not 1 <= channels <= hidden:
        raise ValueError("invalid fixed node projection width")
    row = torch.arange(hidden, device=reference.device, dtype=reference.dtype) + 0.5
    column = torch.arange(channels, device=reference.device, dtype=reference.dtype) + 0.5
    return math.sqrt(2.0 / hidden) * torch.cos(
        math.pi * row[:, None] * column[None, :] / hidden
    )


def probe_features(
    captured: dict[str, torch.Tensor],
    noisy_field: torch.Tensor,
    image_shift: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    nodes = captured["nodes"].float()
    source = captured["source"]
    target = captured["target"]
    projection = fixed_cosine_projection(nodes.shape[-1], channels, nodes)
    projected = nodes @ projection
    pair_sum = projected[source] + projected[target]
    pair_difference = (projected[source] - projected[target]).abs()
    self_image = (
        (source == target) & image_shift.ne(0).any(dim=-1)
    ).to(nodes).unsqueeze(-1)
    return torch.cat(
        (
            captured["edge_state"].float(),
            pair_sum,
            pair_difference,
            captured["radial"].float(),
            noisy_field.unsqueeze(-1),
            self_image,
        ),
        dim=-1,
    )


def fit_standardized_ridge(
    values: ScalarRidgeAccumulator, ridge_relative: float
) -> StandardizedRidge:
    if ridge_relative <= 0.0 or float(values.weight) <= 0.0:
        raise ValueError("invalid ridge fit")
    weight = values.weight
    mean = values.xsum / weight
    target_mean = values.ysum / weight
    centered = values.xtx - weight * torch.outer(mean, mean)
    cross = values.xty - weight * mean * target_mean
    scale = (centered.diag() / weight).clamp_min(1.0e-12).sqrt()
    standardized = centered / (scale[:, None] * scale[None, :])
    standardized_cross = cross / scale
    ridge = ridge_relative * float(torch.trace(standardized)) / standardized.shape[0]
    regularized = standardized + ridge * torch.eye(
        standardized.shape[0], dtype=torch.float64
    )
    coefficient = torch.linalg.solve(regularized, standardized_cross)
    eigenvalues = torch.linalg.eigvalsh(standardized)
    threshold = max(float(eigenvalues.max()) * 1.0e-10, 1.0e-14)
    positive = eigenvalues[eigenvalues > threshold]
    condition = float(positive.max() / positive.min()) if positive.numel() else math.inf
    return StandardizedRidge(
        mean=mean,
        scale=scale,
        coefficient=coefficient,
        target_mean=float(target_mean),
        ridge=ridge,
        rank=int(positive.numel()),
        condition_number=condition,
    )


def topology_carrier(
    weights: torch.Tensor,
    radial: torch.Tensor,
    direction: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    nodes: int,
    graphs: int,
) -> torch.Tensor:
    weighted = weights[:, None, None] * radial[:, :, None] * direction[:, None, :]
    carrier = sorted_segment_sum(weighted, target, nodes)
    normalizer = sorted_segment_sum(
        weights.square().unsqueeze(-1), target, nodes
    ).sqrt().clamp_min(1.0e-6)
    carrier = carrier / normalizer.unsqueeze(-1)
    return carrier - graph_mean(carrier, batch, graphs)[batch]


def fit_carrier(values: CarrierAccumulator, ridge_relative: float) -> torch.Tensor:
    matrix = values.xtx
    ridge = ridge_relative * float(torch.trace(matrix)) / matrix.shape[0]
    return torch.linalg.solve(
        matrix + ridge * torch.eye(matrix.shape[0], dtype=torch.float64), values.xty
    )


def _graph_coordinate_mse(
    value: torch.Tensor, batch: torch.Tensor, graphs: int
) -> torch.Tensor:
    squared = value.square().sum(dim=-1)
    counts = torch.bincount(batch, minlength=graphs).clamp_min(1).to(squared)
    return _edge_graph_sum(squared, batch, graphs) / counts


def _binary_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    positive = int(label.sum())
    negative = label.numel() - positive
    if positive == 0 or negative == 0:
        return math.nan
    order = torch.argsort(score, descending=True)
    sorted_label = label[order].to(torch.float64)
    true_positive = torch.cumsum(sorted_label, dim=0) / positive
    false_positive = torch.cumsum(1.0 - sorted_label, dim=0) / negative
    true_positive = torch.cat((true_positive.new_zeros(1), true_positive))
    false_positive = torch.cat((false_positive.new_zeros(1), false_positive))
    return float(torch.trapz(true_positive, false_positive))


def _bootstrap_improvement(
    baseline: torch.Tensor,
    corrected: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    values: list[torch.Tensor] = []
    for _ in range(samples):
        index = torch.randint(baseline.numel(), (baseline.numel(),), generator=generator)
        values.append(1.0 - corrected[index].mean() / baseline[index].mean().clamp_min(1.0e-15))
    stacked = torch.stack(values)
    quantiles = torch.quantile(
        stacked, torch.tensor([0.025, 0.5, 0.975], dtype=stacked.dtype)
    )
    return tuple(map(float, quantiles))


def _parameter_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _forward_batch(
    runtime: TensorFreeEmaRuntime,
    diffusion: TensorFreeHybridDiffusion,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    time_value: float,
    generator: torch.Generator,
    capture: FinalEdgeCapture,
    specification: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    time_tensor = packed.lattice.new_full((graphs,), time_value)
    noisy = diffusion.noise_clean_batch(
        packed.atom_types,
        packed.frac_coords,
        packed.lattice,
        packed.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=time_tensor,
        generator=generator,
    )
    noisy_lattice = LatticeVolumeShape(
        noisy.log_volume.float(), noisy.log_shape.float()
    ).lattice(blueprint.fractional_to_cartesian.float())
    condition = time_tensor.new_zeros((graphs, 18))
    condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    capture.values = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        prediction = runtime.model(
            noisy.element_tokens,
            noisy.fractional_coordinates,
            noisy.log_volume,
            noisy.log_shape,
            packed.batch,
            time_tensor,
            condition,
            condition_present,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
        )
    if not capture.values:
        raise RuntimeError("final edge hook did not run")
    cutoff = float(specification["topology"]["production_cutoff_angstrom"])
    noisy_edges = periodic_radius_multigraph(
        noisy.fractional_coordinates,
        noisy_lattice,
        packed.batch,
        cutoff=cutoff,
    )
    if not (
        torch.equal(noisy_edges.source, capture.values["source"])
        and torch.equal(noisy_edges.target, capture.values["target"])
        and torch.allclose(
            noisy_edges.direction, capture.values["direction"], atol=2.0e-6, rtol=2.0e-6
        )
    ):
        raise RuntimeError("external topology graph does not match model edge order")
    clean_field, noisy_field, coverage = topology_fields(
        packed.frac_coords.float(),
        packed.lattice.float(),
        noisy_edges,
        packed.batch,
        cutoff=cutoff,
        multiplier=float(specification["topology"]["first_shell_multiplier"]),
        relative_width=float(specification["topology"]["transition_relative_width"]),
    )
    features = probe_features(
        capture.values,
        noisy_field,
        noisy_edges.image_shift,
        int(specification["probe"]["fixed_node_projection_channels"]),
    )
    target_cartesian = fractional_tangent_to_cartesian(
        noisy.coordinate_scaled_score_target.float(), noisy_lattice, packed.batch
    )
    cell_scale = torch.exp(noisy.log_volume.float() / 3.0)[packed.batch, None]
    target_score = target_cartesian / cell_scale
    predicted_score = prediction.coordinate_cartesian_scaled_score.float() / cell_scale
    return {
        "packed": packed,
        "graphs": graphs,
        "edges": noisy_edges,
        "radial": capture.values["radial"].float(),
        "features": features,
        "clean_field": clean_field,
        "noisy_field": noisy_field,
        "coverage": coverage,
        "residual": target_score - predicted_score,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty audit CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_decision(
    topology_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
    specification: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float | int], str]:
    middle_times = set(map(float, specification["diagnostic"]["middle_times"]))
    topology_middle = [row for row in topology_rows if float(row["time"]) in middle_times]
    probe_middle = [row for row in probe_rows if float(row["time"]) in middle_times]
    by_variant = {
        name: [
            row
            for row in carrier_rows
            if row["variant"] == name and float(row["time"]) in middle_times
        ]
        for name in ("clean_oracle", "noisy_current", "learned_probe")
    }
    acceptance = specification["acceptance"]
    mean_jaccard = sum(float(row["soft_jaccard"]) for row in topology_middle) / len(
        topology_middle
    )
    mean_switch = sum(float(row["hard_switch_fraction"]) for row in topology_middle) / len(
        topology_middle
    )
    minimum_coverage = min(float(row["clean_mass_coverage"]) for row in topology_rows)
    oracle_mean = sum(float(row["relative_improvement"]) for row in by_variant["clean_oracle"]) / len(
        by_variant["clean_oracle"]
    )
    noisy_mean = sum(float(row["relative_improvement"]) for row in by_variant["noisy_current"]) / len(
        by_variant["noisy_current"]
    )
    learned_mean = sum(float(row["relative_improvement"]) for row in by_variant["learned_probe"]) / len(
        by_variant["learned_probe"]
    )
    oracle_support = sum(
        float(row["relative_improvement"])
        >= float(acceptance["oracle_each_supporting_time_improvement_min"])
        for row in by_variant["clean_oracle"]
    )
    probe_explained = sum(float(row["explained_fraction"]) for row in probe_middle) / len(
        probe_middle
    )
    probe_auc = sum(float(row["auc"]) for row in probe_middle) / len(probe_middle)
    probe_over_noisy = sum(
        float(row["improvement_over_noisy"]) for row in probe_middle
    ) / len(probe_middle)
    checks = {
        "middle_topology_is_disrupted": (
            mean_jaccard <= float(acceptance["middle_soft_jaccard_max"])
            and mean_switch >= float(acceptance["middle_hard_switch_fraction_min"])
        ),
        "clean_topology_mass_is_covered": minimum_coverage
        >= float(acceptance["clean_topology_mass_coverage_min"]),
        "clean_topology_oracle_helps": (
            oracle_mean >= float(acceptance["oracle_middle_mean_improvement_min"])
            and oracle_mean - noisy_mean
            >= float(acceptance["oracle_minus_noisy_middle_mean_min"])
            and oracle_support
            >= int(acceptance["oracle_middle_supporting_times_min"])
        ),
        "clean_topology_probe_is_predictive": (
            probe_explained
            >= float(acceptance["probe_middle_mean_explained_fraction_min"])
            and probe_auc >= float(acceptance["probe_middle_mean_auc_min"])
            and probe_over_noisy
            >= float(acceptance["probe_middle_mean_improvement_over_noisy_min"])
        ),
        "probe_weighted_carrier_retains_oracle_gain": (
            learned_mean >= float(acceptance["learned_middle_mean_improvement_min"])
            and learned_mean / max(oracle_mean, 1.0e-12)
            >= float(acceptance["learned_to_oracle_improvement_ratio_min"])
        ),
    }
    metrics: dict[str, float | int] = {
        "middle_soft_jaccard": mean_jaccard,
        "middle_hard_switch_fraction": mean_switch,
        "minimum_clean_mass_coverage": minimum_coverage,
        "oracle_middle_mean_improvement": oracle_mean,
        "noisy_middle_mean_improvement": noisy_mean,
        "oracle_minus_noisy_middle_mean": oracle_mean - noisy_mean,
        "oracle_middle_supporting_times": oracle_support,
        "probe_middle_mean_explained_fraction": probe_explained,
        "probe_middle_mean_auc": probe_auc,
        "probe_middle_mean_improvement_over_noisy": probe_over_noisy,
        "learned_middle_mean_improvement": learned_mean,
        "learned_to_oracle_improvement_ratio": learned_mean / max(oracle_mean, 1.0e-12),
    }
    if not checks["middle_topology_is_disrupted"]:
        decision = "noisy_topology_not_materially_disrupted"
    elif not checks["clean_topology_mass_is_covered"]:
        decision = "audit_invalid_clean_topology_mass_not_covered"
    elif not checks["clean_topology_oracle_helps"]:
        decision = "clean_topology_hypothesis_rejected"
    elif not checks["clean_topology_probe_is_predictive"]:
        decision = "oracle_useful_but_topology_not_recoverable_from_current_states"
    elif not checks["probe_weighted_carrier_retains_oracle_gain"]:
        decision = "probe_predictive_but_topology_correction_not_residual_causal"
    else:
        decision = "authorize_separate_latent_topology_mechanism_qualification"
    return checks, metrics, decision


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    specification = load_json_object(args.protocol)
    if specification.get("protocol") != "h1a_latent_clean_topology_attribution_v1":
        raise ValueError("unexpected clean-topology protocol")
    prerequisites = specification["prerequisites"]
    checkpoint_protocol = load_json_object(args.checkpoint_protocol)
    if sha256_file(args.checkpoint_protocol) != str(
        prerequisites["checkpoint_protocol_sha256"]
    ):
        raise ValueError("checkpoint protocol hash mismatch")
    if sha256_file(args.checkpoint) != str(prerequisites["checkpoint_sha256"]):
        raise ValueError("checkpoint hash mismatch")
    if sha256_file(args.checkpoint_result) != str(
        prerequisites["checkpoint_result_sha256"]
    ):
        raise ValueError("checkpoint result hash mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        prerequisites["cache_manifest_sha256"]
    ):
        raise ValueError("cache manifest hash mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    runtime = load_tensor_free_ema_runtime(
        args.checkpoint,
        device,
        protocol_name=str(checkpoint_protocol["protocol"]),
        protocol_sha256=canonical_json_hash(checkpoint_protocol),
    )
    runtime.model.eval()
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    train_dataset = PackedAlexP1Dataset(args.cache_root, "train")
    validation_dataset = PackedAlexP1Dataset(args.cache_root, "val")
    diagnostic = specification["diagnostic"]
    train_indices = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(int(diagnostic["train_selection_seed"])),
    )[: int(diagnostic["train_graphs"])]
    validation_indices = torch.randperm(
        len(validation_dataset),
        generator=torch.Generator().manual_seed(int(diagnostic["validation_selection_seed"])),
    )[: int(diagnostic["validation_graphs"])]
    fingerprint_before = _parameter_fingerprint(runtime.model)
    capture = FinalEdgeCapture(runtime.model)
    topology_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for time_index, time_value in enumerate(diagnostic["times"]):
            probe_accumulator: ScalarRidgeAccumulator | None = None
            carrier_accumulators: dict[str, CarrierAccumulator] = {}
            train_generator = torch.Generator(device=device).manual_seed(
                int(diagnostic["train_noise_seed"]) + time_index
            )
            for start in range(0, train_indices.numel(), int(diagnostic["batch_size"])):
                batch_data = _forward_batch(
                    runtime,
                    diffusion,
                    train_dataset,
                    train_indices[start : start + int(diagnostic["batch_size"])],
                    float(time_value),
                    train_generator,
                    capture,
                    specification,
                    device=device,
                )
                edges = batch_data["edges"]
                packed = batch_data["packed"]
                edge_graph = packed.batch[edges.target]
                if probe_accumulator is None:
                    probe_accumulator = ScalarRidgeAccumulator.create(
                        batch_data["features"].shape[-1]
                    )
                    carrier_accumulators = {
                        "clean_oracle": CarrierAccumulator.create(
                            batch_data["radial"].shape[-1]
                        ),
                        "noisy_current": CarrierAccumulator.create(
                            batch_data["radial"].shape[-1]
                        ),
                    }
                probe_accumulator.update(
                    batch_data["features"],
                    batch_data["clean_field"],
                    edge_graph,
                    batch_data["graphs"],
                )
                for name, field in (
                    ("clean_oracle", batch_data["clean_field"]),
                    ("noisy_current", batch_data["noisy_field"]),
                ):
                    carrier = topology_carrier(
                        field,
                        batch_data["radial"],
                        edges.direction,
                        edges.target,
                        packed.batch,
                        packed.num_nodes,
                        batch_data["graphs"],
                    )
                    carrier_accumulators[name].update(
                        carrier,
                        batch_data["residual"],
                        packed.batch,
                        batch_data["graphs"],
                    )
            if probe_accumulator is None:
                raise RuntimeError("empty topology train panel")
            probe = fit_standardized_ridge(
                probe_accumulator, float(specification["probe"]["ridge_relative"])
            )
            coefficients = {
                name: fit_carrier(
                    accumulator,
                    float(specification["oracle_carrier"]["ridge_relative"]),
                )
                for name, accumulator in carrier_accumulators.items()
            }
            validation_generator = torch.Generator(device=device).manual_seed(
                int(diagnostic["validation_noise_seed"]) + time_index
            )
            topology_parts: dict[str, list[torch.Tensor]] = {
                name: []
                for name in (
                    "soft_jaccard",
                    "hard_switch_fraction",
                    "hard_precision",
                    "hard_recall",
                    "coverage",
                )
            }
            probe_parts: dict[str, list[torch.Tensor]] = {
                name: [] for name in ("probe_mse", "noisy_mse", "mean_mse")
            }
            auc_values: list[float] = []
            baseline_parts: list[torch.Tensor] = []
            corrected_parts: dict[str, list[torch.Tensor]] = {
                name: [] for name in ("clean_oracle", "noisy_current", "learned_probe")
            }
            for start in range(
                0, validation_indices.numel(), int(diagnostic["batch_size"])
            ):
                batch_data = _forward_batch(
                    runtime,
                    diffusion,
                    validation_dataset,
                    validation_indices[
                        start : start + int(diagnostic["batch_size"])
                    ],
                    float(time_value),
                    validation_generator,
                    capture,
                    specification,
                    device=device,
                )
                edges = batch_data["edges"]
                packed = batch_data["packed"]
                graphs = batch_data["graphs"]
                edge_graph = packed.batch[edges.target]
                clean_field = batch_data["clean_field"]
                noisy_field = batch_data["noisy_field"]
                prediction = probe.predict(batch_data["features"])
                clipped_prediction = prediction.clamp(0.0, 1.0)
                minimum = torch.minimum(clean_field, noisy_field)
                maximum = torch.maximum(clean_field, noisy_field)
                topology_parts["soft_jaccard"].append(
                    _edge_graph_sum(minimum, edge_graph, graphs)
                    / _edge_graph_sum(maximum, edge_graph, graphs).clamp_min(1.0e-12)
                )
                threshold = float(specification["topology"]["hard_probability_threshold"])
                clean_hard = clean_field >= threshold
                noisy_hard = noisy_field >= threshold
                true_positive = _edge_graph_sum(
                    (clean_hard & noisy_hard).to(clean_field), edge_graph, graphs
                )
                predicted_positive = _edge_graph_sum(
                    noisy_hard.to(clean_field), edge_graph, graphs
                )
                clean_positive = _edge_graph_sum(
                    clean_hard.to(clean_field), edge_graph, graphs
                )
                edge_count = torch.bincount(edge_graph, minlength=graphs).to(clean_field)
                topology_parts["hard_switch_fraction"].append(
                    _edge_graph_sum(
                        (clean_hard != noisy_hard).to(clean_field), edge_graph, graphs
                    )
                    / edge_count.clamp_min(1.0)
                )
                topology_parts["hard_precision"].append(
                    true_positive / predicted_positive.clamp_min(1.0)
                )
                topology_parts["hard_recall"].append(
                    true_positive / clean_positive.clamp_min(1.0)
                )
                topology_parts["coverage"].append(batch_data["coverage"])
                counts = torch.bincount(edge_graph, minlength=graphs).clamp_min(1).to(clean_field)
                probe_parts["probe_mse"].append(
                    _edge_graph_sum(
                        (clipped_prediction - clean_field).square(), edge_graph, graphs
                    )
                    / counts
                )
                probe_parts["noisy_mse"].append(
                    _edge_graph_sum(
                        (noisy_field - clean_field).square(), edge_graph, graphs
                    )
                    / counts
                )
                probe_parts["mean_mse"].append(
                    _edge_graph_sum(
                        (probe.target_mean - clean_field).square(), edge_graph, graphs
                    )
                    / counts
                )
                for graph in range(graphs):
                    mask = edge_graph == graph
                    value = _binary_auc(prediction[mask].cpu(), clean_hard[mask].cpu())
                    if math.isfinite(value):
                        auc_values.append(value)
                residual = batch_data["residual"]
                baseline_parts.append(
                    _graph_coordinate_mse(residual, packed.batch, graphs).cpu()
                )
                for name, field, coefficient_name in (
                    ("clean_oracle", clean_field, "clean_oracle"),
                    ("noisy_current", noisy_field, "noisy_current"),
                    ("learned_probe", clipped_prediction, "clean_oracle"),
                ):
                    carrier = topology_carrier(
                        field,
                        batch_data["radial"],
                        edges.direction,
                        edges.target,
                        packed.batch,
                        packed.num_nodes,
                        graphs,
                    )
                    correction = torch.einsum(
                        "ncd,c->nd",
                        carrier,
                        coefficients[coefficient_name].to(carrier),
                    )
                    corrected_parts[name].append(
                        _graph_coordinate_mse(
                            residual - correction, packed.batch, graphs
                        ).cpu()
                    )
            topology_rows.append(
                {
                    "time": float(time_value),
                    "soft_jaccard": float(
                        torch.cat(topology_parts["soft_jaccard"]).double().mean()
                    ),
                    "hard_switch_fraction": float(
                        torch.cat(topology_parts["hard_switch_fraction"]).double().mean()
                    ),
                    "hard_precision": float(
                        torch.cat(topology_parts["hard_precision"]).double().mean()
                    ),
                    "hard_recall": float(
                        torch.cat(topology_parts["hard_recall"]).double().mean()
                    ),
                    "clean_mass_coverage": float(
                        torch.cat(topology_parts["coverage"]).double().mean()
                    ),
                    "minimum_graph_clean_mass_coverage": float(
                        torch.cat(topology_parts["coverage"]).double().min()
                    ),
                }
            )
            probe_mse = torch.cat(probe_parts["probe_mse"]).double().mean()
            noisy_mse = torch.cat(probe_parts["noisy_mse"]).double().mean()
            mean_mse = torch.cat(probe_parts["mean_mse"]).double().mean()
            probe_rows.append(
                {
                    "time": float(time_value),
                    "probe_mse": float(probe_mse),
                    "noisy_field_mse": float(noisy_mse),
                    "constant_train_mean_mse": float(mean_mse),
                    "explained_fraction": 1.0 - float(probe_mse / mean_mse.clamp_min(1.0e-15)),
                    "improvement_over_noisy": 1.0
                    - float(probe_mse / noisy_mse.clamp_min(1.0e-15)),
                    "auc": sum(auc_values) / len(auc_values),
                    "ridge": probe.ridge,
                    "rank": probe.rank,
                    "condition_number": probe.condition_number,
                    "features": int(probe.coefficient.numel()),
                }
            )
            baseline = torch.cat(baseline_parts).double()
            for variant_index, (name, parts) in enumerate(corrected_parts.items()):
                corrected = torch.cat(parts).double()
                interval = _bootstrap_improvement(
                    baseline,
                    corrected,
                    seed=int(diagnostic["bootstrap_seed"])
                    + 10 * time_index
                    + variant_index,
                    samples=int(diagnostic["bootstrap_samples"]),
                )
                carrier_rows.append(
                    {
                        "time": float(time_value),
                        "variant": name,
                        "baseline_mse": float(baseline.mean()),
                        "corrected_mse": float(corrected.mean()),
                        "relative_improvement": 1.0
                        - float(corrected.mean() / baseline.mean().clamp_min(1.0e-15)),
                        "bootstrap_95_low": interval[0],
                        "bootstrap_median": interval[1],
                        "bootstrap_95_high": interval[2],
                    }
                )
    finally:
        capture.close()
    fingerprint_after = _parameter_fingerprint(runtime.model)
    checks, metrics, decision = _aggregate_decision(
        topology_rows, probe_rows, carrier_rows, specification
    )
    result = {
        "protocol": specification["protocol"],
        "protocol_sha256": canonical_json_hash(specification),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "train_indices_sha256": canonical_json_hash(train_indices.tolist()),
        "validation_indices_sha256": canonical_json_hash(validation_indices.tolist()),
        "checks": checks,
        "decision_metrics": metrics,
        "decision": decision,
        "qualified": decision
        == "authorize_separate_latent_topology_mechanism_qualification",
        "optimizer_steps": 0,
        "checkpoint_parameters_unchanged": fingerprint_before == fingerprint_after,
        "parameter_fingerprint": fingerprint_after,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
        "decision_boundary": specification["decision_rule"]["boundary"],
    }
    if not result["checkpoint_parameters_unchanged"]:
        raise RuntimeError("frozen topology audit modified checkpoint parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "topology_state.csv", topology_rows)
    _write_csv(args.output / "topology_probe.csv", probe_rows)
    _write_csv(args.output / "topology_oracle_carrier.csv", carrier_rows)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = [
        "# H1a latent clean-topology attribution v1",
        "",
        f"Decision: **{decision}**.",
        "",
        "This is a zero-optimizer frozen-checkpoint diagnostic. It does not add a production branch or change H1a.",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{value}`" for name, value in checks.items()],
        "",
        "## Decision metrics",
        "",
        *[f"- {name}: `{value:.6f}`" for name, value in metrics.items()],
        "",
    ]
    (args.output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
