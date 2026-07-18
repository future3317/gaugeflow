"""Qualify a compact target-free Cartesian moment/Krylov coordinate carrier."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.geometry import PeriodicEdges, periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.state_projection import graph_mean
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _fixed_indices,
    _make_batch,
    _make_model,
    _predict,
)

READOUT_NAMES = (
    "coordinate_vector_head.weight",
    "coordinate_edge_head.2.weight",
    "coordinate_edge_head.2.bias",
)


def _vector_rms_normalize(
    vectors: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    epsilon: float,
) -> torch.Tensor:
    if vectors.ndim != 3 or vectors.shape[-1] != 3 or epsilon <= 0.0:
        raise ValueError("vector carrier must have shape [nodes,channels,3]")
    energy = graph_mean(vectors.square().sum(-1) / 3.0, batch, graph_count)
    return vectors * (energy + epsilon).rsqrt()[batch, :, None]


def _stf_rms_normalize(
    tensors: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    epsilon: float,
) -> torch.Tensor:
    if tensors.ndim != 4 or tensors.shape[-2:] != (3, 3) or epsilon <= 0.0:
        raise ValueError("STF carrier must have shape [nodes,channels,3,3]")
    energy = graph_mean(
        tensors.square().sum(dim=(-1, -2)) / 5.0, batch, graph_count
    )
    return tensors * (energy + epsilon).rsqrt()[batch, :, None, None]


def compact_cartesian_krylov_carrier(
    vector_basis: torch.Tensor,
    edge_hidden: torch.Tensor,
    edges: PeriodicEdges,
    edge_envelope: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    projection: torch.Tensor,
    rms_epsilon: float,
) -> torch.Tensor:
    """Build bounded O(3)-polar carriers from first and second moments."""
    if vector_basis.ndim != 3 or vector_basis.shape[-1] != 3:
        raise ValueError("vector basis must have shape [nodes,channels,3]")
    if edge_hidden.ndim != 2 or edge_hidden.shape[0] != edges.source.numel():
        raise ValueError("edge hidden features do not match the periodic graph")
    if projection.shape[0] != edge_hidden.shape[1] or projection.shape[1] % 2:
        raise ValueError("moment projection must contain paired scalar channels")
    if edge_envelope.shape != (edges.source.numel(), 1):
        raise ValueError("edge envelope must have shape [edges,1]")
    if batch.shape != (vector_basis.shape[0],) or graph_count < 1:
        raise ValueError("carrier batch does not match nodes")
    moment_channels = projection.shape[1] // 2
    coefficients = torch.tanh(edge_hidden.float() @ projection.float())
    first_coefficients, second_coefficients = coefficients.split(moment_channels, dim=-1)
    envelope = edge_envelope.float()
    direction = edges.direction.float()
    degree = torch.bincount(edges.target, minlength=vector_basis.shape[0]).float()
    degree_scale = degree.clamp_min(1.0).rsqrt()

    first_messages = first_coefficients[:, :, None] * envelope[:, :, None] * direction[:, None, :]
    first = vector_basis.new_zeros((vector_basis.shape[0], moment_channels, 3)).float()
    first.index_add_(0, edges.target, first_messages)
    first = first * degree_scale[:, None, None]

    identity = torch.eye(3, dtype=direction.dtype, device=direction.device)
    dyad = torch.einsum("ei,ej->eij", direction, direction) - identity / 3.0
    second_messages = (
        second_coefficients[:, :, None, None]
        * envelope[:, :, None, None]
        * dyad[:, None, :, :]
    )
    second = vector_basis.new_zeros(
        (vector_basis.shape[0], moment_channels, 3, 3)
    ).float()
    second.index_add_(0, edges.target, second_messages)
    second = second * degree_scale[:, None, None, None]

    vector = _vector_rms_normalize(
        vector_basis.float(), batch, graph_count, rms_epsilon
    )
    first = _vector_rms_normalize(first, batch, graph_count, rms_epsilon)
    second = _stf_rms_normalize(second, batch, graph_count, rms_epsilon)
    second_first = torch.einsum("ncij,ncj->nci", second, first)
    second_first = _vector_rms_normalize(
        second_first, batch, graph_count, rms_epsilon
    )
    second_squared_first = torch.einsum(
        "ncij,ncj->nci", second, second_first
    )
    second_squared_first = _vector_rms_normalize(
        second_squared_first, batch, graph_count, rms_epsilon
    )
    carrier = torch.cat((vector, first, second_first, second_squared_first), dim=1)
    return carrier - graph_mean(carrier, batch, graph_count)[batch]


@dataclass(frozen=True)
class CapturedCarrierInput:
    vector_basis: torch.Tensor
    edge_hidden: torch.Tensor
    edges: PeriodicEdges
    edge_envelope: torch.Tensor
    lattice: torch.Tensor


def _capture_input(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    *,
    use_bf16: bool,
) -> CapturedCarrierInput:
    captures: dict[str, torch.Tensor] = {}

    def vector_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["vector"] = inputs[0]

    def edge_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["edge"] = inputs[0]

    vector_handle = model.coordinate_vector_head.register_forward_pre_hook(  # type: ignore[attr-defined]
        vector_hook
    )
    edge_handle = model.coordinate_edge_head[2].register_forward_pre_hook(  # type: ignore[attr-defined,index]
        edge_hook
    )
    try:
        _ = _predict(model, noisy, batch_data, blueprint, use_bf16=use_bf16)
    finally:
        vector_handle.remove()
        edge_handle.remove()
    with torch.autocast(device_type=noisy.time.device.type, enabled=False):
        lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
            blueprint.fractional_to_cartesian
        ).float()
        edges = periodic_radius_multigraph(
            noisy.fractional_coordinates.float(),
            lattice,
            batch_data.batch,
            cutoff=float(model.radial.cutoff),  # type: ignore[attr-defined]
        )
        envelope = model.radial.envelope(edges.distance).float()  # type: ignore[attr-defined]
    if captures["edge"].shape[0] != edges.source.numel():
        raise RuntimeError("captured edge features do not match production edge order")
    return CapturedCarrierInput(
        vector_basis=captures["vector"].float().transpose(-1, -2),
        edge_hidden=captures["edge"],
        edges=edges,
        edge_envelope=envelope,
        lattice=lattice,
    )


def _projection(hidden: int, channels: int, seed: int, device: torch.device) -> torch.Tensor:
    random = torch.randn(
        (hidden, 2 * channels),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    orthogonal, _ = torch.linalg.qr(random, mode="reduced")
    return orthogonal.float().to(device)


def _helmert(node_count: int, device: torch.device) -> torch.Tensor:
    if node_count < 2:
        raise ValueError("quotient rank audit requires at least two nodes")
    matrix = torch.zeros((node_count, node_count - 1), dtype=torch.float64, device=device)
    for column in range(node_count - 1):
        prefix = column + 1
        denominator = math.sqrt(prefix * (prefix + 1))
        matrix[:prefix, column] = 1.0 / denominator
        matrix[prefix, column] = -prefix / denominator
    return torch.kron(matrix, torch.eye(3, dtype=torch.float64, device=device))


def _rank_metrics(
    carrier: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    relative_tolerance: float,
) -> list[dict[str, float | int | bool]]:
    fractional = torch.einsum(
        "nci,nji->ncj", carrier.double(), lattice[batch].double()
    )
    rows: list[dict[str, float | int | bool]] = []
    for graph in range(graph_count):
        selected = fractional[batch == graph]
        nodes = selected.shape[0]
        design = selected.transpose(1, 2).reshape(3 * nodes, -1)
        quotient = _helmert(nodes, design.device).T @ design
        singular = torch.linalg.svdvals(quotient.cpu())
        threshold = relative_tolerance * singular[0].clamp_min(1e-30)
        active = singular > threshold
        expected = min(3 * (nodes - 1), design.shape[1])
        rank = int(active.sum())
        rows.append(
            {
                "graph": graph,
                "nodes": nodes,
                "expected_rank": expected,
                "rank": rank,
                "full_rank": rank == expected,
                "condition_number": float(singular[0] / singular[active][-1])
                if active.any()
                else math.inf,
            }
        )
    return rows


def _gradient_fields(model: nn.Module) -> tuple[dict[str, torch.Tensor], float, bool]:
    gradients: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=next(model.parameters()).device)
    finite = True
    for name, parameter in model.named_parameters():
        if name in READOUT_NAMES or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradients[name] = gradient.cpu().clone()
        total = total + gradient.square().sum()
        finite = finite and bool(torch.isfinite(gradient).all())
    return gradients, float(total.sqrt()), finite


def _gradient_agreement(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, float]:
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("gradient fields must have matching nonempty keys")
    dot = sum(float((reference[name] * candidate[name]).sum()) for name in reference)
    reference_norm = math.sqrt(
        sum(float(reference[name].square().sum()) for name in reference)
    )
    candidate_norm = math.sqrt(
        sum(float(candidate[name].square().sum()) for name in candidate)
    )
    return {
        "cosine": dot / max(reference_norm * candidate_norm, 1e-30),
        "candidate_over_reference_norm": candidate_norm / max(reference_norm, 1e-30),
    }


def _carrier_and_gradients(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
    projection: torch.Tensor,
    probe: torch.Tensor,
    epsilon: float,
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float, bool, CapturedCarrierInput]:
    model.train()
    model.zero_grad(set_to_none=True)
    captured = _capture_input(
        model, noisy, batch_data, blueprint, use_bf16=use_bf16
    )
    carrier = compact_cartesian_krylov_carrier(
        captured.vector_basis,
        captured.edge_hidden,
        captured.edges,
        captured.edge_envelope,
        batch_data.batch,
        int(batch_data.num_graphs),
        projection,
        epsilon,
    )
    field = torch.einsum("nci,c->ni", carrier, probe)
    energy = field.square().mean()
    energy.backward()
    gradients, norm, finite = _gradient_fields(model)
    model.zero_grad(set_to_none=True)
    return carrier.detach(), gradients, norm, finite and math.isfinite(float(energy)), captured


def _operator_benchmark(
    captured: CapturedCarrierInput,
    batch: torch.Tensor,
    graph_count: int,
    projection: torch.Tensor,
    epsilon: float,
    repeats: int,
) -> dict[str, float]:
    device = captured.vector_basis.device
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    for _ in range(10):
        _ = compact_cartesian_krylov_carrier(
            captured.vector_basis.detach(),
            captured.edge_hidden.detach(),
            captured.edges,
            captured.edge_envelope,
            batch,
            graph_count,
            projection,
            epsilon,
        )
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _ = compact_cartesian_krylov_carrier(
            captured.vector_basis.detach(),
            captured.edge_hidden.detach(),
            captured.edges,
            captured.edge_envelope,
            batch,
            graph_count,
            projection,
            epsilon,
        )
    stop.record()
    torch.cuda.synchronize(device)
    return {
        "latency_ms": float(start.elapsed_time(stop) / repeats),
        "peak_memory_mib": float(
            (torch.cuda.max_memory_allocated(device) - baseline) / (1024**2)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_compact_cartesian_krylov_carrier_v1":
        raise ValueError("compact Cartesian carrier protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("compact Cartesian carrier cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("compact Cartesian carrier audit forbids optimizer steps")
    if int(protocol["audit"]["coordinate_targets_read"]) != 0:
        raise ValueError("compact Cartesian carrier audit forbids coordinate targets")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen compact Cartesian carrier audit requires CUDA")

    path = protocol["path"]
    torch.manual_seed(int(path["model_seed"]))
    torch.cuda.manual_seed_all(int(path["model_seed"]))
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    data = protocol["data"]
    indices = _fixed_indices(
        len(dataset), int(data["fixed_graphs"]), int(data["fixed_selection_seed"])
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float()
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(path["coordinate_sigma_min"]),
        coordinate_sigma_max=float(path["coordinate_sigma_max"]),
        minimum_time=float(path["minimum_time"]),
        maximum_time=float(path["maximum_time"]),
    )
    times = batch_data.lattice.new_tensor(path["time_grid"])
    graph_time = times[
        torch.arange(int(batch_data.num_graphs), device=device) % times.numel()
    ]
    with torch.no_grad():
        noisy = diffusion.noise_clean_batch(
            batch_data.atom_types,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=graph_time,
            generator=torch.Generator(device=device).manual_seed(int(path["noise_seed"])),
        )
    carrier_config = protocol["carrier"]
    projection = _projection(
        int(protocol["model"]["hidden_dim"]),
        int(carrier_config["moment_channels"]),
        int(carrier_config["projection_seed"]),
        device,
    )
    carrier_channels = int(protocol["model"]["vector_dim"]) + 3 * int(
        carrier_config["moment_channels"]
    )
    probe = torch.randn(
        carrier_channels,
        generator=torch.Generator().manual_seed(int(carrier_config["probe_seed"])),
    ).to(device)
    probe = probe / torch.linalg.vector_norm(probe)
    epsilon = float(carrier_config["rms_epsilon"])
    fp32_carrier, fp32_gradients, fp32_gradient_norm, fp32_finite, captured = (
        _carrier_and_gradients(
            model,
            noisy,
            batch_data,
            blueprint,
            projection,
            probe,
            epsilon,
            use_bf16=False,
        )
    )
    bf16_carrier, bf16_gradients, bf16_gradient_norm, bf16_finite, _ = (
        _carrier_and_gradients(
            model,
            noisy,
            batch_data,
            blueprint,
            projection,
            probe,
            epsilon,
            use_bf16=True,
        )
    )
    carrier_relative_rmse = float(
        (bf16_carrier - fp32_carrier).square().mean().sqrt()
        / fp32_carrier.square().mean().sqrt().clamp_min(1e-30)
    )
    carrier_cosine = float(
        (bf16_carrier * fp32_carrier).sum()
        / (
            torch.linalg.vector_norm(bf16_carrier)
            * torch.linalg.vector_norm(fp32_carrier)
        ).clamp_min(1e-30)
    )
    gradient_agreement = _gradient_agreement(fp32_gradients, bf16_gradients)
    rank_rows = _rank_metrics(
        fp32_carrier,
        captured.lattice,
        batch_data.batch,
        int(batch_data.num_graphs),
        float(protocol["audit"]["rank_relative_tolerance"]),
    )

    orthogonal, _ = torch.linalg.qr(
        torch.randn(
            (3, 3),
            generator=torch.Generator(device=device).manual_seed(5927),
            device=device,
        )
    )
    orthogonal[:, 0] = orthogonal[:, 0] * -1.0
    rotated_edges = PeriodicEdges(
        source=captured.edges.source,
        target=captured.edges.target,
        displacement=captured.edges.displacement @ orthogonal,
        direction=captured.edges.direction @ orthogonal,
        distance=captured.edges.distance,
        image_shift=captured.edges.image_shift,
    )
    rotated = compact_cartesian_krylov_carrier(
        captured.vector_basis @ orthogonal,
        captured.edge_hidden,
        rotated_edges,
        captured.edge_envelope,
        batch_data.batch,
        int(batch_data.num_graphs),
        projection,
        epsilon,
    )
    covariance_error = float(
        torch.linalg.vector_norm(rotated - fp32_carrier @ orthogonal)
        / torch.linalg.vector_norm(fp32_carrier).clamp_min(1e-30)
    )
    horizontal_error = float(
        graph_mean(
            fp32_carrier, batch_data.batch, int(batch_data.num_graphs)
        ).abs().max()
    )
    benchmark = _operator_benchmark(
        captured,
        batch_data.batch,
        int(batch_data.num_graphs),
        projection,
        epsilon,
        int(protocol["audit"]["latency_repeats"]),
    )
    model.load_state_dict(initial_state, strict=True)
    parameters_restored = all(
        torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items()
    )
    acceptance = protocol["acceptance"]
    maximum_condition = max(float(row["condition_number"]) for row in rank_rows)
    checks = {
        "all_graphs_full_quotient_rank": all(bool(row["full_rank"]) for row in rank_rows)
        is bool(acceptance["all_graphs_full_quotient_rank"]),
        "quotient_condition": maximum_condition
        <= float(acceptance["maximum_quotient_condition_number_max"]),
        "o3_covariance": covariance_error
        <= float(acceptance["carrier_o3_covariance_error_max"]),
        "translation_horizontal": horizontal_error
        <= float(acceptance["translation_horizontal_error_max"]),
        "bf16_carrier_rmse": carrier_relative_rmse
        <= float(acceptance["bf16_carrier_relative_rmse_max"]),
        "bf16_carrier_cosine": carrier_cosine
        >= float(acceptance["bf16_carrier_cosine_min"]),
        "fp32_gradient": fp32_finite
        and fp32_gradient_norm <= float(acceptance["fp32_probe_gradient_norm_max"]),
        "bf16_gradient": bf16_finite
        and bf16_gradient_norm <= float(acceptance["bf16_probe_gradient_norm_max"]),
        "gradient_norm_agreement": float(
            acceptance["bf16_over_fp32_gradient_norm_min"]
        )
        <= gradient_agreement["candidate_over_reference_norm"]
        <= float(acceptance["bf16_over_fp32_gradient_norm_max"]),
        "gradient_direction": gradient_agreement["cosine"]
        >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
        "operator_latency": benchmark["latency_ms"]
        <= float(acceptance["carrier_operator_latency_ms_max"]),
        "operator_memory": benchmark["peak_memory_mib"]
        <= float(acceptance["carrier_operator_peak_memory_mib_max"]),
        "finite": fp32_finite and bf16_finite
        is bool(acceptance["finite_forward_and_backward"]),
        "parameters_restored": parameters_restored
        is bool(acceptance["parameters_restored"]),
        "sampling_failures": int(acceptance["sampling_failures"]) == 0,
        "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
        "coordinate_targets_read": int(acceptance["coordinate_targets_read"]) == 0,
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "carrier_channels": carrier_channels,
        "edge_count": int(captured.edges.source.numel()),
        "rank_rows": rank_rows,
        "maximum_quotient_condition_number": maximum_condition,
        "carrier_o3_covariance_error": covariance_error,
        "translation_horizontal_error": horizontal_error,
        "bf16_carrier_relative_rmse": carrier_relative_rmse,
        "bf16_carrier_cosine": carrier_cosine,
        "fp32_probe_gradient_norm": fp32_gradient_norm,
        "bf16_probe_gradient_norm": bf16_gradient_norm,
        "gradient_agreement": gradient_agreement,
        "operator_benchmark": benchmark,
        "checks": checks,
        "qualified": qualified,
        "optimizer_steps": 0,
        "coordinate_targets_read": 0,
        "sampling_failures": 0,
        "tensor_candidates": 0,
        "decision": (
            "compact_cartesian_krylov_carrier_qualified_freeze_production_integration"
            if qualified
            else "compact_cartesian_krylov_carrier_failed_retain_production_head"
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
