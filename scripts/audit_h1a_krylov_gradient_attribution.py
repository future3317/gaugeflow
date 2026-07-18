"""Attribute the failed historical Krylov integration's absolute Jacobian."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.state_projection import graph_mean
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _fixed_indices,
    _make_batch,
)


def classify_attribution(
    rms_amplification: float,
    chart_amplification: float,
    q2m_over_full: float,
    maximum_group_fraction: float,
    thresholds: dict[str, float],
) -> str:
    """Apply the frozen causal priority without inspecting any target."""
    if rms_amplification >= thresholds["rms_derivative_amplification_min"]:
        return "rms_derivative_dominant"
    if chart_amplification >= thresholds["fractional_over_cartesian_gradient_min"]:
        return "fractional_chart_dominant"
    if q2m_over_full >= thresholds["q2m_over_full_gradient_min"]:
        return "q2m_order_dominant"
    if maximum_group_fraction >= thresholds["parameter_group_squared_fraction_min"]:
        return "parameter_group_dominant"
    return "distributed_jacobian_scale"


def _parameter_group(name: str) -> str:
    prefixes = {
        "carrier_head": ("coordinate_carrier_head.",),
        "moment_projection": ("coordinate_carrier.moment_projection.",),
        "edge_encoder": ("coordinate_edge_encoder.",),
        "control_gate": ("coordinate_control_gate.",),
        "message_blocks": ("blocks.",),
    }
    for group, values in prefixes.items():
        if name.startswith(values):
            return group
    return "shared"


def _gradient_metrics(
    loss: torch.Tensor,
    model: nn.Module,
    *,
    retain_graph: bool,
) -> dict[str, Any]:
    named = [(name, parameter) for name, parameter in model.named_parameters()]
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = {
        group: 0.0
        for group in (
            "carrier_head",
            "moment_projection",
            "edge_encoder",
            "control_gate",
            "message_blocks",
            "shared",
        )
    }
    finite = math.isfinite(float(loss.detach()))
    for (name, _), gradient in zip(named, gradients, strict=True):
        if gradient is None:
            continue
        value = gradient.detach().float()
        finite = finite and bool(torch.isfinite(value).all())
        squared[_parameter_group(name)] += float(value.square().sum())
    total_squared = sum(squared.values())
    return {
        "loss": float(loss.detach()),
        "gradient_norm": math.sqrt(total_squared),
        "group_gradient_norms": {
            group: math.sqrt(value) for group, value in squared.items()
        },
        "group_squared_fractions": {
            group: value / max(total_squared, 1e-30)
            for group, value in squared.items()
        },
        "finite": finite,
    }


def _rms_normalize(
    value: torch.Tensor,
    batch: torch.Tensor,
    graphs: int,
    dimensions: float,
    epsilon: float,
    *,
    detach_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(2, value.ndim))
    energy = graph_mean(
        value.square().sum(dim=reduce_dims) / dimensions,
        batch,
        graphs,
    )
    scale = (energy + epsilon).rsqrt()
    applied = scale.detach() if detach_scale else scale
    extra = (None,) * len(reduce_dims)
    return value * applied[(batch, slice(None), *extra)], scale


def _diagnostic_carrier(
    module: nn.Module,
    vector_basis: torch.Tensor,
    edge_hidden: torch.Tensor,
    edge_target: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_envelope: torch.Tensor,
    batch: torch.Tensor,
    graphs: int,
    *,
    detach_scale: bool,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    vectors = vector_basis.float()
    hidden = edge_hidden.float()
    directions = edge_direction.float()
    envelope = edge_envelope.float()
    coefficients = torch.tanh(
        F.linear(hidden, module.moment_projection.weight.float())  # type: ignore[attr-defined]
    )
    channels = int(module.moment_channels)  # type: ignore[attr-defined]
    first_coefficients, second_coefficients = coefficients.split(channels, dim=-1)
    degree = torch.bincount(edge_target, minlength=vectors.shape[0]).float()
    degree_scale = degree.clamp_min(1.0).rsqrt()
    first_messages = (
        first_coefficients[:, :, None]
        * envelope[:, :, None]
        * directions[:, None, :]
    )
    first = vectors.new_zeros((vectors.shape[0], channels, 3))
    first.index_add_(0, edge_target, first_messages)
    first = first * degree_scale[:, None, None]
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    dyad = torch.einsum("ei,ej->eij", directions, directions) - identity / 3.0
    second_messages = (
        second_coefficients[:, :, None, None]
        * envelope[:, :, None, None]
        * dyad[:, None, :, :]
    )
    second = vectors.new_zeros((vectors.shape[0], channels, 3, 3))
    second.index_add_(0, edge_target, second_messages)
    second = second * degree_scale[:, None, None, None]
    epsilon = float(module.rms_epsilon)  # type: ignore[attr-defined]
    vectors, vector_scale = _rms_normalize(
        vectors, batch, graphs, 3.0, epsilon, detach_scale=detach_scale
    )
    first, first_scale = _rms_normalize(
        first, batch, graphs, 3.0, epsilon, detach_scale=detach_scale
    )
    second, second_scale = _rms_normalize(
        second, batch, graphs, 5.0, epsilon, detach_scale=detach_scale
    )
    second_first = torch.einsum("ncij,ncj->nci", second, first)
    second_first, q1_scale = _rms_normalize(
        second_first, batch, graphs, 3.0, epsilon, detach_scale=detach_scale
    )
    second_squared_first = torch.einsum("ncij,ncj->nci", second, second_first)
    second_squared_first, q2_scale = _rms_normalize(
        second_squared_first,
        batch,
        graphs,
        3.0,
        epsilon,
        detach_scale=detach_scale,
    )
    carrier = torch.cat(
        (vectors, first, second_first, second_squared_first), dim=1
    )
    carrier = carrier - graph_mean(carrier, batch, graphs)[batch]

    def statistics(scale: torch.Tensor) -> dict[str, float]:
        return {
            "minimum": float(scale.detach().min()),
            "median": float(scale.detach().median()),
            "maximum": float(scale.detach().max()),
            "fraction_at_half_max_gain": float(
                (scale.detach() >= 0.5 / math.sqrt(epsilon)).float().mean()
            ),
        }

    return carrier, {
        "vector": statistics(vector_scale),
        "m": statistics(first_scale),
        "Q": statistics(second_scale),
        "Qm": statistics(q1_scale),
        "Q2m": statistics(q2_scale),
    }


def _forward_capture(
    model: nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
) -> tuple[Any, dict[str, Any]]:
    captures: dict[str, Any] = {}

    def carrier_pre(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        captures["carrier_inputs"] = inputs

    def carrier_post(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captures["carrier"] = output

    pre = model.coordinate_carrier.register_forward_pre_hook(carrier_pre)  # type: ignore[attr-defined]
    post = model.coordinate_carrier.register_forward_hook(carrier_post)  # type: ignore[attr-defined]
    graphs = int(batch_data.num_graphs)
    try:
        output = model(
            noisy.element_tokens,
            noisy.fractional_coordinates,
            noisy.log_volume,
            noisy.log_shape,
            batch_data.batch,
            noisy.time,
            noisy.time.new_zeros((graphs, 18)),
            torch.zeros((graphs, 1), dtype=torch.bool, device=noisy.time.device),
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
        )
    finally:
        pre.remove()
        post.remove()
    return output, captures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_krylov_gradient_attribution_v1":
        raise ValueError("Krylov gradient-attribution protocol mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("Krylov gradient-attribution cache mismatch")
    if int(protocol["audit"]["optimizer_steps"]) != 0:
        raise ValueError("Krylov gradient attribution forbids optimizer steps")
    if int(protocol["audit"]["coordinate_targets_read"]) != 0:
        raise ValueError("Krylov gradient attribution forbids coordinate targets")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Krylov gradient attribution requires CUDA")
    audit = protocol["audit"]
    workspace = str(audit["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
        raise RuntimeError(
            "Krylov gradient attribution requires "
            f"CUBLAS_WORKSPACE_CONFIG={workspace}"
        )
    torch.use_deterministic_algorithms(bool(audit["deterministic_algorithms"]))
    path = protocol["path"]
    torch.manual_seed(int(path["model_seed"]))
    torch.cuda.manual_seed_all(int(path["model_seed"]))
    model_spec = protocol["model"]
    model = HybridCrystalDenoiser(
        hidden_dim=int(model_spec["hidden_dim"]),
        vector_dim=int(model_spec["vector_dim"]),
        layers=int(model_spec["layers"]),
        radial_dim=int(model_spec["radial_dim"]),
        radial_cutoff=float(model_spec["radial_cutoff_angstrom"]),
    ).to(device).float()
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        protocol["prerequisites"]["candidate_parameter_count"]
    ):
        raise RuntimeError("loaded source is not the frozen Krylov candidate")
    if not hasattr(model, "coordinate_carrier"):
        raise RuntimeError("loaded source lacks the frozen Krylov carrier")

    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    indices = _fixed_indices(
        len(dataset),
        int(protocol["data"]["fixed_graphs"]),
        int(protocol["data"]["fixed_selection_seed"]),
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
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
    _, captures = _forward_capture(model, noisy, batch_data, blueprint)
    carrier_inputs = captures["carrier_inputs"]
    actual_carrier = captures["carrier"]
    carrier, scale_statistics = _diagnostic_carrier(
        model.coordinate_carrier,  # type: ignore[attr-defined]
        *carrier_inputs[:-1],
        int(carrier_inputs[-1]),
        detach_scale=False,
    )
    detached_carrier, _ = _diagnostic_carrier(
        model.coordinate_carrier,  # type: ignore[attr-defined]
        *carrier_inputs[:-1],
        int(carrier_inputs[-1]),
        detach_scale=True,
    )
    reconstruction = float((carrier - actual_carrier).abs().max())
    forward_detach_error = float((detached_carrier - carrier).abs().max())
    if reconstruction > float(protocol["audit"]["carrier_reconstruction_max_abs"]):
        raise RuntimeError(
            "diagnostic carrier does not reconstruct the frozen candidate: "
            f"max_abs={reconstruction:.9g}"
        )

    head_weight = model.coordinate_carrier_head.weight.float().reshape(-1)  # type: ignore[attr-defined]
    splits = list(torch.split(head_weight, list(model_spec["carrier_splits"])))
    carrier_parts = list(torch.split(carrier, list(model_spec["carrier_splits"]), dim=1))
    labels = ("vector", "m", "Qm", "Q2m")
    cartesian_parts = {
        label: torch.einsum("nci,c->ni", part, weight)
        for label, part, weight in zip(labels, carrier_parts, splits, strict=True)
    }
    cartesian = sum(cartesian_parts.values())
    cartesian = cartesian - graph_mean(
        cartesian, batch_data.batch, int(batch_data.num_graphs)
    )[batch_data.batch]
    detached_cartesian = torch.einsum("nci,c->ni", detached_carrier, head_weight)
    detached_cartesian = detached_cartesian - graph_mean(
        detached_cartesian, batch_data.batch, int(batch_data.num_graphs)
    )[batch_data.batch]
    lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
        blueprint.fractional_to_cartesian
    )

    def fractional(value: torch.Tensor) -> torch.Tensor:
        output = torch.einsum(
            "ni,nij->nj",
            value,
            lattice[batch_data.batch].transpose(-1, -2),
        )
        return output - graph_mean(
            output, batch_data.batch, int(batch_data.num_graphs)
        )[batch_data.batch]

    probe = torch.randn(
        carrier.shape[1],
        generator=torch.Generator().manual_seed(int(protocol["audit"]["probe_seed"])),
    ).to(device)
    probe = probe / torch.linalg.vector_norm(probe)
    probe_field = torch.einsum("nci,c->ni", carrier, probe)
    stages = {
        "carrier_probe": _gradient_metrics(
            probe_field.square().mean(), model, retain_graph=True
        ),
        "cartesian_output": _gradient_metrics(
            cartesian.square().mean(), model, retain_graph=True
        ),
        "fractional_output": _gradient_metrics(
            fractional(cartesian).square().mean(), model, retain_graph=True
        ),
        "fractional_detached_rms": _gradient_metrics(
            fractional(detached_cartesian).square().mean(), model, retain_graph=True
        ),
    }
    order_metrics = {
        label: _gradient_metrics(
            fractional(value).square().mean(), model, retain_graph=True
        )
        for label, value in cartesian_parts.items()
    }
    full_norm = float(stages["fractional_output"]["gradient_norm"])
    cartesian_norm = float(stages["cartesian_output"]["gradient_norm"])
    detached_norm = float(stages["fractional_detached_rms"]["gradient_norm"])
    rms_amplification = full_norm / max(detached_norm, 1e-30)
    chart_amplification = full_norm / max(cartesian_norm, 1e-30)
    q2m_over_full = float(order_metrics["Q2m"]["gradient_norm"]) / max(full_norm, 1e-30)
    group_fractions = stages["fractional_output"]["group_squared_fractions"]
    maximum_group = max(group_fractions, key=group_fractions.get)
    maximum_group_fraction = float(group_fractions[maximum_group])
    thresholds = {
        key: float(value) for key, value in protocol["decision_thresholds"].items()
    }
    attribution = classify_attribution(
        rms_amplification,
        chart_amplification,
        q2m_over_full,
        maximum_group_fraction,
        thresholds,
    )
    result = {
        "protocol": protocol["protocol"],
        "candidate_commit": protocol["prerequisites"]["candidate_commit"],
        "fixed_indices": indices.tolist(),
        "carrier_reconstruction_max_abs": reconstruction,
        "detached_rms_forward_max_abs": forward_detach_error,
        "scale_statistics": scale_statistics,
        "lattice_spectral_norm": {
            "minimum": float(torch.linalg.matrix_norm(lattice, ord=2).min()),
            "median": float(torch.linalg.matrix_norm(lattice, ord=2).median()),
            "maximum": float(torch.linalg.matrix_norm(lattice, ord=2).max()),
        },
        "stages": stages,
        "carrier_orders": order_metrics,
        "rms_derivative_amplification": rms_amplification,
        "fractional_over_cartesian_gradient": chart_amplification,
        "q2m_over_full_gradient": q2m_over_full,
        "maximum_parameter_group": maximum_group,
        "maximum_parameter_group_squared_fraction": maximum_group_fraction,
        "attribution": attribution,
        "optimizer_steps": 0,
        "coordinate_targets_read": 0,
        "deterministic_algorithms": bool(audit["deterministic_algorithms"]),
        "cublas_workspace_config": workspace,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
