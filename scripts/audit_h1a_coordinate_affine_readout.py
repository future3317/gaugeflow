"""Audit the exact affine coordinate-readout span on one fixed H1a state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _make_batch,
    _make_model,
    _predict,
)


def project_common_translation(value: torch.Tensor) -> torch.Tensor:
    """Project a one-graph coordinate field to the translation quotient."""
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError("coordinate field must have shape [nodes, 3]")
    return value - value.mean(dim=0, keepdim=True)


def affine_readout_solution(
    jacobian: torch.Tensor,
    desired_change: torch.Tensor,
    *,
    relative_threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the minimum-norm active-subspace solution and span metrics."""
    if jacobian.ndim != 2:
        raise ValueError("readout Jacobian must be a matrix")
    if desired_change.shape != (jacobian.shape[0],):
        raise ValueError("desired output change does not match the Jacobian")
    if relative_threshold <= 0.0:
        raise ValueError("relative rank threshold must be positive")
    jacobian64 = jacobian.double()
    gram = jacobian64 @ jacobian64.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[-1].clamp_min(torch.finfo(torch.float64).tiny)
    active = eigenvalues > relative_threshold * maximum
    active_vectors = eigenvectors[:, active]
    active_values = eigenvalues[active]
    desired64 = desired_change.double()
    projection = active_vectors @ (active_vectors.T @ desired64)
    coefficients = active_vectors @ (
        (active_vectors.T @ desired64) / active_values
    )
    delta = jacobian64.T @ coefficients
    residual = torch.linalg.vector_norm(desired64 - projection) / torch.linalg.vector_norm(
        desired64
    ).clamp_min(1e-30)
    normalized = active_values / active_values.sum().clamp_min(1e-30)
    effective_rank = (
        float(torch.exp(-(normalized * normalized.log()).sum()))
        if active_values.numel()
        else 0.0
    )
    return delta, {
        "output_dimension": int(jacobian.shape[0]),
        "parameter_dimension": int(jacobian.shape[1]),
        "rank": int(active.sum()),
        "nullity": int((~active).sum()),
        "maximum_eigenvalue": float(maximum),
        "minimum_active_eigenvalue": (
            float(active_values[0]) if active_values.numel() else 0.0
        ),
        "condition_number": (
            float(maximum / active_values[0]) if active_values.numel() else None
        ),
        "effective_rank": effective_rank,
        "target_projection_relative_residual": float(residual),
        "eigenvalues": eigenvalues.tolist(),
    }


def _readout_parameters(
    model: torch.nn.Module, expected_names: list[str]
) -> list[tuple[str, torch.nn.Parameter]]:
    available = dict(model.named_parameters())
    missing = [name for name in expected_names if name not in available]
    if missing:
        raise ValueError(f"coordinate readout parameters are missing: {missing}")
    return [(name, available[name]) for name in expected_names]


def _jacobian(
    output: torch.Tensor, parameters: list[tuple[str, torch.nn.Parameter]]
) -> torch.Tensor:
    values = [value for _, value in parameters]
    rows: list[torch.Tensor] = []
    for component in range(output.numel()):
        gradients = torch.autograd.grad(
            output.reshape(-1)[component], values, retain_graph=True
        )
        rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]).detach())
    return torch.stack(rows)


def _column_slices(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for name, value in parameters:
        result[name] = slice(offset, offset + value.numel())
        offset += value.numel()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_coordinate_affine_readout_audit_v1":
        raise ValueError("coordinate affine-readout protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate affine-readout cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed = int(protocol["prerequisites"]["model_seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    graph_index = int(protocol["prerequisites"]["fixed_graph_index"])
    batch_data = _make_batch(dataset, torch.tensor([graph_index]), device)
    if int(batch_data.num_nodes) != int(protocol["prerequisites"]["fixed_graph_nodes"]):
        raise ValueError("coordinate affine-readout fixed graph changed")
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float().eval()
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(args.lattice_standardization)
    )
    path = protocol["path"]
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(path["coordinate_sigma_min"]),
        coordinate_sigma_max=float(path["coordinate_sigma_max"]),
        minimum_time=float(path["minimum_time"]),
        maximum_time=float(path["maximum_time"]),
    )
    time = batch_data.lattice.new_tensor(
        [float(protocol["prerequisites"]["fixed_time"])]
    )
    noisy = diffusion.noise_clean_batch(
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=time,
        generator=torch.Generator(device=device).manual_seed(
            int(protocol["prerequisites"]["noise_seed"])
        ),
    )

    def predict() -> torch.Tensor:
        return project_common_translation(
            _predict(model, noisy, batch_data, blueprint, use_bf16=False)
        )

    prediction = predict()
    target = project_common_translation(noisy.coordinate_scaled_score_target)
    desired = (target - prediction).reshape(-1)
    readout = protocol["readout"]
    parameters = _readout_parameters(model, list(readout["parameter_names"]))
    parameter_count = sum(value.numel() for _, value in parameters)
    if parameter_count != int(readout["parameter_count"]):
        raise ValueError("coordinate affine-readout parameter count mismatch")
    jacobian = _jacobian(prediction, parameters)
    threshold = float(readout["rank_relative_eigenvalue_threshold"])
    delta, combined = affine_readout_solution(
        jacobian, desired.detach(), relative_threshold=threshold
    )
    slices = _column_slices(parameters)
    groups = {
        "vector": ["coordinate_vector_head.weight"],
        "edge": [
            "coordinate_edge_head.2.weight",
            "coordinate_edge_head.2.bias",
        ],
    }
    group_metrics: dict[str, dict[str, Any]] = {}
    for group, names in groups.items():
        columns = torch.cat(
            [
                torch.arange(
                    slices[name].start,
                    slices[name].stop,
                    device=jacobian.device,
                )
                for name in names
            ]
        )
        _, group_metrics[group] = affine_readout_solution(
            jacobian[:, columns], desired.detach(), relative_threshold=threshold
        )
    originals = [value.detach().clone() for _, value in parameters]
    offset = 0
    try:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                count = parameter.numel()
                parameter.copy_(
                    original + delta[offset : offset + count].reshape_as(parameter)
                )
                offset += count
        with torch.no_grad():
            actual = predict().reshape(-1).float()
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
    restored = all(
        torch.equal(value.detach(), original)
        for (_, value), original in zip(parameters, originals, strict=True)
    )
    initial = prediction.reshape(-1).detach()
    target_flat = target.reshape(-1).detach()
    linear = initial.double() + jacobian.double() @ delta
    affine_error = torch.linalg.vector_norm(actual.double() - linear) / torch.linalg.vector_norm(
        desired.double()
    ).clamp_min(1e-30)
    actual_mse = float((actual.double() - target_flat.double()).square().mean())
    acceptance = protocol["acceptance"]
    checks = {
        "quotient_rank": int(combined["rank"]) == int(acceptance["quotient_rank"]),
        "target_projection": float(combined["target_projection_relative_residual"])
        <= float(acceptance["target_projection_relative_residual_max"]),
        "actual_coordinate_mse": actual_mse
        <= float(acceptance["actual_coordinate_mse_max"]),
        "affine_forward": float(affine_error)
        <= float(acceptance["affine_forward_relative_error_max"]),
        "parameters_restored_exactly": restored
        is bool(acceptance["parameters_restored_exactly"]),
        "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
    }
    if not checks["quotient_rank"] or not checks["target_projection"]:
        decision = "affine_coordinate_features_do_not_span_target"
    elif not checks["affine_forward"]:
        decision = "declared_coordinate_readout_not_numerically_affine"
    elif all(checks.values()):
        decision = "affine_readout_spans_target_qualify_two_timescale_optimization"
    else:
        decision = "affine_readout_qualification_failed"
    result = {
        "protocol": protocol["protocol"],
        "graph_index": graph_index,
        "nodes": int(batch_data.num_nodes),
        "initial_coordinate_mse": float(
            (initial.double() - target_flat.double()).square().mean()
        ),
        "actual_coordinate_mse": actual_mse,
        "linear_coordinate_mse": float(
            (linear - target_flat.double()).square().mean()
        ),
        "affine_forward_relative_error": float(affine_error),
        "readout_step_norm": float(torch.linalg.vector_norm(delta)),
        "readout_parameter_norm": float(
            torch.linalg.vector_norm(
                torch.cat([value.reshape(-1) for value in originals]).double()
            )
        ),
        "combined": combined,
        "groups": group_metrics,
        "checks": checks,
        "decision": decision,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
