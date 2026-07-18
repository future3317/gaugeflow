"""Audit one output-space Gauss--Newton trust step on a fixed H1a state."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

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
    """Project one graph's fractional vector field to the translation quotient."""
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError("coordinate field must have shape [nodes, 3]")
    return value - value.mean(dim=0, keepdim=True)


def damped_output_coefficients(
    gram: torch.Tensor,
    desired_change: torch.Tensor,
    *,
    relative_damping: float,
) -> tuple[torch.Tensor, float]:
    """Solve the damped output-space normal equation in FP64."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("output Gram matrix must be square")
    if desired_change.shape != (gram.shape[0],):
        raise ValueError("desired output change does not match the Gram matrix")
    if relative_damping <= 0.0:
        raise ValueError("relative damping must be positive")
    gram64 = gram.double()
    maximum = torch.linalg.eigvalsh(gram64)[-1].clamp_min(
        torch.finfo(torch.float64).tiny
    )
    damping = float(maximum * relative_damping)
    identity = torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device)
    coefficients = torch.linalg.solve(
        gram64 + damping * identity, desired_change.double()
    )
    return coefficients, damping


def relative_loss_reduction(initial: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return the fraction of initial squared error removed by a candidate."""
    initial_loss = initial.double().square().mean().clamp_min(1e-30)
    candidate_loss = candidate.double().square().mean()
    return float(1.0 - candidate_loss / initial_loss)


def _active_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, value)
        for name, value in model.named_parameters()
        if value.requires_grad
        and not name.startswith(("gauge_atlas.", "geometry_query_encoder."))
    ]


def _jacobian(
    output: torch.Tensor, parameters: list[tuple[str, torch.nn.Parameter]]
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    values = [value for _, value in parameters]
    for component in range(output.numel()):
        gradients = torch.autograd.grad(
            output.reshape(-1)[component],
            values,
            retain_graph=True,
            allow_unused=True,
        )
        rows.append(
            torch.cat(
                [
                    (torch.zeros_like(value) if gradient is None else gradient)
                    .reshape(-1)
                    .detach()
                    for (_, value), gradient in zip(parameters, gradients, strict=True)
                ]
            )
        )
    return torch.stack(rows)


def _evaluate_steps(
    model: torch.nn.Module,
    parameters: list[tuple[str, torch.nn.Parameter]],
    delta: torch.Tensor,
    jacobian_delta: torch.Tensor,
    initial: torch.Tensor,
    target: torch.Tensor,
    predict: Callable[[], torch.Tensor],
    radius_scale_pairs: list[tuple[float, float]],
) -> tuple[list[dict[str, float | bool]], bool]:
    originals = [value.detach().clone() for _, value in parameters]
    initial_error = initial - target
    rows: list[dict[str, float | bool]] = []
    try:
        for radius, scale in radius_scale_pairs:
            offset = 0
            with torch.no_grad():
                for (_, parameter), original in zip(parameters, originals, strict=True):
                    count = parameter.numel()
                    parameter.copy_(
                        original
                        + float(scale)
                        * delta[offset : offset + count].reshape_as(parameter)
                    )
                    offset += count
            with torch.no_grad():
                actual = project_common_translation(predict()).reshape(-1).float()
            linear = initial + float(scale) * jacobian_delta
            linear_change = linear - initial
            mismatch = torch.linalg.vector_norm(
                (actual - linear).double()
            ) / torch.linalg.vector_norm(linear_change.double()).clamp_min(1e-30)
            actual_error = actual - target
            linear_error = linear - target
            rows.append(
                {
                    "relative_parameter_norm_radius": float(radius),
                    "scale": float(scale),
                    "finite": bool(torch.isfinite(actual).all()),
                    "linear_coordinate_mse": float(linear_error.double().square().mean()),
                    "actual_coordinate_mse": float(actual_error.double().square().mean()),
                    "linear_loss_reduction": relative_loss_reduction(
                        initial_error, linear_error
                    ),
                    "actual_loss_reduction": relative_loss_reduction(
                        initial_error, actual_error
                    ),
                    "linearization_relative_error": float(mismatch),
                }
            )
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
    restored = all(
        torch.equal(value.detach(), original)
        for (_, value), original in zip(parameters, originals, strict=True)
    )
    return rows, restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_coordinate_ntk_trust_radius_audit_v2":
        raise ValueError("coordinate NTK trust-radius protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate NTK trust-step cache mismatch")
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
        raise ValueError("coordinate trust-step fixed graph changed")
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
        return _predict(model, noisy, batch_data, blueprint, use_bf16=False)

    prediction = project_common_translation(predict())
    target = project_common_translation(noisy.coordinate_scaled_score_target)
    initial = prediction.reshape(-1)
    desired = (target - prediction).reshape(-1)
    parameters = _active_parameters(model)
    jacobian = _jacobian(initial, parameters)
    gram = jacobian.double() @ jacobian.double().T
    trust = protocol["trust_radius"]
    coefficients, damping = damped_output_coefficients(
        gram,
        desired.detach(),
        relative_damping=float(trust["damping_relative_to_max_gram_eigenvalue"]),
    )
    delta = jacobian.T @ coefficients.to(jacobian.dtype)
    jacobian_delta = (gram @ coefficients).to(initial.dtype)
    parameter_norm = torch.linalg.vector_norm(
        torch.cat([value.detach().reshape(-1) for _, value in parameters]).double()
    )
    delta_norm = torch.linalg.vector_norm(delta.double())
    relative_step_norm = float(delta_norm / parameter_norm.clamp_min(1e-30))
    radii = [float(value) for value in trust["relative_parameter_norm_radii"]]
    radius_scale_pairs = [
        (radius, min(1.0, radius / relative_step_norm)) for radius in radii
    ]
    steps, parameters_restored = _evaluate_steps(
        model,
        parameters,
        delta,
        jacobian_delta,
        initial.detach(),
        target.reshape(-1).detach(),
        predict,
        radius_scale_pairs,
    )
    first = steps[0]
    full_linear_error = initial.detach() + jacobian_delta - target.reshape(-1)
    full_linear_reduction = relative_loss_reduction(
        initial.detach() - target.reshape(-1), full_linear_error
    )
    linear_reachable = (
        full_linear_reduction >= float(trust["full_linear_loss_reduction_min"])
    )
    locally_consistent = (
        bool(first["finite"])
        and float(first["linearization_relative_error"])
        <= float(trust["smallest_radius_linearization_relative_error_max"])
    )
    useful = [
        row
        for row in steps
        if bool(row["finite"])
        and float(row["actual_loss_reduction"])
        >= float(trust["useful_actual_loss_reduction_min"])
    ]
    if not linear_reachable:
        decision = "damped_tangent_does_not_fit_target"
    elif not locally_consistent:
        decision = "local_linearization_or_numeric_path_inconsistent"
    elif useful:
        decision = "bounded_natural_gradient_effective_but_curvature_limited"
    else:
        decision = "linear_reachable_but_no_useful_trust_radius"
    best = max(steps, key=lambda row: float(row["actual_loss_reduction"]))
    result = {
        "protocol": protocol["protocol"],
        "graph_index": graph_index,
        "nodes": int(batch_data.num_nodes),
        "output_dimension": int(initial.numel()),
        "quotient_dimension": int(initial.numel() - 3),
        "initial_coordinate_mse": float(
            (initial - target.reshape(-1)).double().square().mean()
        ),
        "damping": damping,
        "maximum_gram_eigenvalue": float(torch.linalg.eigvalsh(gram)[-1]),
        "full_linear_loss_reduction": full_linear_reduction,
        "parameter_norm": float(parameter_norm),
        "gauss_newton_step_norm": float(delta_norm),
        "gauss_newton_relative_step_norm": relative_step_norm,
        "steps": steps,
        "best_preregistered_radius": best["relative_parameter_norm_radius"],
        "best_actual_loss_reduction": best["actual_loss_reduction"],
        "checks": {
            "linear_target_reachable": linear_reachable,
            "small_step_linearization_consistent": locally_consistent,
            "any_useful_nonlinear_step": bool(useful),
            "parameters_restored_exactly": parameters_restored,
            "tensor_candidates": 0,
        },
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
