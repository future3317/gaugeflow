"""Audit the quotient-output tangent rank and conditioning on one fixed state."""

from __future__ import annotations

import argparse
import json
import math
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


def tangent_spectrum_metrics(
    gram: torch.Tensor,
    desired_change: torch.Tensor,
    *,
    relative_threshold: float,
) -> dict[str, Any]:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("tangent Gram matrix must be square")
    if desired_change.shape != (gram.shape[0],):
        raise ValueError("desired output change does not match tangent Gram matrix")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    maximum = eigenvalues[-1].clamp_min(torch.finfo(torch.float64).tiny)
    active = eigenvalues > float(relative_threshold) * maximum
    projection = eigenvectors[:, active] @ (eigenvectors[:, active].T @ desired_change.double())
    residual = torch.linalg.vector_norm(desired_change.double() - projection)
    denominator = torch.linalg.vector_norm(desired_change.double()).clamp_min(1e-30)
    positive = eigenvalues[active]
    condition = float(maximum / positive[0]) if positive.numel() else math.inf
    normalized = positive / positive.sum().clamp_min(1e-30)
    entropy_rank = float(torch.exp(-(normalized * normalized.log()).sum())) if positive.numel() else 0.0
    return {
        "output_dimension": int(gram.shape[0]),
        "tangent_rank": int(active.sum()),
        "nullity": int((~active).sum()),
        "maximum_eigenvalue": float(maximum),
        "minimum_active_eigenvalue": float(positive[0]) if positive.numel() else 0.0,
        "condition_number": condition,
        "effective_rank": entropy_rank,
        "target_projection_relative_residual": float(residual / denominator),
        "eigenvalues": eigenvalues.tolist(),
    }


def _parameter_group(name: str) -> str:
    for prefix in (
        "coordinate_vector_head",
        "coordinate_control_gate",
        "coordinate_edge_head",
        "blocks",
        "element_embedding",
        "degree_embedding",
        "time_embedding",
        "state_embedding",
    ):
        if name.startswith(prefix):
            return prefix
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_coordinate_tangent_audit_v1":
        raise ValueError("coordinate tangent protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate tangent cache mismatch")
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
        raise ValueError("coordinate tangent fixed graph changed")
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float()
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
    time = batch_data.lattice.new_tensor([float(protocol["prerequisites"]["fixed_time"])])
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
    prediction = _predict(model, noisy, batch_data, blueprint, use_bf16=False)
    target = noisy.coordinate_scaled_score_target
    desired = (target - prediction).reshape(-1)
    parameters = [
        (name, value)
        for name, value in model.named_parameters()
        if value.requires_grad
        and not name.startswith(("gauge_atlas.", "geometry_query_encoder."))
    ]
    rows: list[torch.Tensor] = []
    for component in range(prediction.numel()):
        gradients = torch.autograd.grad(
            prediction.reshape(-1)[component],
            [value for _, value in parameters],
            retain_graph=True,
            allow_unused=True,
        )
        rows.append(
            torch.cat(
                [
                    (torch.zeros_like(value) if gradient is None else gradient).reshape(-1)
                    for (_, value), gradient in zip(parameters, gradients, strict=True)
                ]
            ).detach()
        )
    jacobian = torch.stack(rows)
    gram = jacobian @ jacobian.T
    numeric = protocol["numeric"]
    spectrum = tangent_spectrum_metrics(
        gram,
        desired.detach(),
        relative_threshold=float(numeric["rank_relative_eigenvalue_threshold"]),
    )
    loss = (prediction - target).square().mean()
    loss_gradients = torch.autograd.grad(
        loss, [value for _, value in parameters], allow_unused=True
    )
    group_energy: dict[str, float] = {}
    for (name, _), gradient in zip(parameters, loss_gradients, strict=True):
        if gradient is None:
            continue
        group = _parameter_group(name)
        group_energy[group] = group_energy.get(group, 0.0) + float(
            gradient.detach().double().square().sum()
        )
    group_norm = {name: math.sqrt(value) for name, value in group_energy.items()}
    quotient_dimension = 3 * int(batch_data.num_nodes) - 3
    rank_full = int(spectrum["tangent_rank"]) >= quotient_dimension
    target_reachable = (
        float(spectrum["target_projection_relative_residual"])
        <= float(numeric["target_projection_relative_residual_max"])
    )
    ill_conditioned = (
        float(spectrum["condition_number"])
        >= float(numeric["severe_ntk_condition_number_min"])
    )
    if not rank_full or not target_reachable:
        decision = "rank_or_target_projection_deficient_repair_vector_output_basis"
    elif ill_conditioned:
        decision = "full_rank_but_ill_conditioned_qualify_output_preconditioning"
    else:
        decision = "full_rank_well_conditioned_inspect_optimizer_nonlinear_curvature"
    result = {
        "protocol": protocol["protocol"],
        "graph_index": graph_index,
        "nodes": int(batch_data.num_nodes),
        "quotient_dimension": quotient_dimension,
        "initial_coordinate_mse": float(loss),
        "prediction_norm": float(torch.linalg.vector_norm(prediction)),
        "target_norm": float(torch.linalg.vector_norm(target)),
        "prediction_target_cosine": float(
            (prediction * target).sum()
            / (
                torch.linalg.vector_norm(prediction)
                * torch.linalg.vector_norm(target)
            ).clamp_min(1e-30)
        ),
        "tangent": spectrum,
        "module_gradient_norm": group_norm,
        "checks": {
            "quotient_tangent_full_rank": rank_full,
            "target_direction_reachable": target_reachable,
            "severely_ill_conditioned": ill_conditioned,
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
