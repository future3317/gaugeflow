"""Audit exact affine coordinate-readout fits on nested fixed H1a panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.geometry import periodic_radius_multigraph
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.state_projection import graph_mean
from scripts.audit_h1a_coordinate_memorization import (
    _blueprint,
    _endpoint_rms,
    _fixed_indices,
    _make_batch,
    _make_model,
    _predict,
)


def weighted_affine_fit(
    design: torch.Tensor,
    target: torch.Tensor,
    row_graph: torch.Tensor,
    graph_count: int,
    *,
    rcond: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Solve graph-equal least squares and report the weighted spectrum."""
    if design.ndim != 2 or target.shape != (design.shape[0],):
        raise ValueError("affine design and target shapes do not match")
    if row_graph.shape != target.shape:
        raise ValueError("each affine row needs one graph index")
    if graph_count < 1 or rcond <= 0.0:
        raise ValueError("invalid affine panel fit configuration")
    counts = torch.bincount(row_graph, minlength=graph_count).double()
    weights = counts[row_graph].rsqrt()
    weighted_design = design.double() * weights[:, None]
    weighted_target = target.double() * weights
    solution = torch.linalg.lstsq(
        weighted_design.cpu(), weighted_target.cpu(), rcond=rcond, driver="gelsd"
    ).solution.to(design.device)
    singular_values = torch.linalg.svdvals(weighted_design.cpu())
    active = singular_values > rcond * singular_values[0].clamp_min(1e-30)
    return solution, {
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "rank": int(active.sum()),
        "maximum_singular_value": float(singular_values[0]),
        "minimum_active_singular_value": (
            float(singular_values[active][-1]) if active.any() else 0.0
        ),
        "condition_number": (
            float(singular_values[0] / singular_values[active][-1])
            if active.any()
            else math.inf
        ),
    }


def _capture_affine_design(
    model: torch.nn.Module,
    noisy: Any,
    batch_data: Any,
    blueprint: Any,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    captures: dict[str, torch.Tensor] = {}

    def vector_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["vector"] = inputs[0].detach()

    def edge_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captures["edge"] = inputs[0].detach()

    vector_handle = model.coordinate_vector_head.register_forward_pre_hook(vector_hook)
    edge_handle = model.coordinate_edge_head[2].register_forward_pre_hook(edge_hook)
    try:
        with torch.no_grad():
            prediction = _predict(
                model, noisy, batch_data, blueprint, use_bf16=False
            ).detach()
    finally:
        vector_handle.remove()
        edge_handle.remove()
    vector_input = captures["vector"]
    edge_hidden = captures["edge"]
    graphs = int(batch_data.num_graphs)
    with torch.no_grad():
        lattice = LatticeVolumeShape(noisy.log_volume, noisy.log_shape).lattice(
            blueprint.fractional_to_cartesian
        )
        edges = periodic_radius_multigraph(
            noisy.fractional_coordinates,
            lattice,
            batch_data.batch,
            cutoff=float(model.radial.cutoff),
        )
        if edge_hidden.shape[0] != edges.source.numel():
            raise RuntimeError("captured edge basis does not match production edge order")
        envelope = model.radial.envelope(edges.distance)
        degree = torch.bincount(
            edges.target, minlength=int(batch_data.num_nodes)
        ).to(lattice)
        vector_basis = vector_input.transpose(-1, -2)
        edge_messages = (
            edge_hidden[:, :, None]
            * envelope[:, :, None]
            * edges.displacement[:, None, :]
        )
        edge_basis = vector_basis.new_zeros(
            (int(batch_data.num_nodes), edge_hidden.shape[1], 3)
        )
        edge_basis.index_add_(0, edges.target, edge_messages.to(edge_basis.dtype))
        edge_basis = edge_basis / degree.clamp_min(1).sqrt()[:, None, None]
        bias_messages = envelope * edges.displacement
        bias_basis = vector_basis.new_zeros((int(batch_data.num_nodes), 1, 3))
        bias_basis.index_add_(
            0, edges.target, bias_messages[:, None, :].to(bias_basis.dtype)
        )
        bias_basis = bias_basis / degree.clamp_min(1).sqrt()[:, None, None]
        cartesian_basis = torch.cat((vector_basis, edge_basis, bias_basis), dim=1)
        cartesian_basis = cartesian_basis - graph_mean(
            cartesian_basis, batch_data.batch, graphs
        )[batch_data.batch]
        fractional_basis = torch.einsum(
            "nci,nji->ncj", cartesian_basis, lattice[batch_data.batch]
        )
        fractional_basis = fractional_basis - graph_mean(
            fractional_basis, batch_data.batch, graphs
        )[batch_data.batch]
        design = fractional_basis.transpose(1, 2).reshape(
            3 * int(batch_data.num_nodes), -1
        )
        weights = torch.cat(
            (
                model.coordinate_vector_head.weight.reshape(-1),
                model.coordinate_edge_head[2].weight.reshape(-1),
                model.coordinate_edge_head[2].bias.reshape(-1),
            )
        )
        reconstruction = (design @ weights).reshape_as(prediction)
        error = float((reconstruction - prediction).abs().max())
    return design, prediction, error


def _panel_metrics(
    design: torch.Tensor,
    target: torch.Tensor,
    node_batch: torch.Tensor,
    panel_graphs: int,
    *,
    solution: torch.Tensor,
) -> dict[str, float]:
    node_mask = node_batch < panel_graphs
    row_mask = node_mask[:, None].expand(-1, 3).reshape(-1)
    prediction = (design.double() @ solution).reshape(-1, 3)
    error = prediction[node_mask] - target[node_mask].double()
    graph_error = scatter(
        error.square().sum(-1),
        node_batch[node_mask],
        dim=0,
        dim_size=panel_graphs,
        reduce="mean",
    )
    graph_target = scatter(
        target[node_mask].double().square().sum(-1),
        node_batch[node_mask],
        dim=0,
        dim_size=panel_graphs,
        reduce="mean",
    )
    mse = graph_error.mean() / 3.0
    zero_mse = graph_target.mean() / 3.0
    return {
        "coordinate_mse": float(mse),
        "zero_predictor_mse": float(zero_mse),
        "explained_fraction": float(1.0 - mse / zero_mse.clamp_min(1e-30)),
        "row_count": float(row_mask.sum()),
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
    if protocol.get("protocol") != "h1a_coordinate_readout_panel_audit_v1":
        raise ValueError("coordinate readout-panel protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate readout-panel cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    path = protocol["path"]
    torch.manual_seed(int(path["model_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(path["model_seed"]))
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    data = protocol["data"]
    indices = _fixed_indices(
        len(dataset), int(data["fixed_graphs"]), int(data["fixed_selection_seed"])
    )
    batch_data = _make_batch(dataset, indices, device)
    blueprint = _blueprint(batch_data)
    model = _make_model(protocol, device).float().eval()
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
    design, initial_prediction, reconstruction_error = _capture_affine_design(
        model, noisy, batch_data, blueprint
    )
    if design.shape[1] != int(protocol["readout"]["parameter_count"]):
        raise ValueError("captured affine readout has the wrong parameter count")
    target = noisy.coordinate_scaled_score_target.detach()
    panels: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, float]] = []
    for panel_graphs in data["panels"]:
        panel_graphs = int(panel_graphs)
        node_mask = batch_data.batch < panel_graphs
        row_mask = node_mask[:, None].expand(-1, 3).reshape(-1)
        row_graph = batch_data.batch[:, None].expand(-1, 3).reshape(-1)[row_mask]
        solution, spectrum = weighted_affine_fit(
            design[row_mask],
            target.reshape(-1)[row_mask],
            row_graph,
            panel_graphs,
            rcond=float(protocol["readout"]["lstsq_rcond"]),
        )
        metrics = _panel_metrics(
            design,
            target,
            batch_data.batch,
            panel_graphs,
            solution=solution,
        )
        full_prediction = (design.double() @ solution).reshape_as(target).float()
        endpoint = _endpoint_rms(
            full_prediction,
            noisy,
            batch_data.frac_coords,
            batch_data.lattice,
            batch_data.batch,
            diffusion,
        )
        panel_time = noisy.time[:panel_graphs]
        low = panel_time <= 0.02
        metrics["endpoint_rms_angstrom"] = float(
            endpoint[:panel_graphs].square().mean().sqrt()
        )
        metrics["low_time_endpoint_rms_angstrom"] = float(
            endpoint[:panel_graphs][low].square().mean().sqrt()
        )
        endpoint_rows.append(
            {
                "graphs": float(panel_graphs),
                "endpoint_rms_angstrom": metrics["endpoint_rms_angstrom"],
            }
        )
        panels.append(
            {"graphs": panel_graphs, "metrics": metrics, "spectrum": spectrum}
        )
    acceptance = protocol["acceptance"]
    thresholds = {
        1: float(acceptance["one_graph_coordinate_mse_max"]),
        4: float(acceptance["four_graph_coordinate_mse_max"]),
        16: float(acceptance["sixteen_graph_coordinate_mse_max"]),
        64: float(acceptance["sixty_four_graph_coordinate_mse_max"]),
    }
    panel_checks = {
        str(row["graphs"]): float(row["metrics"]["coordinate_mse"])
        <= thresholds[int(row["graphs"])]
        for row in panels
    }
    design_check = reconstruction_error <= float(
        acceptance["design_reconstruction_max_abs"]
    )
    if not design_check or not panel_checks["1"]:
        decision = "affine_design_or_single_state_fit_contradiction"
    elif all(panel_checks.values()):
        decision = "frozen_features_support_training_only_readout_calibration"
    else:
        decision = "small_panels_fit_large_panel_requires_backbone_feature_learning"
    initial_mse = scatter(
        (initial_prediction - target).square().sum(-1),
        batch_data.batch,
        dim=0,
        dim_size=int(batch_data.num_graphs),
        reduce="mean",
    ).mean() / 3.0
    result = {
        "protocol": protocol["protocol"],
        "fixed_indices": indices.tolist(),
        "design_shape": list(design.shape),
        "design_reconstruction_max_abs": reconstruction_error,
        "initial_coordinate_mse": float(initial_mse),
        "panels": panels,
        "endpoint_summary": endpoint_rows,
        "checks": {
            "design_reconstruction": design_check,
            "panels": panel_checks,
            "tensor_candidates": int(acceptance["tensor_candidates"]) == 0,
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
