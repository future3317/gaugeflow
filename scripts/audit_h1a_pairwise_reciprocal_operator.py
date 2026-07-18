"""Qualify the frozen H1a pairwise reciprocal-torus score operator."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import (
    PackedAlexP1Dataset,
    collate_packed_alex,
)
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.pairwise_reciprocal_score import (
    PairwiseReciprocalScore,
    complete_unordered_node_pairs,
    projective_reciprocal_ball,
)
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--lattice-standardization",
        type=Path,
        default=Path("configs/statistics/h1a_p1_lattice_standardization.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _active_head(
    operator: dict[str, Any],
    *,
    hidden_dim: int,
    dtype: torch.dtype,
) -> PairwiseReciprocalScore:
    torch.manual_seed(8702)
    head = PairwiseReciprocalScore(
        hidden_dim,
        pair_width=int(operator["pair_width"]),
        channels=int(operator["channels"]),
        radial_dim=int(operator["reciprocal_radial_dim"]),
        cutoff=float(operator["reciprocal_cutoff_inverse_angstrom"]),
    ).to(dtype=dtype)
    torch.nn.init.normal_(head.mode_channels[-1].weight, std=0.15)
    torch.nn.init.normal_(head.mode_channels[-1].bias, std=0.05)
    return head


def _numeric_input(hidden_dim: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(8701)
    nodes = torch.randn((7, hidden_dim), dtype=torch.float64, generator=generator)
    coordinates = torch.rand((7, 3), dtype=torch.float64, generator=generator)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    lattice = torch.tensor(
        [
            [[4.1, 0.2, 0.1], [0.4, 3.8, 0.3], [0.2, 0.5, 4.5]],
            [[3.7, 0.1, 0.2], [0.3, 4.4, 0.1], [0.4, 0.2, 3.9]],
        ],
        dtype=torch.float64,
    )
    return nodes, coordinates, lattice, batch


def _relative_error(observed: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(observed - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1.0e-30)
    )


def _explicit_projective_pair_reference(
    head: PairwiseReciprocalScore,
    nodes: torch.Tensor,
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Slow scalar-loop reference independent of production reductions."""
    graphs = lattice.shape[0]
    pairs = complete_unordered_node_pairs(batch, graphs)
    ball = projective_reciprocal_ball(lattice, head.radial.cutoff)
    projected = head.node_projection(nodes)
    pair_channels = head.pair_channels(
        torch.cat(
            (
                projected[pairs.first] + projected[pairs.second],
                projected[pairs.first] * projected[pairs.second],
            ),
            dim=-1,
        )
    )
    radial = head.radial(ball.norms.reshape(-1)).reshape(
        graphs, ball.norms.shape[1], -1
    )
    mode_channels = head.mode_channels(radial)
    counts = torch.bincount(batch, minlength=graphs)
    result = torch.zeros_like(coordinates)
    for pair_index in range(pairs.first.numel()):
        graph = int(pairs.graph[pair_index])
        first = int(pairs.first[pair_index])
        second = int(pairs.second[pair_index])
        modes = int(ball.mask[graph].sum())
        pair_value = torch.zeros(3, dtype=coordinates.dtype)
        for mode_index in range(modes):
            integer_mode = ball.integer_modes[graph, mode_index].to(coordinates)
            phase = 2.0 * math.pi * torch.dot(
                coordinates[first] - coordinates[second], integer_mode
            )
            channel = torch.dot(
                pair_channels[pair_index], mode_channels[graph, mode_index]
            ) / math.sqrt(head.channels)
            pair_value = pair_value + (
                channel
                * phase.sin()
                * ball.cartesian_covectors[graph, mode_index]
            )
        pair_value = (
            math.sqrt(2.0)
            * pair_value
            / math.sqrt(float(counts[graph]) * modes)
        )
        result[first] += pair_value
        result[second] -= pair_value
    return result


def _full_symmetric_ball_reference(
    head: PairwiseReciprocalScore,
    nodes: torch.Tensor,
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    graphs = lattice.shape[0]
    pairs = complete_unordered_node_pairs(batch, graphs)
    ball = projective_reciprocal_ball(lattice, head.radial.cutoff)
    projected = head.node_projection(nodes)
    pair_channels = head.pair_channels(
        torch.cat(
            (
                projected[pairs.first] + projected[pairs.second],
                projected[pairs.first] * projected[pairs.second],
            ),
            dim=-1,
        )
    )
    mode_features = head.radial(ball.norms.reshape(-1)).reshape(
        graphs, ball.norms.shape[1], -1
    )
    projective_channels = head.mode_channels(mode_features)
    integer_modes = torch.cat((ball.integer_modes, -ball.integer_modes), dim=1)
    covectors = torch.cat(
        (ball.cartesian_covectors, -ball.cartesian_covectors), dim=1
    )
    mode_channels = torch.cat((projective_channels, projective_channels), dim=1)
    mask = torch.cat((ball.mask, ball.mask), dim=1)
    phase = 2.0 * math.pi * torch.einsum(
        "pi,pki->pk",
        coordinates[pairs.first] - coordinates[pairs.second],
        integer_modes[pairs.graph].to(coordinates),
    )
    coefficient = torch.einsum(
        "pc,pkc->pk", pair_channels, mode_channels[pairs.graph]
    ) / math.sqrt(head.channels)
    coefficient = coefficient * phase.sin() * mask[pairs.graph].to(coefficient)
    pair_score = torch.einsum(
        "pk,pki->pi", coefficient, covectors[pairs.graph]
    )
    counts = torch.bincount(batch, minlength=graphs).to(pair_score)
    modes = mask.sum(-1).clamp_min(1).to(pair_score)
    pair_score = pair_score / (
        counts[pairs.graph] * modes[pairs.graph]
    ).sqrt().unsqueeze(-1)
    result = torch.zeros_like(coordinates)
    result.index_add_(0, pairs.first, pair_score)
    result.index_add_(0, pairs.second, -pair_score)
    return result


def _numeric_audit(
    operator: dict[str, Any],
    acceptance: dict[str, Any],
) -> tuple[dict[str, float | bool], dict[str, bool]]:
    hidden_dim = 12
    nodes, coordinates, lattice, batch = _numeric_input(hidden_dim)
    head = _active_head(operator, hidden_dim=hidden_dim, dtype=torch.float64)
    reference = head(nodes, coordinates, lattice, batch)

    translation = torch.tensor([0.271, -0.193, 0.417], dtype=torch.float64)
    translated = head(nodes, coordinates + translation, lattice, batch)
    integer_shift = torch.tensor(
        [
            [1, -2, 3], [0, 1, -1], [-3, 2, 0], [2, 0, 1],
            [-1, 1, 2], [3, -2, -1], [0, 4, -3],
        ],
        dtype=torch.float64,
    )
    represented = head(nodes, coordinates + integer_shift, lattice, batch)
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    permuted = head(nodes[permutation], coordinates[permutation], lattice, batch)

    matrix = torch.tensor(
        [[0.3, -0.8, 0.5], [0.7, 0.5, 0.4], [-0.6, 0.2, 0.9]],
        dtype=torch.float64,
    )
    orthogonal, _ = torch.linalg.qr(matrix)
    orthogonal[:, 0] *= -1.0
    rotated = head(nodes, coordinates, lattice @ orthogonal.T, batch)

    basis = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    transformed = head(
        nodes,
        coordinates @ torch.linalg.inv(basis),
        basis.unsqueeze(0) @ lattice,
        batch,
    )
    explicit = _explicit_projective_pair_reference(
        head, nodes, coordinates, lattice, batch
    )
    full_ball = _full_symmetric_ball_reference(
        head, nodes, coordinates, lattice, batch
    )
    graph_mean_max = max(
        float(reference[batch == graph].sum(0).abs().max()) for graph in range(2)
    )

    radius = torch.tensor(
        [head.radial.cutoff], dtype=torch.float64, requires_grad=True
    )
    cutoff_value = head.radial.envelope(radius).sum()
    cutoff_derivative = torch.autograd.grad(cutoff_value, radius)[0]
    nodes_gradient = nodes.detach().requires_grad_(True)
    coordinates_gradient = coordinates.detach().requires_grad_(True)
    lattice_gradient = lattice.detach().requires_grad_(True)
    differentiable = head(
        nodes_gradient, coordinates_gradient, lattice_gradient, batch
    ).square().mean()
    gradients = torch.autograd.grad(
        differentiable,
        (nodes_gradient, coordinates_gradient, lattice_gradient, *tuple(head.parameters())),
    )

    initial = PairwiseReciprocalScore(
        hidden_dim,
        pair_width=int(operator["pair_width"]),
        channels=int(operator["channels"]),
        radial_dim=int(operator["reciprocal_radial_dim"]),
        cutoff=float(operator["reciprocal_cutoff_inverse_angstrom"]),
    ).double()
    initial_output = initial(nodes, coordinates, lattice, batch)
    source_tree = ast.parse(inspect.getsource(__import__(
        "gaugeflow.production.pairwise_reciprocal_score", fromlist=["*"]
    )))
    loop_free = not any(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(source_tree)
    )

    metrics: dict[str, float | bool] = {
        "translation_covariance_max_fp64": float((translated - reference).abs().max()),
        "periodic_representative_error_max_fp64": float((represented - reference).abs().max()),
        "node_permutation_equivariance_max_fp64": float(
            (permuted - reference[permutation]).abs().max()
        ),
        "physical_o3_covariance_max_fp64": float(
            (rotated - reference @ orthogonal.T).abs().max()
        ),
        "unimodular_basis_covariance_max_fp64": float(
            (transformed - reference).abs().max()
        ),
        "projective_vs_full_ball_error_max_fp64": float(
            (full_ball - reference).abs().max()
        ),
        "explicit_pair_reference_error_max_fp64": float(
            (explicit - reference).abs().max()
        ),
        "graphwise_mean_max_fp64": graph_mean_max,
        "cutoff_value_max_fp64": abs(float(cutoff_value)),
        "cutoff_radial_derivative_max_fp64": abs(float(cutoff_derivative)),
        "finite_forward_and_backward": bool(
            torch.isfinite(reference).all()
            and all(torch.isfinite(value).all() for value in gradients)
        ),
        "nonzero_input_gradients": bool(
            all(float(value.abs().max()) > 0.0 for value in gradients[:3])
        ),
        "initial_residual_exact_zero": bool(torch.equal(initial_output, torch.zeros_like(initial_output))),
        "production_python_loop_free": loop_free,
        "operator_parameter_count": sum(value.numel() for value in head.parameters()),
    }
    checks = {
        key: float(metrics[key]) <= float(limit)
        for key, limit in acceptance.items()
        if key in metrics and isinstance(limit, (float, int))
    }
    checks.update(
        {
            "finite_forward_and_backward": bool(metrics["finite_forward_and_backward"]),
            "nonzero_input_gradients": bool(metrics["nonzero_input_gradients"]),
            "initial_residual_exact_zero": bool(metrics["initial_residual_exact_zero"]),
            "production_python_loop_free": bool(metrics["production_python_loop_free"]),
        }
    )
    return metrics, checks


def _precision_audit(
    operator: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    nodes, coordinates, lattice, batch = _numeric_input(12)
    reference_head = _active_head(operator, hidden_dim=12, dtype=torch.float64)
    reference = reference_head(nodes, coordinates, lattice, batch)
    fp32_head = _active_head(operator, hidden_dim=12, dtype=torch.float32).to(device)
    fp32_head.load_state_dict(reference_head.state_dict(), strict=True)
    fp32_input = (
        nodes.float().to(device),
        coordinates.float().to(device),
        lattice.float().to(device),
        batch.to(device),
    )
    with torch.no_grad():
        fp32 = fp32_head(*fp32_input)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bf16 = fp32_head(*fp32_input)
    return {
        "fp32_vs_fp64_relative_error": _relative_error(
            fp32.double().cpu(), reference
        ),
        "bf16_vs_fp32_relative_error": _relative_error(bf16.float(), fp32.float()),
    }


def _cuda_training_benchmark(
    protocol: dict[str, Any],
    cache_root: Path,
    lattice_standardization: Path,
    device: torch.device,
) -> dict[str, float | int]:
    model_spec = protocol["benchmark_model"]
    operator = protocol["operator"]
    dataset = PackedAlexP1Dataset(cache_root, "train")
    batch_size = int(model_spec["batch_size"])
    packed = collate_packed_alex([dataset[index] for index in range(batch_size)]).to(
        device
    )
    graphs = int(packed.num_graphs)
    counts = torch.bincount(packed.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    standardizer = P1LatticeStandardizer.from_json(lattice_standardization)
    model = HybridCrystalDenoiser(
        hidden_dim=int(model_spec["hidden_dim"]),
        vector_dim=int(model_spec["vector_dim"]),
        layers=int(model_spec["layers"]),
        radial_dim=int(model_spec["radial_dim"]),
        radial_cutoff=float(model_spec["radial_cutoff_angstrom"]),
        reciprocal_pair_width=int(operator["pair_width"]),
        reciprocal_channels=int(operator["channels"]),
        reciprocal_radial_dim=int(operator["reciprocal_radial_dim"]),
        reciprocal_cutoff=float(operator["reciprocal_cutoff_inverse_angstrom"]),
    ).to(device)
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(model_spec["coordinate_sigma_min"]),
        coordinate_sigma_max=float(model_spec["coordinate_sigma_max"]),
        minimum_time=float(model_spec["minimum_time"]),
        maximum_time=float(model_spec["maximum_time"]),
    )
    trainer = ProductionTrainer(
        diffusion,
        ProductionTrainingConfig(
            learning_rate=float(model_spec["learning_rate"]),
            weight_decay=float(model_spec["weight_decay"]),
            gradient_clip_norm=float(model_spec["gradient_clip_norm"]),
            ema_decay=float(model_spec["ema_decay"]),
            coordinate_sigma_min=float(model_spec["coordinate_sigma_min"]),
            coordinate_sigma_max=float(model_spec["coordinate_sigma_max"]),
            minimum_time=float(model_spec["minimum_time"]),
            maximum_time=float(model_spec["maximum_time"]),
            precision="bf16",
            objective=str(model_spec["objective"]),
        ),
    )
    generator = torch.Generator(device=device).manual_seed(
        int(protocol["qualification"]["seed"]) + 1
    )
    warmup = int(model_spec["warmup_steps"])
    measured = int(model_spec["measured_steps"])
    output = None
    for _ in range(warmup):
        output, _ = trainer.train_step(
            packed.atom_types,
            packed.frac_coords,
            packed.lattice,
            packed.batch,
            blueprint,
            generator=generator,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(measured):
        output, _ = trainer.train_step(
            packed.atom_types,
            packed.frac_coords,
            packed.lattice,
            packed.batch,
            blueprint,
            generator=generator,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if output is None:
        raise RuntimeError("CUDA benchmark did not execute")
    ball = projective_reciprocal_ball(
        packed.lattice, float(operator["reciprocal_cutoff_inverse_angstrom"])
    )
    pairs = complete_unordered_node_pairs(packed.batch, graphs)
    return {
        "graphs": graphs,
        "nodes": int(packed.num_nodes),
        "unordered_pairs": int(pairs.first.numel()),
        "projective_modes_min": int(ball.mask.sum(-1).min()),
        "projective_modes_median": int(ball.mask.sum(-1).float().median()),
        "projective_modes_max": int(ball.mask.sum(-1).max()),
        "warmup_steps": warmup,
        "measured_steps": measured,
        "cuda_training_step_graphs_per_second": graphs * measured / elapsed,
        "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0**2),
        "tensor_candidates": int(
            output.prediction.gauge_atlas.effective_frame_count.sum()
        ),
        "full_model_parameter_count": sum(value.numel() for value in model.parameters()),
    }


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_pairwise_reciprocal_operator_v1":
        raise ValueError("unexpected pairwise reciprocal operator protocol")
    observed_hash = sha256_file(args.cache_root / "manifest.json")
    if observed_hash != protocol["prerequisites"]["cache_manifest_sha256"]:
        raise ValueError("pairwise reciprocal audit cache manifest mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("frozen operator audit requires CUDA")
    torch.manual_seed(int(protocol["qualification"]["seed"]))
    torch.cuda.manual_seed_all(int(protocol["qualification"]["seed"]))

    acceptance = protocol["acceptance"]
    numeric, checks = _numeric_audit(protocol["operator"], acceptance)
    precision = _precision_audit(protocol["operator"], device)
    benchmark = _cuda_training_benchmark(
        protocol, args.cache_root, args.lattice_standardization, device
    )
    checks.update(
        {
            "fp32_vs_fp64_relative_error": precision["fp32_vs_fp64_relative_error"]
            <= float(acceptance["fp32_vs_fp64_relative_error_max"]),
            "bf16_vs_fp32_relative_error": precision["bf16_vs_fp32_relative_error"]
            <= float(acceptance["bf16_vs_fp32_relative_error_max"]),
            "cuda_training_step_graphs_per_second": benchmark[
                "cuda_training_step_graphs_per_second"
            ]
            >= float(acceptance["cuda_training_step_graphs_per_second_min"]),
            "cuda_peak_allocated_mib": benchmark["cuda_peak_allocated_mib"]
            <= float(acceptance["cuda_peak_allocated_mib_max"]),
            "tensor_candidates": benchmark["tensor_candidates"]
            == int(acceptance["tensor_candidates"]),
        }
    )
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "cache_manifest_sha256": observed_hash,
        "numeric": numeric,
        "precision": precision,
        "cuda_benchmark": benchmark,
        "checks": checks,
        "qualified": qualified,
        "decision": (
            "operator_qualified_freeze_one_pass_coordinate_experiment"
            if qualified
            else "operator_failed_do_not_train"
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
