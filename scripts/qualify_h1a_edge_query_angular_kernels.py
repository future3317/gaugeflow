"""Matched CUDA qualification of explicit-triplet and induced-slot kernels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from qualify_h1a_factorized_cartesian_angular_moments import (
    _benchmark,
    _cosine,
    _precision_probe,
)

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.edge_query_angular_kernel import (
    InducedEdgeQueryAngularKernel,
    ShellCompleteTopKTripletKernel,
)
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.matched_initialization import matched_angular_model


def _model_config(shared: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_dim": int(shared["hidden_dim"]),
        "vector_dim": int(shared["vector_dim"]),
        "layers": int(shared["layers"]),
        "radial_dim": int(shared["radial_dim"]),
        "radial_cutoff": float(shared["radial_cutoff_angstrom"]),
        "atlas_residual_circle_samples": 8,
        "edge_dim": int(shared["edge_dim"]),
        "angular_channels": int(shared["angular_channels"]),
        "edge_refresh_rank": int(shared["edge_refresh_rank"]),
        "angular_operator": str(variant["angular_operator"]),
        "angular_slots": int(variant.get("angular_slots", 8)),
        "triplet_k": int(variant.get("triplet_k", 8)),
    }


def main() -> None:
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
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol")
        != "h1a_edge_query_angular_kernel_comparison_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen edge-query qualification protocol")
    prerequisites = protocol["prerequisites"]
    if sha256_file(args.cache_root / "manifest.json") != str(
        prerequisites["cache_manifest_sha256"]
    ):
        raise ValueError("edge-query qualification cache mismatch")
    retained = Path("reports") / str(prerequisites["retained_backbone"]) / "result.json"
    if sha256_file(retained) != str(prerequisites["retained_result_sha256"]):
        raise ValueError("retained dynamic-edge result hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal edge-query qualification requires CUDA")

    data = protocol["data"]
    dataset = PackedAlexP1Dataset(args.cache_root, str(data["split"]))
    graphs = int(data["fixed_graphs"])
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(data["fixed_selection_seed"])),
    )[:graphs]
    batch_data = collate_packed_alex([dataset[int(index)] for index in indices]).to(device)
    counts = torch.bincount(batch_data.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=batch_data.frac_coords.dtype, device=device
    )
    clean = (
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
    )
    time_values = torch.tensor(data["times"], dtype=torch.float32, device=device)
    times = time_values.repeat((graphs + time_values.numel() - 1) // time_values.numel())[
        :graphs
    ]
    standardizer = P1LatticeStandardizer.from_json(args.lattice_standardization)
    benchmark = protocol["benchmark"]
    acceptance = protocol["acceptance"]
    results: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    for name, variant in protocol["variants"].items():
        config = _model_config(protocol["shared_model"], variant)
        model, shared_parameters = matched_angular_model(
            config, seed=int(data["model_seed"])
        )
        model = model.to(device)
        if sum(parameter.numel() for parameter in model.parameters()) != int(
            variant["parameter_count"]
        ):
            raise ValueError(f"{name} parameter count differs from protocol")
        initial_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        captured: dict[str, torch.Tensor] = {}

        def capture_input(
            module: torch.nn.Module, inputs: tuple[Any, ...]
        ) -> None:
            if isinstance(module, InducedEdgeQueryAngularKernel):
                captured["probability"] = module.assignment_probabilities(
                    inputs[0]
                ).detach()
            elif isinstance(module, ShellCompleteTopKTripletKernel):
                captured["selected_count"] = inputs[4].selected_count.detach()

        handle = model.blocks[0].angular_moments.register_forward_pre_hook(capture_input)
        diffusion = TensorFreeHybridDiffusion(model, standardizer)
        fp32_output, fp32_gradient, fp32_loss, fp32_candidates = _precision_probe(
            diffusion,
            clean,
            blueprint,
            times,
            seed=int(data["noise_seed"]),
            bf16=False,
        )
        angular_gradients = [
            parameter.grad
            for parameter_name, parameter in model.named_parameters()
            if ".angular_moments." in parameter_name
        ]
        angular_gradient_norm = math.sqrt(
            sum(
                float(gradient.detach().float().square().sum().cpu())
                for gradient in angular_gradients
                if gradient is not None
            )
        )
        assignment_context_gradients = [
            parameter.grad
            for parameter_name, parameter in model.named_parameters()
            if ".induced_assignment_refresh." in parameter_name
        ]
        assignment_context_gradient_norm = math.sqrt(
            sum(
                float(gradient.detach().float().square().sum().cpu())
                for gradient in assignment_context_gradients
                if gradient is not None
            )
        )
        bf16_output, bf16_gradient, bf16_loss, bf16_candidates = _precision_probe(
            diffusion,
            clean,
            blueprint,
            times,
            seed=int(data["noise_seed"]),
            bf16=True,
        )
        graphs_per_second, peak_mib = _benchmark(
            diffusion,
            clean,
            blueprint,
            times,
            seed=int(data["noise_seed"]),
            warmup=int(benchmark["warmup"]),
            iterations=int(benchmark["iterations"]),
        )
        handle.remove()
        output_delta = bf16_output - fp32_output
        metrics: dict[str, Any] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "shared_parameter_values": shared_parameters,
            "fp32_loss": fp32_loss,
            "bf16_loss": bf16_loss,
            "angular_gradient_norm": angular_gradient_norm,
            "assignment_context_gradient_norm": assignment_context_gradient_norm,
            "bf16_fp32_output_relative_rmse": float(
                (
                    output_delta.square().mean().sqrt()
                    / fp32_output.square().mean().sqrt().clamp_min(1.0e-30)
                ).cpu()
            ),
            "bf16_fp32_output_cosine": _cosine(
                fp32_output.reshape(-1), bf16_output.reshape(-1)
            ),
            "bf16_fp32_gradient_cosine": _cosine(fp32_gradient, bf16_gradient),
            "forward_graphs_per_second": graphs_per_second,
            "peak_cuda_memory_mib": peak_mib,
            "tensor_candidates": max(fp32_candidates, bf16_candidates),
            "optimizer_steps": 0,
            "parameters_unchanged": all(
                torch.equal(initial_state[key], value.detach().cpu())
                for key, value in model.state_dict().items()
            ),
        }
        if "probability" in captured:
            probability = captured["probability"].float()
            entropy = -(
                probability * probability.clamp_min(1.0e-30).log()
            ).sum(dim=-1)
            global_mass = probability.sum(dim=0) / probability.shape[0]
            metrics.update(
                {
                    "normalized_assignment_entropy": float(
                        (entropy.mean() / math.log(probability.shape[1])).cpu()
                    ),
                    "effective_slot_count": float(entropy.exp().mean().cpu()),
                    "maximum_global_slot_mass": float(global_mass.max().cpu()),
                }
            )
        if "selected_count" in captured:
            selected_count = captured["selected_count"]
            positive = selected_count[selected_count > 0]
            metrics.update(
                {
                    "minimum_selected_neighbors": int(positive.min()),
                    "mean_selected_neighbors": float(positive.float().mean()),
                    "maximum_selected_neighbors": int(positive.max()),
                    "boundary_shell_expansion_fraction": float(
                        (positive > int(variant["triplet_k"])).float().mean()
                    ),
                }
            )
        variant_checks = {
            "finite": all(
                torch.isfinite(value).all()
                for value in (fp32_output, bf16_output, fp32_gradient, bf16_gradient)
            ),
            "angular_gradient": angular_gradient_norm
            >= float(acceptance["first_step_angular_gradient_norm_min"]),
            "output_relative_rmse": metrics["bf16_fp32_output_relative_rmse"]
            <= float(acceptance["bf16_fp32_output_relative_rmse_max"]),
            "output_cosine": metrics["bf16_fp32_output_cosine"]
            >= float(acceptance["bf16_fp32_output_cosine_min"]),
            "gradient_cosine": metrics["bf16_fp32_gradient_cosine"]
            >= float(acceptance["bf16_fp32_gradient_cosine_min"]),
            "throughput": graphs_per_second
            >= float(acceptance[f"{name}_graphs_per_second_min"]),
            "memory": peak_mib <= float(acceptance["peak_cuda_memory_mib_max"]),
            "atlas_bypass": metrics["tensor_candidates"]
            == int(acceptance["tensor_candidates"]),
            "parameters_unchanged": bool(metrics["parameters_unchanged"]),
        }
        if "probability" in captured:
            variant_checks.update(
                {
                    "assignment_context_gradient": metrics[
                        "assignment_context_gradient_norm"
                    ]
                    >= float(
                        acceptance["first_step_assignment_context_gradient_norm_min"]
                    ),
                    "assignment_entropy": metrics["normalized_assignment_entropy"]
                    >= float(acceptance["normalized_assignment_entropy_min"]),
                    "effective_slots": metrics["effective_slot_count"]
                    >= float(acceptance["effective_slot_count_min"]),
                    "slot_mass": metrics["maximum_global_slot_mass"]
                    <= float(acceptance["maximum_global_slot_mass_max"]),
                }
            )
        if "selected_count" in captured:
            variant_checks["minimum_neighbors"] = metrics[
                "minimum_selected_neighbors"
            ] >= int(acceptance["minimum_selected_neighbors"])
        results[name] = {"metrics": metrics, "checks": variant_checks}
        checks[name] = all(variant_checks.values())
        del diffusion, model
        torch.cuda.empty_cache()

    result = {
        "protocol": protocol["protocol"],
        "qualified_variants": [name for name, passed in checks.items() if passed],
        "variant_qualified": checks,
        "results": results,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not any(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
