"""Evaluate the frozen J1 independent-modality-time attribution gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.runtime import load_tensor_free_ema_runtime

CORNER_NAMES = (
    "clean_clean",
    "noisy_element",
    "noisy_lattice",
    "diagonal",
    "interior",
)


def _corner_side_times(
    name: str,
    coordinate_time: torch.Tensor,
    interior_element_time: torch.Tensor,
    interior_lattice_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros_like(coordinate_time)
    if name == "clean_clean":
        return zeros, zeros
    if name == "noisy_element":
        return coordinate_time, zeros
    if name == "noisy_lattice":
        return zeros, coordinate_time
    if name == "diagonal":
        return coordinate_time, coordinate_time
    if name == "interior":
        return interior_element_time, interior_lattice_time
    raise ValueError(f"unknown modality corner: {name}")


def _paired_bootstrap_ratio(
    initial: torch.Tensor,
    final: torch.Tensor,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    if initial.shape != final.shape or initial.ndim != 1 or initial.numel() < 2:
        raise ValueError("paired bootstrap requires matching structure vectors")
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        initial.numel(),
        (replicates, initial.numel()),
        generator=generator,
    )
    ratios = final[draws].mean(-1) / initial[draws].mean(-1).clamp_min(1.0e-12)
    quantiles = torch.quantile(
        ratios.double(), torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64)
    )
    return {
        "q025": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q975": float(quantiles[2]),
    }


@torch.no_grad()
def _corner_graph_losses(
    checkpoint: Path,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    coordinate_time: torch.Tensor,
    interior_element_time: torch.Tensor,
    interior_lattice_time: torch.Tensor,
    *,
    device: torch.device,
    noise_seed: int,
    protocol_name: str,
    protocol_sha256: str,
    batch_size: int = 16,
) -> tuple[dict[str, torch.Tensor], int]:
    runtime = load_tensor_free_ema_runtime(
        checkpoint,
        device,
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
    )
    if not runtime.model.independent_modality_times:
        raise ValueError("J1 checkpoint does not implement independent modality clocks")
    diffusion = TensorFreeHybridDiffusion(
        runtime.model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    use_bf16 = runtime.training_config["precision"] == "bf16" and device.type == "cuda"
    losses: dict[str, list[torch.Tensor]] = {name: [] for name in CORNER_NAMES}
    candidate_count = 0
    for corner in CORNER_NAMES:
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        element_time, lattice_time = _corner_side_times(
            corner,
            coordinate_time,
            interior_element_time,
            interior_lattice_time,
        )
        for start in range(0, indices.numel(), batch_size):
            stop = min(start + batch_size, indices.numel())
            selected = indices[start:stop]
            packed = Batch.from_data_list([dataset[int(index)] for index in selected]).to(device)
            graphs = int(packed.num_graphs)
            counts = torch.bincount(packed.batch, minlength=graphs)
            blueprint = ParentBlueprintBatch.from_node_counts(
                counts,
                dtype=packed.frac_coords.dtype,
                device=device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = diffusion(
                    packed.atom_types,
                    packed.frac_coords,
                    packed.lattice,
                    packed.batch,
                    blueprint.shape_projector,
                    blueprint.fractional_to_cartesian,
                    time=coordinate_time[start:stop],
                    element_time=element_time[start:stop],
                    lattice_time=lattice_time[start:stop],
                    generator=generator,
                )
            losses[corner].append((output.graph_coordinate_loss / 3.0).float().cpu())
            candidate_count += int(output.prediction.gauge_atlas.effective_frame_count.sum())
    return {name: torch.cat(values) for name, values in losses.items()}, candidate_count


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    training = protocol["training"]
    if (
        protocol.get("protocol") != "h1a_j1_independent_modality_times_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
        or training.get("seeds") != [5705]
        or training.get("steps") != 2111
        or training.get("modality_time_mode") != "independent_corner_mixture"
        or training.get("coordinate_clean_side_information") is not False
    ):
        raise ValueError("unexpected or unfrozen J1 protocol")
    prerequisites = protocol["prerequisites"]
    hash_contract = {
        args.cache_root / "manifest.json": prerequisites["cache_manifest_sha256"],
        Path("reports/h1a_j0_side_information_sensitivity_v1/result.json"): prerequisites[
            "qualification_result_sha256"
        ],
        Path("configs/gates/h1a_j0_side_information_sensitivity_v1.json"): prerequisites[
            "j0_protocol_sha256"
        ],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen prerequisite hash mismatch: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    evaluation = protocol["evaluation"]
    dataset = PackedAlexP1Dataset(args.cache_root, "val")
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(evaluation["validation_seed"])),
    )[: int(evaluation["validation_graphs"])]
    graph_count = indices.numel()
    reference = torch.zeros(1, device=device)
    time_diffusion_checkpoint = args.run_root / "checkpoint_step_00000000.pt"
    protocol_hash = canonical_json_hash(protocol)
    time_runtime = load_tensor_free_ema_runtime(
        time_diffusion_checkpoint,
        device,
        protocol_name=protocol["protocol"],
        protocol_sha256=protocol_hash,
    )
    time_diffusion = TensorFreeHybridDiffusion(
        time_runtime.model,
        time_runtime.lattice_standardizer,
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
    )
    time_generator = torch.Generator(device=device).manual_seed(int(evaluation["validation_noise_seed"]) - 1)
    coordinate_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    interior_element_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    interior_lattice_time = time_diffusion.sample_time(graph_count, reference, generator=time_generator)
    del time_runtime, time_diffusion

    losses_by_step: dict[str, dict[str, torch.Tensor]] = {}
    candidates = 0
    for step in (0, int(training["steps"])):
        losses, count = _corner_graph_losses(
            args.run_root / f"checkpoint_step_{step:08d}.pt",
            dataset,
            indices,
            coordinate_time,
            interior_element_time,
            interior_lattice_time,
            device=device,
            noise_seed=int(evaluation["validation_noise_seed"]),
            protocol_name=protocol["protocol"],
            protocol_sha256=protocol_hash,
        )
        losses_by_step[str(step)] = losses
        candidates += count

    corner_results: dict[str, Any] = {}
    for index, name in enumerate(CORNER_NAMES):
        initial = losses_by_step["0"][name]
        final = losses_by_step[str(training["steps"])][name]
        ratio = float(final.mean() / initial.mean())
        corner_results[name] = {
            "initial_coordinate_mse": float(initial.mean()),
            "final_coordinate_mse": float(final.mean()),
            "validation_ratio": ratio,
            "bootstrap_ratio": _paired_bootstrap_ratio(
                initial,
                final,
                seed=int(evaluation["bootstrap_seed"]) + index,
                replicates=int(evaluation["bootstrap_replicates"]),
            ),
        }

    records = [
        json.loads(line)
        for line in (args.run_root / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final_log = records[-1]
    time_gradient_rows = [row.get("modality_time_gradient_norms", {}) for row in records]
    positive_time_gradients = all(
        set(row) == {"coordinate", "element", "lattice", "fusion"}
        and all(math.isfinite(float(value)) and float(value) > 0.0 for value in row.values())
        for row in time_gradient_rows
    )
    acceptance = protocol["acceptance"]
    checks = {
        "clean_clean_retention": corner_results["clean_clean"]["validation_ratio"]
        <= float(acceptance["clean_clean_validation_ratio_max"]),
        "diagonal_improvement": corner_results["diagonal"]["validation_ratio"]
        <= float(acceptance["diagonal_validation_ratio_max"]),
        "modality_time_gradients": positive_time_gradients,
        "finite_corner_losses": _finite_tree(corner_results),
        "tensor_bypass": candidates == int(acceptance["tensor_candidates"]),
        "throughput": float(final_log["graphs_per_second"])
        >= float(acceptance["training_graphs_per_second_min"]),
        "memory": float(final_log["peak_cuda_memory_mib"])
        <= float(acceptance["peak_cuda_memory_mib_max"]),
        "target_leakage_fields": int(acceptance["target_leakage_fields"]) == 0,
    }
    qualified = all(checks.values())
    decision = "pass" if qualified else "fail"
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "seed": 5705,
        "checkpoint_sha256": sha256_file(
            args.run_root / f"checkpoint_step_{int(training['steps']):08d}.pt"
        ),
        "validation_indices_sha256": canonical_json_hash(indices.tolist()),
        "corner_results": corner_results,
        "training": {
            "final_loss": float(final_log["loss"]),
            "graphs_per_second": float(final_log["graphs_per_second"]),
            "peak_cuda_memory_mib": float(final_log["peak_cuda_memory_mib"]),
            "clip_fraction": float(final_log["clip_fraction"]),
            "final_modality_time_gradient_norms": final_log["modality_time_gradient_norms"],
        },
        "checks": checks,
        "qualified": qualified,
        "decision": decision,
        "decision_text": protocol["decision_rule"][decision],
        "historical_h1a_status_changed": False,
        "j2_authorized": qualified,
        "acf_authorized": False,
        "tensor_work_authorized": False,
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
