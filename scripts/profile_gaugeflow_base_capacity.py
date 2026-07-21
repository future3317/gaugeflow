"""Profile one frozen GaugeFlow-base capacity candidate on real P1 batches."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def _model_config(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_dim": int(specification["hidden_dim"]),
        "vector_dim": int(specification["vector_dim"]),
        "layers": int(specification["layers"]),
        "radial_dim": int(specification["radial_dim"]),
        "radial_cutoff": float(specification["radial_cutoff_angstrom"]),
        "atlas_residual_circle_samples": 8,
        "edge_dim": int(specification["edge_dim"]),
        "angular_channels": int(specification["angular_channels"]),
        "edge_refresh_rank": int(specification["edge_refresh_rank"]),
        "modality_time_conditioning": str(specification["modality_time_conditioning"]),
    }


def _next_microbatches(iterator: Any, count: int) -> list[Any]:
    values = []
    for _ in range(count):
        values.append(next(iterator))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "gaugeflow_base_capacity_execution_smoke_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen capacity smoke protocol")
    if args.candidate not in protocol["candidates"]:
        raise ValueError("unknown capacity candidate")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["source"]["cache_manifest_sha256"]
    ):
        raise ValueError("capacity smoke cache identity changed")
    generated_side = load_json_object(
        Path("reports/h1a_generated_side_coordinate_exposure_v1/result.json")
    )
    if canonical_json_hash(generated_side) != str(
        protocol["source"]["generated_side_result_canonical_sha256"]
    ) or generated_side.get("qualified") is not True:
        raise ValueError("generated-side prerequisite identity changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capacity smoke requires CUDA")
    shared = protocol["shared_training"]
    candidate = protocol["candidates"][args.candidate]
    seed = int(shared["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = HybridCrystalDenoiser(**_model_config(candidate)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(candidate["parameter_count"]):
        raise ValueError("capacity candidate parameter count changed")
    standardizer = P1LatticeStandardizer.from_mapping(
        load_json_object(Path("configs/statistics/h1a_p1_lattice_standardization.json"))
    )
    config = ProductionTrainingConfig(
        learning_rate=float(shared["learning_rate"]),
        weight_decay=float(shared["weight_decay"]),
        gradient_clip_norm=float(shared["gradient_clip_norm"]),
        ema_decay=float(shared["ema_decay"]),
        coordinate_sigma_min=float(shared["coordinate_sigma_min"]),
        coordinate_sigma_max=float(shared["coordinate_sigma_max"]),
        minimum_time=float(shared["minimum_time"]),
        maximum_time=float(shared["maximum_time"]),
        precision=str(shared["precision"]),
        objective="coordinate",
        coordinate_clean_side_information=True,
    )
    trainer = ProductionTrainer(
        TensorFreeHybridDiffusion(
            model,
            standardizer,
            coordinate_sigma_min=config.coordinate_sigma_min,
            coordinate_sigma_max=config.coordinate_sigma_max,
            minimum_time=config.minimum_time,
            maximum_time=config.maximum_time,
        ),
        config,
    )
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(candidate["physical_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_packed_alex,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    iterator = iter(loader)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    accumulation = int(candidate["gradient_accumulation_steps"])

    def optimizer_step() -> tuple[float, float, int]:
        microbatches = _next_microbatches(iterator, accumulation)
        graphs = sum(int(value.num_graphs) for value in microbatches)
        trainer.begin_optimization_step()
        loss = 0.0
        for host in microbatches:
            moved = host.to(device, non_blocking=True)
            count = int(moved.num_graphs)
            node_counts = torch.bincount(moved.batch, minlength=count)
            blueprint = ParentBlueprintBatch.from_node_counts(
                node_counts,
                dtype=moved.frac_coords.dtype,
                device=device,
            )
            weight = count / graphs
            output = trainer.accumulate_hybrid_step(
                moved.atom_types,
                moved.frac_coords,
                moved.lattice,
                moved.batch,
                blueprint,
                loss_weight=weight,
                generator=generator,
            )
            loss += weight * float(output.coordinate_loss.detach().cpu())
        gradient = float(trainer.finish_optimization_step().cpu())
        return loss, gradient, graphs

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(int(shared["warmup_optimizer_steps"])):
        optimizer_step()
    torch.cuda.synchronize(device)
    losses: list[float] = []
    gradients: list[float] = []
    graphs = 0
    start = time.perf_counter()
    for _ in range(int(shared["measured_optimizer_steps"])):
        loss, gradient, count = optimizer_step()
        losses.append(loss)
        gradients.append(gradient)
        graphs += count
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    metrics = {
        "candidate": args.candidate,
        "parameter_count": parameter_count,
        "physical_batch_size": int(candidate["physical_batch_size"]),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": int(candidate["physical_batch_size"]) * accumulation,
        "graphs_per_second": graphs / elapsed,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "losses": losses,
        "gradient_norms": gradients,
        "hardware": torch.cuda.get_device_name(device),
    }
    acceptance = protocol["acceptance"]
    checks = {
        "finite_loss": all(math.isfinite(value) for value in losses),
        "finite_gradient": all(math.isfinite(value) for value in gradients),
        "exact_effective_batch": metrics["effective_batch_size"]
        == int(acceptance["exact_effective_batch"]),
        "memory": metrics["peak_cuda_memory_mib"]
        <= float(acceptance["peak_cuda_memory_mib_max"]),
        "throughput": metrics["graphs_per_second"]
        >= float(candidate["graphs_per_second_min"]),
    }
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "metrics": metrics,
        "checks": checks,
        "qualified": all(checks.values()),
        "decision": protocol["decision_rule"]["pass" if all(checks.values()) else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
