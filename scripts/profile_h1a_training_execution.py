"""Profile the unchanged H1a production training execution path on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from gaugeflow.file_utils import canonical_json_hash, load_json_object
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset, collate_packed_alex
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import load_production_checkpoint
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--measure-steps", type=int, default=32)
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument(
        "--sync-debug",
        action="store_true",
        help="emit CUDA synchronization warnings with Python call sites",
    )
    parser.add_argument(
        "--reference-unfused-adamw",
        action="store_true",
        help="benchmark-only reconstruction of the archived unfused AdamW step",
    )
    parser.add_argument(
        "--high-matmul-precision",
        action="store_true",
        help="pilot CUDA TF32 execution for FP32 matrix multiplications",
    )
    parser.add_argument(
        "--equivalence-snapshot",
        type=Path,
        help="write the first real-batch update tensors for backend equivalence",
    )
    return parser.parse_args()


def _next_batch(iterator: Iterator[object], loader: DataLoader[object]) -> tuple[object, Iterator[object]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _device_batch(batch_data: object, device: torch.device) -> tuple[object, ParentBlueprintBatch]:
    moved = batch_data.to(device, non_blocking=True)
    graphs = int(moved.num_graphs)
    counts = torch.bincount(moved.batch, minlength=graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=moved.frac_coords.dtype, device=device
    )
    return moved, blueprint


def _train(
    trainer: ProductionTrainer,
    batch_data: object,
    blueprint: ParentBlueprintBatch,
    generator: torch.Generator,
) -> None:
    trainer.train_step(
        batch_data.atom_types,
        batch_data.frac_coords,
        batch_data.lattice,
        batch_data.batch,
        blueprint,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    if min(args.warmup_steps, args.measure_steps, args.profile_steps) < 1:
        raise ValueError("all profiler step counts must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the production training profiler requires CUDA")
    if args.high_matmul_precision:
        torch.set_float32_matmul_precision("high")
    if args.sync_debug:
        torch.cuda.set_sync_debug_mode("warn")
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_fixed_dynamic_coordinate_learning_curve_v1":
        raise ValueError("profiler requires the frozen H1a exposure protocol")
    training_spec = protocol["training"]
    seed = int(training_spec["seeds"][0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    loader_generator = torch.Generator().manual_seed(seed)
    loader: DataLoader[object] = DataLoader(
        dataset,
        batch_size=int(training_spec["batch_size"]),
        shuffle=True,
        num_workers=int(training_spec["num_workers"]),
        collate_fn=collate_packed_alex,
        generator=loader_generator,
        drop_last=False,
        pin_memory=True,
        persistent_workers=int(training_spec["num_workers"]) > 0,
    )

    metadata_protocol_hash = canonical_json_hash(protocol)
    model = HybridCrystalDenoiser(
        hidden_dim=int(protocol["model"]["hidden_dim"]),
        vector_dim=int(protocol["model"]["vector_dim"]),
        layers=int(protocol["model"]["layers"]),
        radial_dim=int(protocol["model"]["radial_dim"]),
        radial_cutoff=float(protocol["model"]["radial_cutoff_angstrom"]),
        atlas_residual_circle_samples=8,
        edge_dim=int(protocol["model"]["edge_dim"]),
        angular_channels=int(protocol["model"]["angular_channels"]),
        edge_refresh_rank=int(protocol["model"]["edge_refresh_rank"]),
    ).to(device)
    standardizer = P1LatticeStandardizer.from_mapping(
        json.loads(
            Path("configs/statistics/h1a_p1_lattice_standardization.json").read_text(
                encoding="utf-8"
            )
        )
    )
    training_config = ProductionTrainingConfig(
        learning_rate=float(training_spec["learning_rate"]),
        weight_decay=float(training_spec["weight_decay"]),
        gradient_clip_norm=float(training_spec["gradient_clip_norm"]),
        ema_decay=float(training_spec["ema_decay"]),
        coordinate_sigma_min=float(training_spec["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training_spec["coordinate_sigma_max"]),
        minimum_time=float(training_spec["minimum_time"]),
        maximum_time=float(training_spec["maximum_time"]),
        precision=str(training_spec["precision"]),
        objective=str(training_spec["objective"]),
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=training_config.coordinate_sigma_min,
        coordinate_sigma_max=training_config.coordinate_sigma_max,
        minimum_time=training_config.minimum_time,
        maximum_time=training_config.maximum_time,
    )
    trainer = ProductionTrainer(diffusion, training_config)
    source_step, _, metadata = load_production_checkpoint(
        args.checkpoint,
        model=model,
        ema=trainer.ema,
        optimizer=trainer.optimizer,
        map_location=device,
    )
    if args.reference_unfused_adamw:
        optimizer_state = trainer.optimizer.state_dict()
        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            fused=False,
        )
        trainer.optimizer.load_state_dict(optimizer_state)
        for group in trainer.optimizer.param_groups:
            group["fused"] = False
            group["foreach"] = False
        torch.set_float32_matmul_precision("highest")
    if (
        metadata.get("protocol") != protocol["protocol"]
        or metadata.get("protocol_sha256") != metadata_protocol_hash
    ):
        raise ValueError("profile checkpoint does not match the frozen protocol")

    iterator = iter(loader)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    if args.equivalence_snapshot is not None:
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        host_batch, iterator = _next_batch(iterator, loader)
        device_batch, blueprint = _device_batch(host_batch, device)
        output, gradient_norm = trainer.train_step(
            device_batch.atom_types,
            device_batch.frac_coords,
            device_batch.lattice,
            device_batch.batch,
            blueprint,
            generator=generator,
        )
        optimizer_state = trainer.optimizer.state_dict()
        args.equivalence_snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "selected_loss": output.coordinate_loss.detach().cpu(),
                "coordinate_prediction": output.prediction.coordinate_cartesian_scaled_score.detach().cpu(),
                "gradient_norm": gradient_norm.cpu(),
                "gradients": {
                    name: parameter.grad.detach().cpu()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                },
                "parameters": {
                    name: parameter.detach().cpu()
                    for name, parameter in model.named_parameters()
                },
                "initial_parameters": initial_parameters,
                "optimizer": optimizer_state,
                "matmul_precision": torch.get_float32_matmul_precision(),
                "fused": [group.get("fused") for group in trainer.optimizer.param_groups],
            },
            args.equivalence_snapshot,
        )
    for _ in range(args.warmup_steps):
        host_batch, iterator = _next_batch(iterator, loader)
        device_batch, blueprint = _device_batch(host_batch, device)
        _train(trainer, device_batch, blueprint, generator)
    torch.cuda.synchronize(device)

    data_seconds = 0.0
    transfer_seconds = 0.0
    train_seconds = 0.0
    measured_graphs = 0
    peak_before = torch.cuda.max_memory_allocated(device)
    for _ in range(args.measure_steps):
        start = time.perf_counter()
        host_batch, iterator = _next_batch(iterator, loader)
        data_seconds += time.perf_counter() - start

        start = time.perf_counter()
        device_batch, blueprint = _device_batch(host_batch, device)
        torch.cuda.synchronize(device)
        transfer_seconds += time.perf_counter() - start

        start = time.perf_counter()
        _train(trainer, device_batch, blueprint, generator)
        torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - start
        measured_graphs += int(device_batch.num_graphs)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as execution_profile:
        for _ in range(args.profile_steps):
            host_batch, iterator = _next_batch(iterator, loader)
            device_batch, blueprint = _device_batch(host_batch, device)
            _train(trainer, device_batch, blueprint, generator)
    torch.cuda.synchronize(device)

    total_seconds = data_seconds + transfer_seconds + train_seconds
    result = {
        "protocol": "h1a_training_execution_profile_v1",
        "source_protocol": protocol["protocol"],
        "source_protocol_sha256": metadata_protocol_hash,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": source_step,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "optimizer_fused": [group.get("fused") for group in trainer.optimizer.param_groups],
        "batch_size": int(training_spec["batch_size"]),
        "num_workers": int(training_spec["num_workers"]),
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "profile_steps": args.profile_steps,
        "graphs": measured_graphs,
        "seconds": {
            "data_wait": data_seconds,
            "host_to_device_and_blueprint": transfer_seconds,
            "train_step": train_seconds,
            "total_synchronized": total_seconds,
        },
        "fractions": {
            "data_wait": data_seconds / total_seconds,
            "host_to_device_and_blueprint": transfer_seconds / total_seconds,
            "train_step": train_seconds / total_seconds,
        },
        "graphs_per_second_synchronized": measured_graphs / total_seconds,
        "peak_cuda_memory_mib": max(
            peak_before, torch.cuda.max_memory_allocated(device)
        )
        / (1024.0**2),
        "scientific_state_changed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "torch_profiler.txt").write_text(
        execution_profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40
        )
        + "\n",
        encoding="utf-8",
    )
    execution_profile.export_stacks(
        str(args.output / "torch_profiler_cpu_stacks.txt"),
        metric="self_cpu_time_total",
    )
    execution_profile.export_chrome_trace(str(args.output / "torch_profiler_trace.json"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
