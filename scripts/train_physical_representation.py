"""Train frozen Stage-B physical representation transfer under torchrun."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.matpes_data import collate_matpes_records
from gaugeflow.production.matpes_index import IndexedMatPESDataset
from gaugeflow.production.physical_checkpointing import (
    load_physical_checkpoint,
    read_physical_checkpoint_metadata,
    save_physical_checkpoint,
)
from gaugeflow.production.physical_pretraining import (
    PhysicalRepresentationModel,
    load_functional_physical_normalizer,
)
from gaugeflow.production.physical_training import (
    PhysicalTransferTrainer,
    PhysicalTransferTrainingConfig,
)
from gaugeflow.production.rank_sharded_data import ExactRankShardedStream
from gaugeflow.production.teacher_feature_cache import MatPESTeacherFeatureCache
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--matpes-index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args()


def _runtime_state(
    physical_stream: ExactRankShardedStream,
    replay_stream: ExactRankShardedStream,
    device_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "physical_stream": physical_stream.state_dict(),
        "replay_stream": replay_stream.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "device_generator_state": device_generator.get_state(),
    }


def _restore_runtime(
    state: dict[str, Any],
    physical_stream: ExactRankShardedStream,
    replay_stream: ExactRankShardedStream,
    device_generator: torch.Generator,
    device: torch.device,
) -> None:
    required = {
        "physical_stream",
        "replay_stream",
        "cpu_rng_state",
        "cuda_rng_state",
        "device_generator_state",
    }
    if not required.issubset(state):
        raise ValueError("Stage-B rank runtime checkpoint is incomplete")
    physical_stream.load_state_dict(state["physical_stream"])
    replay_stream.load_state_dict(state["replay_stream"])
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device)
    device_generator.set_state(state["device_generator_state"].cpu())


def _all_reduce_metrics(values: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "stage_b_physical_representation_v1" or protocol.get(
        "status_before_run"
    ) != "frozen_method_not_run":
        raise ValueError("unexpected or unfrozen Stage-B protocol")
    prerequisites = protocol.get("prerequisites")
    training = protocol.get("training")
    data = protocol.get("data")
    if not all(isinstance(value, dict) for value in (prerequisites, training, data)):
        raise ValueError("Stage-B protocol lacks prerequisites, data, or training")
    assert isinstance(prerequisites, dict) and isinstance(training, dict) and isinstance(data, dict)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != len(training["devices"]):
        raise ValueError("torchrun world size disagrees with the frozen Stage-B device count")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    observed = {
        "a1_checkpoint_sha256": sha256_file(args.a1_checkpoint),
        "matpes_index_manifest_sha256": sha256_file(args.matpes_index / "manifest.json"),
        "matpes_index_sha256": sha256_file(args.matpes_index / "index.pt"),
        "matpes_normalizer_sha256": sha256_file(args.normalizer),
    }
    for name, digest in observed.items():
        if digest != str(prerequisites[name]):
            raise ValueError(f"Stage-B prerequisite hash mismatch: {name}")
    protocol_sha256 = canonical_json_hash(protocol)

    a1_metadata = read_production_checkpoint_metadata(args.a1_checkpoint)
    model_config = a1_metadata.get("model_config")
    a1_training = a1_metadata.get("training_config")
    standardization = a1_metadata.get("lattice_standardization")
    if not all(isinstance(value, dict) for value in (model_config, a1_training, standardization)):
        raise ValueError("A1 checkpoint metadata is incomplete")
    assert isinstance(model_config, dict) and isinstance(a1_training, dict)
    assert isinstance(standardization, dict)
    backbone = HybridCrystalDenoiser(**model_config).to(device)
    if args.resume is None:
        a1_ema = ExponentialMovingAverage(backbone, float(a1_training["ema_decay"]))
        load_production_checkpoint(args.a1_checkpoint, model=backbone, ema=a1_ema, map_location=device)
        a1_ema.copy_to(backbone)
        del a1_ema

    normalizer, vocabulary = load_functional_physical_normalizer(args.normalizer)
    expected_vocabulary = {str(key): int(value) for key, value in data["functional_vocabulary"].items()}
    if vocabulary != expected_vocabulary:
        raise ValueError("normalizer functional vocabulary disagrees with Stage-B protocol")
    physical_dataset = IndexedMatPESDataset(
        args.matpes_index,
        "train",
        verify_hashes=rank == 0,
        teacher_feature_cache=args.teacher_cache,
    )
    if physical_dataset.teacher_feature_cache is None:
        raise AssertionError("Stage-B physical dataset did not attach its teacher cache")
    feature_cache: MatPESTeacherFeatureCache = physical_dataset.teacher_feature_cache
    replay_dataset = PackedAlexP1Dataset(
        args.alex_cache,
        "train",
        verify_hashes=rank == 0,
    )
    dist.barrier()
    if len(physical_dataset) != int(training["matpes_rows_without_replacement"]):
        raise ValueError("MatPES training size disagrees with frozen one-pass exposure")

    model = PhysicalRepresentationModel(
        backbone,
        teacher_dim=feature_cache.feature_dim,
        functional_count=len(vocabulary),
    ).to(device)
    diffusion = TensorFreeHybridDiffusion(
        backbone,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(a1_training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(a1_training["coordinate_sigma_max"]),
        minimum_time=float(a1_training["minimum_time"]),
        maximum_time=float(a1_training["maximum_time"]),
        categorical_path=str(a1_training["categorical_path"]),
        composition_conditioning=bool(a1_training["composition_conditioning"]),
    )
    config = PhysicalTransferTrainingConfig(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        ema_decay=float(training["ema_decay"]),
        precision="bf16",
        energy_weight=float(training["energy_weight"]),
        force_weight=float(training["force_weight"]),
        stress_weight=float(training["stress_weight"]),
        feature_weight=float(training["feature_weight"]),
    )
    trainer = PhysicalTransferTrainer(model, diffusion, config, optimizer_owner=rank == 0)
    physical_stream = ExactRankShardedStream(
        len(physical_dataset),
        int(training["global_matpes_batch_except_tail"]),
        rank=rank,
        world_size=world_size,
        seed=seed + 101,
        wrap=False,
    )
    replay_stream = ExactRankShardedStream(
        len(replay_dataset),
        int(training["global_alex_replay_batch"]),
        rank=rank,
        world_size=world_size,
        seed=seed + 211,
        wrap=True,
    )
    device_generator = torch.Generator(device=device).manual_seed(seed + 307 + rank)

    metadata = {
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_sha256,
        "a1_checkpoint_sha256": observed["a1_checkpoint_sha256"],
        "matpes_index_manifest_sha256": observed["matpes_index_manifest_sha256"],
        "matpes_normalizer_sha256": observed["matpes_normalizer_sha256"],
        "teacher_cache_manifest_sha256": sha256_file(args.teacher_cache / "manifest.json"),
        "model_config": model_config,
        "a1_training_config": a1_training,
        "lattice_standardization": standardization,
        "physical_training_config": dataclasses.asdict(config),
        "functional_vocabulary": vocabulary,
        "teacher_feature_dim": feature_cache.feature_dim,
        "world_size": world_size,
        "seed": seed,
    }
    if args.resume is not None:
        resume_metadata = read_physical_checkpoint_metadata(args.resume)
        if resume_metadata != metadata:
            raise ValueError("physical resume metadata disagrees with current frozen run")
        owner_runtime: list[dict[str, Any]] | None = None
        if rank == 0:
            owner_runtime, _ = load_physical_checkpoint(
                args.resume,
                model=model,
                trainer=trainer,
                map_location=device,
            )
        trainer.broadcast_distributed_state(owner_rank=0)
        payload: list[Any] = [owner_runtime]
        dist.broadcast_object_list(payload, src=0)
        runtime_states = payload[0]
        if not isinstance(runtime_states, list) or len(runtime_states) != world_size:
            raise ValueError("physical checkpoint does not contain every rank")
        _restore_runtime(runtime_states[rank], physical_stream, replay_stream, device_generator, device)
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to append a fresh Stage-B run to a nonempty output")

    args.output.mkdir(parents=True, exist_ok=True)
    total_steps = int(training["optimizer_steps"])
    stop = total_steps if args.stop_after_step is None else args.stop_after_step
    if not trainer.step <= stop <= total_steps:
        raise ValueError("Stage-B operational stop lies outside the frozen run")
    checkpoints = {int(value) for value in training["checkpoint_steps"]} | {stop}
    log_path = args.output / "training_metrics.jsonl"

    def save_checkpoint() -> None:
        local_runtime = _runtime_state(physical_stream, replay_stream, device_generator, device)
        gathered: list[Any] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_runtime, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            save_physical_checkpoint(
                args.output / f"checkpoint_step_{trainer.step:08d}.pt",
                model=model,
                trainer=trainer,
                rank_runtime_states=gathered,
                metadata=metadata,
            )
        dist.barrier()

    if args.resume is None:
        save_checkpoint()
    start = time.perf_counter()
    graphs_since_log = 0
    while trainer.step < stop:
        physical_indices = physical_stream.next_indices()
        replay_indices = replay_stream.next_indices()
        records = [physical_dataset[int(index)] for index in physical_indices]
        physical_batch = collate_matpes_records(
            records,
            functional_vocabulary=vocabulary,
            teacher_dim=feature_cache.feature_dim,
        ).to(device)
        replay = replay_dataset.select_model_batch(replay_indices, device=device)
        replay_graphs = replay_indices.numel()
        replay_counts = torch.bincount(replay.batch, minlength=replay_graphs)
        blueprint = ParentBlueprintBatch.from_node_counts(
            replay_counts,
            dtype=replay.fractional_coordinates.dtype,
            device=device,
        )
        physical_denominators = trainer.distributed_physical_denominators(physical_batch)
        replay_fraction = trainer.distributed_local_fraction(replay_graphs, device=device)
        trainer.begin_optimization_step()
        physical = trainer.accumulate_physical_step(
            physical_batch,
            normalizer,
            loss_weight=float(training["physical_loss_weight"]),
            denominators=physical_denominators,
        )
        replay_loss = trainer.accumulate_alex_replay_step(
            replay.atom_types,
            replay.fractional_coordinates,
            replay.lattice,
            replay.batch,
            blueprint,
            loss_weight=float(training["alex_replay_loss_weight"]) * replay_fraction,
            generator=device_generator,
        )
        gradient_norm = trainer.finish_distributed_optimization_step(owner_rank=0)
        metrics = torch.tensor(
            [
                float(physical.loss.detach()),
                float(physical.energy_loss.detach()),
                float(physical.force_loss.detach()),
                float(physical.stress_loss.detach()),
                float(physical.feature_loss.detach()),
                float(replay_loss) * replay_fraction,
                float(physical_indices.numel()),
                float(replay_graphs),
            ],
            device=device,
            dtype=torch.float64,
        )
        metrics = _all_reduce_metrics(metrics)
        graphs_since_log += int(metrics[6] + metrics[7])
        if trainer.step == 1 or trainer.step % 100 == 0 or trainer.step in checkpoints:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if rank == 0:
                record = {
                    "step": trainer.step,
                    "physical_loss": float(metrics[0]),
                    "energy_loss": float(metrics[1]),
                    "force_loss": float(metrics[2]),
                    "stress_loss": float(metrics[3]),
                    "feature_loss": float(metrics[4]),
                    "alex_replay_loss": float(metrics[5]),
                    "gradient_norm": float(gradient_norm),
                    "graphs_per_second": graphs_since_log / elapsed,
                    "matpes_examples": physical_stream.state.global_examples_emitted,
                    "alex_replay_examples": replay_stream.state.global_examples_emitted,
                    "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
                }
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            start = time.perf_counter()
            graphs_since_log = 0
        if trainer.step in checkpoints:
            save_checkpoint()

    if stop == total_steps and (
        not physical_stream.exhausted
        or physical_stream.state.global_examples_emitted != int(training["matpes_rows_without_replacement"])
    ):
        raise RuntimeError("Stage-B completed without one exact MatPES pass")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
