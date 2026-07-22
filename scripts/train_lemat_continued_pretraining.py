"""Run Stage-C LeMat continuation with MatPES and Alex replay under torchrun."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.balanced_rank_sharded_data import BalancedRankShardedStream
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.continued_pretraining import (
    ContinuedPretrainingStreams,
    ContinuedPretrainingWeights,
    StructureReplayBatch,
    accumulate_stream_parallel_pretraining_step,
    collate_structure_records,
    pack_structure_batch,
)
from gaugeflow.production.lemat_index import IndexedLeMatDataset
from gaugeflow.production.matpes_data import MatPESPhysicalBatch, collate_matpes_records
from gaugeflow.production.matpes_index import IndexedMatPESDataset
from gaugeflow.production.physical_checkpointing import (
    load_physical_checkpoint,
    read_physical_checkpoint_metadata,
    save_physical_checkpoint,
)
from gaugeflow.production.physical_pretraining import load_functional_physical_normalizer
from gaugeflow.production.rank_sharded_data import ExactRankShardedStream
from gaugeflow.production.teacher_feature_cache import MatPESTeacherFeatureCache


@dataclass(frozen=True)
class _PreparedRoleBatch:
    """One role-local batch whose pinned transfer has been enqueued.

    This object deliberately never crosses a checkpoint boundary.  The stream
    cursor is therefore always saved before a future batch is selected, which
    makes an interrupted run restart from the same next indices as an
    uninterrupted run.
    """

    indices: torch.Tensor
    batch: StructureReplayBatch | MatPESPhysicalBatch
    ready: torch.cuda.Event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--lemat-index", type=Path, required=True)
    parser.add_argument("--matpes-index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args()


def _runtime_state(
    streams: ContinuedPretrainingStreams,
    lemat_generator: torch.Generator,
    alex_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "streams": streams.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "lemat_generator_state": lemat_generator.get_state(),
        "alex_generator_state": alex_generator.get_state(),
    }


def _restore_runtime(
    state: dict[str, Any],
    streams: ContinuedPretrainingStreams,
    lemat_generator: torch.Generator,
    alex_generator: torch.Generator,
    device: torch.device,
) -> None:
    required = {
        "schema",
        "streams",
        "cpu_rng_state",
        "cuda_rng_state",
        "lemat_generator_state",
        "alex_generator_state",
    }
    if set(state) != required or state.get("schema") != 1:
        raise ValueError("Stage-C rank runtime checkpoint is incomplete")
    stream_state = state["streams"]
    if not isinstance(stream_state, dict):
        raise ValueError("Stage-C stream checkpoint is invalid")
    streams.load_state_dict(stream_state)
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device)
    lemat_generator.set_state(state["lemat_generator_state"].cpu())
    alex_generator.set_state(state["alex_generator_state"].cpu())


def _verify_protocol_implementation(protocol: dict[str, Any]) -> dict[str, str]:
    prerequisites = protocol.get("prerequisites")
    if not isinstance(prerequisites, dict):
        raise ValueError("Stage-C protocol lacks prerequisites")
    paths = {
        "runner_sha256": Path(__file__),
        "continued_pretraining_sha256": Path(
            inspect.getsourcefile(accumulate_stream_parallel_pretraining_step) or ""
        ),
        "continued_checkpointing_sha256": Path(
            inspect.getsourcefile(build_continued_pretraining_objects) or ""
        ),
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Stage-C implementation file is missing: {name}")
        observed[name] = sha256_file(path)
        if observed[name] != str(prerequisites.get(name)):
            raise ValueError(f"Stage-C implementation hash mismatch: {name}")
    return observed


def _prepare_role_batch(
    *,
    role: str,
    streams: ContinuedPretrainingStreams,
    lemat_dataset: IndexedLeMatDataset,
    matpes_dataset: IndexedMatPESDataset,
    alex_dataset: PackedAlexP1Dataset,
    vocabulary: dict[str, int],
    teacher_dim: int,
    device: torch.device,
    transfer_stream: torch.cuda.Stream,
) -> _PreparedRoleBatch:
    """Build and asynchronously transfer exactly one fixed-role batch.

    Selection and CPU collation run while the preceding CUDA work remains
    queued.  The transfer is then issued on a dedicated stream; consumers
    explicitly wait on its event before using the tensors.  No random state or
    model operation is performed here.
    """

    if role == "lemat_structure":
        indices = streams.lemat.next_indices()
        host_batch = collate_structure_records(lemat_dataset.select(indices)).pin_memory()
        with torch.cuda.stream(transfer_stream):
            batch: StructureReplayBatch | MatPESPhysicalBatch = host_batch.to(device)
    elif role == "matpes_physical":
        indices = streams.matpes.next_indices()
        host_batch = collate_matpes_records(
            matpes_dataset.select(indices),
            functional_vocabulary=vocabulary,
            teacher_dim=teacher_dim,
        ).pin_memory()
        with torch.cuda.stream(transfer_stream):
            batch = host_batch.to(device, non_blocking=True)
    elif role == "alex_structure":
        indices = streams.alex.next_indices()
        with torch.cuda.stream(transfer_stream):
            alex_raw = alex_dataset.select_model_batch(indices, device=device)
            batch = pack_structure_batch(
                alex_raw.atom_types,
                alex_raw.fractional_coordinates,
                alex_raw.lattice,
                alex_raw.batch,
            )
    else:
        raise ValueError(f"unknown Stage-C stream-parallel role: {role}")
    ready = torch.cuda.Event()
    ready.record(transfer_stream)
    return _PreparedRoleBatch(indices=indices, batch=batch, ready=ready)


def _consume_prepared_role_batch(
    prepared: _PreparedRoleBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, StructureReplayBatch | MatPESPhysicalBatch]:
    """Make one prefetched transfer visible to the default compute stream."""

    torch.cuda.current_stream(device).wait_event(prepared.ready)
    return prepared.indices, prepared.batch


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "stage_c_lemat_continued_pretraining_v1" or protocol.get(
        "status_before_run"
    ) != "frozen_method_not_run":
        raise ValueError("unexpected or unfrozen Stage-C protocol")
    implementation_hashes = _verify_protocol_implementation(protocol)
    prerequisites = protocol["prerequisites"]
    training = protocol.get("training")
    data = protocol.get("data")
    if not isinstance(training, dict) or not isinstance(data, dict):
        raise ValueError("Stage-C protocol lacks training or data")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    devices = training.get("devices")
    roles = training.get("stream_parallel_roles")
    expected_roles = ["lemat_structure", "matpes_physical", "alex_structure"]
    if (
        not isinstance(devices, list)
        or world_size != len(devices)
        or world_size != 3
        or roles != expected_roles
    ):
        raise ValueError("Stage-C requires the frozen three-role process topology")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    observed = {
        "stage_b_checkpoint_sha256": sha256_file(args.stage_b_checkpoint),
        "lemat_index_manifest_sha256": sha256_file(args.lemat_index / "manifest.json"),
        "lemat_index_sha256": sha256_file(args.lemat_index / "index.pt"),
        "matpes_index_manifest_sha256": sha256_file(args.matpes_index / "manifest.json"),
        "matpes_index_sha256": sha256_file(args.matpes_index / "index.pt"),
        "matpes_normalizer_sha256": sha256_file(args.normalizer),
        "teacher_cache_manifest_sha256": sha256_file(args.teacher_cache / "manifest.json"),
        "alex_cache_manifest_sha256": sha256_file(args.alex_cache / "manifest.json"),
        **implementation_hashes,
    }
    for name, digest in observed.items():
        if digest != str(prerequisites.get(name)):
            raise ValueError(f"Stage-C prerequisite hash mismatch: {name}")

    stage_b_metadata = read_physical_checkpoint_metadata(args.stage_b_checkpoint)
    objects = build_continued_pretraining_objects(
        stage_b_metadata,
        device=device,
        optimizer_owner=True,
    )
    normalizer, vocabulary = load_functional_physical_normalizer(args.normalizer)
    if vocabulary != objects.functional_vocabulary:
        raise ValueError("Stage-C normalizer disagrees with Stage-B vocabulary")
    lemat_dataset = IndexedLeMatDataset(
        args.lemat_index,
        "train",
        verify_hashes=rank == 0,
    )
    matpes_dataset = IndexedMatPESDataset(
        args.matpes_index,
        "train",
        verify_hashes=rank == 0,
        teacher_feature_cache=args.teacher_cache,
    )
    alex_dataset = PackedAlexP1Dataset(
        args.alex_cache,
        "train",
        verify_hashes=rank == 0,
    )
    dist.barrier(device_ids=[local_rank])
    feature_cache = matpes_dataset.teacher_feature_cache
    if not isinstance(feature_cache, MatPESTeacherFeatureCache) or (
        feature_cache.feature_dim != int(stage_b_metadata["teacher_feature_dim"])
    ):
        raise ValueError("Stage-C MatPES teacher cache disagrees with Stage-B")
    source_weight_mapping = training.get("lemat_functional_weights")
    if not isinstance(source_weight_mapping, dict) or set(source_weight_mapping) != set(
        lemat_dataset.functional_names
    ):
        raise ValueError("Stage-C LeMat functional weights are incomplete")
    source_weights = [float(source_weight_mapping[name]) for name in lemat_dataset.functional_names]
    weights = ContinuedPretrainingWeights(
        lemat_structure=float(training["lemat_structure_weight"]),
        matpes_physical=float(training["matpes_physical_weight"]),
        alex_structure=float(training["alex_structure_weight"]),
    )
    weights.validate()
    streams = ContinuedPretrainingStreams(
        BalancedRankShardedStream(
            lemat_dataset.functional_group_index,
            source_weights,
            int(training["global_lemat_batch"]),
            rank=0,
            world_size=1,
            seed=seed + 101,
            block_index=lemat_dataset.sampling_block_index,
        ),
        ExactRankShardedStream(
            len(matpes_dataset),
            int(training["global_matpes_replay_batch"]),
            rank=0,
            world_size=1,
            seed=seed + 211,
            wrap=True,
        ),
        ExactRankShardedStream(
            len(alex_dataset),
            int(training["global_alex_replay_batch"]),
            rank=0,
            world_size=1,
            seed=seed + 307,
            wrap=True,
        ),
    )
    lemat_generator = torch.Generator(device=device).manual_seed(seed + 401)
    alex_generator = torch.Generator(device=device).manual_seed(seed + 503)
    metadata = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "prerequisites": observed,
        "stage_b_metadata": stage_b_metadata,
        "world_size": world_size,
        "seed": seed,
    }

    checkpoint = args.stage_b_checkpoint if args.resume is None else args.resume
    if args.resume is not None and read_physical_checkpoint_metadata(checkpoint) != metadata:
        raise ValueError("Stage-C resume metadata disagrees with the frozen run")
    runtime_states, _ = load_physical_checkpoint(
        checkpoint,
        model=objects.model,
        trainer=objects.trainer,
        map_location=device,
    )
    if args.resume is not None:
        if not isinstance(runtime_states, list) or len(runtime_states) != world_size:
            raise ValueError("Stage-C checkpoint does not contain every rank")
        _restore_runtime(
            runtime_states[rank],
            streams,
            lemat_generator,
            alex_generator,
            device,
        )
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to append a fresh Stage-C run to a nonempty output")

    stage_b_step = int(training["stage_b_completed_step"])
    if args.resume is None and objects.trainer.step != stage_b_step:
        raise ValueError("Stage-B checkpoint step disagrees with Stage-C protocol")
    total_step = stage_b_step + int(training["optimizer_steps"])
    stop = total_step if args.stop_after_step is None else args.stop_after_step
    if not objects.trainer.step <= stop <= total_step:
        raise ValueError("Stage-C operational stop lies outside the frozen run")
    checkpoint_steps = {
        stage_b_step + int(value) for value in training["checkpoint_relative_steps"]
    } | {stop}
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "training_metrics.jsonl"

    def save_checkpoint() -> None:
        local_runtime = _runtime_state(
            streams,
            lemat_generator,
            alex_generator,
            device,
        )
        gathered: list[Any] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_runtime, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            save_physical_checkpoint(
                args.output / f"checkpoint_step_{objects.trainer.step:08d}.pt",
                model=objects.model,
                trainer=objects.trainer,
                rank_runtime_states=gathered,
                metadata=metadata,
            )
        dist.barrier(device_ids=[local_rank])

    if args.resume is None:
        save_checkpoint()
    start = time.perf_counter()
    graphs_since_log = 0
    role = expected_roles[rank]
    transfer_stream = torch.cuda.Stream(device=device)
    prepared = _prepare_role_batch(
        role=role,
        streams=streams,
        lemat_dataset=lemat_dataset,
        matpes_dataset=matpes_dataset,
        alex_dataset=alex_dataset,
        vocabulary=vocabulary,
        teacher_dim=feature_cache.feature_dim,
        device=device,
        transfer_stream=transfer_stream,
    )
    while objects.trainer.step < stop:
        indices, role_batch = _consume_prepared_role_batch(prepared, device=device)
        role_generator = (
            lemat_generator
            if role == "lemat_structure"
            else alex_generator
            if role == "alex_structure"
            else None
        )
        role_loss = accumulate_stream_parallel_pretraining_step(
            objects.trainer,
            role,
            role_batch,
            normalizer,
            weights,
            generator=role_generator,
        )
        next_step = objects.trainer.step + 1
        # A saved checkpoint must describe the next unselected batch.  We only
        # look ahead when the following update is neither a checkpoint nor the
        # requested terminal step, so the normal path overlaps input work with
        # this update's gradient reduction without weakening exact resume.
        if next_step < stop and next_step not in checkpoint_steps:
            prepared = _prepare_role_batch(
                role=role,
                streams=streams,
                lemat_dataset=lemat_dataset,
                matpes_dataset=matpes_dataset,
                alex_dataset=alex_dataset,
                vocabulary=vocabulary,
                teacher_dim=feature_cache.feature_dim,
                device=device,
                transfer_stream=transfer_stream,
            )
        gradient_norm = objects.trainer.finish_replicated_distributed_optimization_step()
        metrics = torch.zeros(6, device=device, dtype=torch.float64)
        metrics[rank] = role_loss
        metrics[3 + rank] = indices.numel()
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        graphs_since_log += int(metrics[3:].sum())
        if objects.trainer.step == stage_b_step + 1 or objects.trainer.step % 100 == 0 or (
            objects.trainer.step in checkpoint_steps
        ):
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if rank == 0:
                record = {
                    "step": objects.trainer.step,
                    "stage_c_step": objects.trainer.step - stage_b_step,
                    "lemat_structure_loss": float(metrics[0]),
                    "matpes_physical_loss": float(metrics[1]),
                    "alex_structure_loss": float(metrics[2]),
                    "gradient_norm": float(gradient_norm),
                    "graphs_per_second": graphs_since_log / elapsed,
                    "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
                }
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            start = time.perf_counter()
            graphs_since_log = 0
        if objects.trainer.step in checkpoint_steps:
            save_checkpoint()
            if objects.trainer.step < stop:
                prepared = _prepare_role_batch(
                    role=role,
                    streams=streams,
                    lemat_dataset=lemat_dataset,
                    matpes_dataset=matpes_dataset,
                    alex_dataset=alex_dataset,
                    vocabulary=vocabulary,
                    teacher_dim=feature_cache.feature_dim,
                    device=device,
                    transfer_stream=transfer_stream,
                )
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
