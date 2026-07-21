"""Run a bounded two-rank Stage-B synchronization smoke under torchrun."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from gaugeflow.file_utils import sha256_file
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
from gaugeflow.production.physical_pretraining import (
    PhysicalRepresentationModel,
    load_functional_physical_normalizer,
)
from gaugeflow.production.physical_training import (
    PhysicalTransferTrainer,
    PhysicalTransferTrainingConfig,
)
from gaugeflow.production.teacher_feature_cache import MatPESTeacherFeatureCache
from gaugeflow.production.training import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs-per-rank", type=int, default=2)
    return parser.parse_args()


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _state_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_args()
    if arguments.graphs_per_rank < 1:
        raise ValueError("distributed smoke requires graphs on every rank")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("frozen distributed smoke requires exactly two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(5705)
    torch.cuda.manual_seed_all(5705)
    torch.use_deterministic_algorithms(True)

    metadata = read_production_checkpoint_metadata(arguments.a1_checkpoint)
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    standardization = metadata.get("lattice_standardization")
    if not all(isinstance(value, dict) for value in (model_config, training_config, standardization)):
        raise ValueError("A1 checkpoint metadata is incomplete")
    assert isinstance(model_config, dict)
    assert isinstance(training_config, dict)
    assert isinstance(standardization, dict)
    backbone = HybridCrystalDenoiser(**model_config).to(device)
    a1_ema = ExponentialMovingAverage(backbone, float(training_config["ema_decay"]))
    load_production_checkpoint(
        arguments.a1_checkpoint,
        model=backbone,
        ema=a1_ema,
        map_location=device,
    )
    a1_ema.copy_to(backbone)
    del a1_ema
    normalizer, vocabulary = load_functional_physical_normalizer(arguments.normalizer)
    feature_cache = MatPESTeacherFeatureCache(
        arguments.teacher_cache,
        index_manifest=arguments.index / "manifest.json",
        require_qualified=False,
    )
    dataset = IndexedMatPESDataset(arguments.index, "train")
    candidates = [
        (local, int(global_row))
        for local, global_row in enumerate(dataset.indices.tolist())
        if int(global_row) < feature_cache.row_count
    ]
    needed = arguments.graphs_per_rank * world_size
    if len(candidates) < needed:
        raise ValueError("bounded teacher cache lacks enough train rows for both ranks")
    selected = candidates[
        rank * arguments.graphs_per_rank : (rank + 1) * arguments.graphs_per_rank
    ]
    records = []
    for local_row, global_row in selected:
        record = dataset[local_row]
        feature = feature_cache.get(global_row, record.element_tokens.numel())
        if feature is None:
            raise ValueError("selected distributed smoke row lacks teacher features")
        records.append(replace(record, teacher_features=feature))
    physical_batch = collate_matpes_records(
        records,
        functional_vocabulary=vocabulary,
        teacher_dim=feature_cache.feature_dim,
    ).to(device)
    model = PhysicalRepresentationModel(
        backbone,
        teacher_dim=feature_cache.feature_dim,
        functional_count=len(vocabulary),
    ).to(device)
    diffusion = TensorFreeHybridDiffusion(
        backbone,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
        minimum_time=float(training_config["minimum_time"]),
        maximum_time=float(training_config["maximum_time"]),
        categorical_path=str(training_config["categorical_path"]),
        composition_conditioning=bool(training_config["composition_conditioning"]),
    )
    trainer = PhysicalTransferTrainer(
        model,
        diffusion,
        PhysicalTransferTrainingConfig(precision="bf16"),
        optimizer_owner=rank == 0,
    )
    counts = torch.bincount(
        physical_batch.batch, minlength=arguments.graphs_per_rank
    )
    blueprint = ParentBlueprintBatch.from_node_counts(counts, device=device)
    local_fraction = trainer.distributed_local_fraction(
        arguments.graphs_per_rank, device=device
    )
    device_generator = torch.Generator(device=device).manual_seed(8705 + rank)

    def one_step() -> torch.Tensor:
        trainer.begin_optimization_step()
        physical = trainer.accumulate_physical_step(
            physical_batch,
            normalizer,
            loss_weight=0.5 * local_fraction,
        )
        replay = trainer.accumulate_alex_replay_step(
            physical_batch.element_tokens,
            physical_batch.fractional_coordinates,
            physical_batch.lattice,
            physical_batch.batch,
            blueprint,
            loss_weight=0.5 * local_fraction,
            generator=device_generator,
        )
        gradient = trainer.finish_distributed_optimization_step(owner_rank=0)
        return torch.tensor(
            [
                float(physical.loss.detach()),
                float(physical.feature_loss.detach()),
                float(replay),
                float(gradient),
                float(physical_batch.targets.teacher_mask.sum()),
            ],
            device=device,
            dtype=torch.float64,
        )

    torch.cuda.reset_peak_memory_stats(device)
    warmup_metrics = one_step()
    saved_generator_state = device_generator.get_state().clone()
    saved_model = (
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if rank == 0
        else None
    )
    saved_trainer = _cpu_tree(trainer.state_dict()) if rank == 0 else None
    reference_metrics = one_step()
    reference_model_digest = _model_digest(model)
    reference_trainer_digest = _state_digest(trainer.state_dict()) if rank == 0 else None
    if rank == 0:
        assert saved_model is not None and isinstance(saved_trainer, dict)
        model.load_state_dict(saved_model, strict=True)
        trainer.load_state_dict(saved_trainer)
    trainer.broadcast_distributed_state(owner_rank=0)
    device_generator.set_state(saved_generator_state)
    repeated_metrics = one_step()
    repeated_model_digest = _model_digest(model)
    repeated_trainer_digest = _state_digest(trainer.state_dict()) if rank == 0 else None
    digest = repeated_model_digest
    digests: list[str | None] = [None for _ in range(world_size)]
    dist.all_gather_object(digests, digest)
    owner = torch.tensor(int(trainer.optimizer is not None), device=device)
    dist.all_reduce(owner, op=dist.ReduceOp.SUM)
    metrics = torch.cat(
        (
            warmup_metrics,
            reference_metrics,
            repeated_metrics,
            torch.tensor(
                [float(torch.cuda.max_memory_allocated(device))],
                device=device,
                dtype=torch.float64,
            ),
        )
    )
    gathered = [torch.zeros_like(metrics) for _ in range(world_size)]
    dist.all_gather(gathered, metrics)
    if rank == 0:
        result = {
            "schema": "gaugeflow.stage_b_distributed_smoke.v1",
            "qualified": False,
            "scope": "bounded two-GPU software synchronization smoke; not physical qualification",
            "a1_checkpoint_sha256": sha256_file(arguments.a1_checkpoint),
            "world_size": world_size,
            "optimizer_owner_count": int(owner),
            "parameter_digests": digests,
            "exact_parameter_replication": len(set(digests)) == 1,
            "resume_reference_model_digest": reference_model_digest,
            "resume_repeated_model_digest": repeated_model_digest,
            "resume_reference_trainer_digest": reference_trainer_digest,
            "resume_repeated_trainer_digest": repeated_trainer_digest,
            "resume_metric_max_abs_error": max(
                float((value[5:10] - value[10:15]).abs().max())
                for value in gathered
            ),
            "exact_resume": reference_model_digest == repeated_model_digest
            and reference_trainer_digest == repeated_trainer_digest
            and all(torch.equal(value[5:10], value[10:15]) for value in gathered),
            "step": trainer.step,
            "per_rank": [
                {
                    "warmup_physical_loss": float(value[0]),
                    "warmup_feature_loss": float(value[1]),
                    "warmup_replay_loss": float(value[2]),
                    "reference_physical_loss": float(value[5]),
                    "reference_feature_loss": float(value[6]),
                    "reference_replay_loss": float(value[7]),
                    "global_gradient_norm": float(value[8]),
                    "teacher_nodes": int(value[9]),
                    "peak_cuda_memory_bytes": int(value[15]),
                }
                for value in gathered
            ],
            "finite": bool(all(torch.isfinite(value).all() for value in gathered)),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if arguments.output.exists():
            raise FileExistsError(f"refusing to overwrite {arguments.output}")
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
