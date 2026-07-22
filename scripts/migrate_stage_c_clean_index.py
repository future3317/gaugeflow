"""Migrate a Stage-C checkpoint onto a cleaned LeMat sampling boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from train_lemat_continued_pretraining import (
    STAGE_C_PROTOCOL,
    _verify_protocol_implementation,
)

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.balanced_rank_sharded_data import BalancedRankShardedStream
from gaugeflow.production.lemat_index import IndexedLeMatDataset
from gaugeflow.production.physical_checkpointing import (
    PHYSICAL_CHECKPOINT_SCHEMA,
    read_physical_checkpoint_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lemat-index", type=Path, required=True)
    parser.add_argument("--matpes-index", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _new_lemat_state(
    dataset: IndexedLeMatDataset,
    training: dict[str, Any],
    batches: int,
) -> dict[str, Any]:
    weight_mapping = training.get("lemat_functional_weights")
    if not isinstance(weight_mapping, dict):
        raise ValueError("Stage-C protocol lacks LeMat functional weights")
    stream = BalancedRankShardedStream(
        dataset.functional_group_index,
        [float(weight_mapping[name]) for name in dataset.functional_names],
        int(training["global_lemat_batch"]),
        rank=0,
        world_size=1,
        seed=int(training["seed"]) + 101,
        block_index=dataset.sampling_block_index,
    )
    for _ in range(batches):
        stream.next_indices()
    return stream.state_dict()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError("refusing to overwrite a migrated checkpoint")
    protocol = load_json_object(args.protocol)
    training = protocol.get("training")
    prerequisites = protocol.get("prerequisites")
    if (
        protocol.get("protocol") != STAGE_C_PROTOCOL
        or protocol.get("status_before_run") != "frozen_method_not_run"
        or not isinstance(training, dict)
        or not isinstance(prerequisites, dict)
    ):
        raise ValueError("clean-index migration requires the frozen Stage-C protocol")
    implementation = _verify_protocol_implementation(protocol)
    observed = {
        "stage_b_checkpoint_sha256": sha256_file(args.stage_b_checkpoint),
        "lemat_index_manifest_sha256": sha256_file(args.lemat_index / "manifest.json"),
        "lemat_index_sha256": sha256_file(args.lemat_index / "index.pt"),
        "matpes_index_manifest_sha256": sha256_file(args.matpes_index / "manifest.json"),
        "matpes_index_sha256": sha256_file(args.matpes_index / "index.pt"),
        "matpes_normalizer_sha256": sha256_file(args.normalizer),
        "teacher_cache_manifest_sha256": sha256_file(args.teacher_cache / "manifest.json"),
        "alex_cache_manifest_sha256": sha256_file(args.alex_cache / "manifest.json"),
        **implementation,
    }
    if any(observed[name] != str(prerequisites.get(name)) for name in observed):
        raise ValueError("clean-index migration prerequisites disagree with the protocol")

    old_metadata = read_physical_checkpoint_metadata(args.source_checkpoint)
    if old_metadata.get("protocol") != "stage_c_lemat_continued_pretraining_v1":
        raise ValueError("source is not the interrupted Stage-C-v1 run")
    payload: Any = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or payload.get("schema") != PHYSICAL_CHECKPOINT_SCHEMA:
        raise ValueError("source checkpoint schema mismatch")
    trainer = payload.get("trainer")
    runtime = payload.get("rank_runtime_states")
    if (
        not isinstance(trainer, dict)
        or not isinstance(trainer.get("step"), int)
        or not isinstance(runtime, list)
        or len(runtime) != 3
        or not all(isinstance(state, dict) for state in runtime)
    ):
        raise ValueError("source checkpoint lacks trainer or three-rank runtime state")
    stage_b_step = int(training["stage_b_completed_step"])
    completed = int(trainer["step"]) - stage_b_step
    if completed < 1:
        raise ValueError("source checkpoint has no Stage-C updates")

    dataset = IndexedLeMatDataset(args.lemat_index, "train")
    initial_lemat_state = _new_lemat_state(dataset, training, 0)
    resumed_lemat_state = _new_lemat_state(dataset, training, completed)
    for rank, state in enumerate(runtime):
        assert isinstance(state, dict)
        streams = state.get("streams")
        if not isinstance(streams, dict) or not isinstance(streams.get("lemat"), dict):
            raise ValueError("source checkpoint has an invalid LeMat stream state")
        streams["lemat"] = resumed_lemat_state if rank == 0 else initial_lemat_state

    checkpoint_migration = protocol.get("checkpoint_migration")
    if not isinstance(checkpoint_migration, dict):
        raise ValueError("Stage-C protocol lacks its checkpoint migration contract")
    if checkpoint_migration.get("source_checkpoint_sha256") != sha256_file(
        args.source_checkpoint
    ):
        raise ValueError("checkpoint migration source hash disagrees with the protocol")
    metadata = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "prerequisites": observed,
        "stage_b_metadata": read_physical_checkpoint_metadata(args.stage_b_checkpoint),
        "world_size": 3,
        "seed": int(training["seed"]),
        "migration": checkpoint_migration,
    }
    payload["rank_runtime_states"] = runtime
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    sidecar = {
        "schema": PHYSICAL_CHECKPOINT_SCHEMA,
        "weights_file": args.output.name,
        "weights_sha256": sha256_file(args.output),
        "metadata": metadata,
        "metadata_sha256": canonical_json_hash(metadata),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
