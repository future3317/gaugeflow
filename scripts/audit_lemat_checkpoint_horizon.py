"""Locate malformed LeMat rows on the exact stream following a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object
from gaugeflow.production.balanced_rank_sharded_data import BalancedRankShardedStream
from gaugeflow.production.lemat_index import IndexedLeMatDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lemat-index", type=Path, required=True)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _row_diagnostic(dataset: IndexedLeMatDataset, split_index: int) -> dict[str, Any]:
    global_row = int(dataset.indices[split_index])
    source = int(dataset.source_index[global_row])
    group = int(dataset.row_group[global_row])
    row_in_group = int(dataset.row_in_group[global_row])
    raw = dataset._read_row_group(source, group).slice(row_in_group, 1).to_pylist()[0]
    cartesian = torch.as_tensor(raw.get("cartesian_site_positions"))
    species = raw.get("species_at_sites")
    result: dict[str, Any] = {
        "split_index": split_index,
        "global_index_row": global_row,
        "source_index": source,
        "source_path": str(dataset.source_paths[source]),
        "row_group": group,
        "row_in_group": row_in_group,
        "immutable_id": raw.get("immutable_id"),
        "functional": raw.get("functional"),
        "indexed_node_count": int(dataset.node_count[global_row]),
        "row_nsites": raw.get("nsites"),
        "cartesian_shape": list(cartesian.shape),
        "species_count": len(species) if isinstance(species, list) else None,
    }
    try:
        dataset[split_index]
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    else:
        result["error"] = None
    return result


def main() -> None:
    args = parse_args()
    if args.skip_batches < 0 or args.batches < 1 or args.output.exists():
        raise ValueError("audit horizon must be valid and output must not exist")
    protocol = load_json_object(args.protocol)
    training = protocol.get("training")
    if not isinstance(training, dict):
        raise ValueError("Stage-C protocol lacks training configuration")
    dataset = IndexedLeMatDataset(args.lemat_index, "train")
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
    payload: Any = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    runtime = payload.get("rank_runtime_states") if isinstance(payload, dict) else None
    if not isinstance(runtime, list) or not runtime or not isinstance(runtime[0], dict):
        raise ValueError("checkpoint lacks rank-0 runtime state")
    streams = runtime[0].get("streams")
    lemat_state = streams.get("lemat") if isinstance(streams, dict) else None
    if not isinstance(lemat_state, dict):
        raise ValueError("checkpoint lacks rank-0 LeMat stream state")
    stream.load_state_dict(lemat_state)
    for _ in range(args.skip_batches):
        stream.next_indices()

    failures: list[dict[str, Any]] = []
    for offset in range(1, args.batches + 1):
        indices = stream.next_indices()
        try:
            dataset.select(indices)
        except (KeyError, RuntimeError, TypeError, ValueError):
            for index in indices.tolist():
                diagnostic = _row_diagnostic(dataset, index)
                if diagnostic["error"] is not None:
                    diagnostic["batch_offset"] = offset
                    diagnostic["checkpoint_batch_offset"] = args.skip_batches + offset
                    failures.append(diagnostic)
            if failures:
                break
    result = {
        "schema": "gaugeflow.lemat_checkpoint_horizon_audit.v1",
        "checkpoint": str(args.checkpoint),
        "batches_skipped": args.skip_batches,
        "batches_requested": args.batches,
        "batches_examined": offset,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
