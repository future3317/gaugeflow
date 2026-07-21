"""Exercise source-balanced LeMat rank shards without running model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gaugeflow.production.balanced_rank_sharded_data import BalancedRankShardedStream
from gaugeflow.production.lemat_index import IndexedLeMatDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--allow-bounded-index", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batches < 2 or args.world_size < 1:
        raise ValueError("balanced LeMat smoke dimensions are invalid")
    dataset = IndexedLeMatDataset(
        args.index,
        "train",
        require_qualified=not args.allow_bounded_index,
    )
    random_access_checked: list[str] = []
    for functional, name in enumerate(dataset.functional_names):
        row = int(torch.nonzero(dataset.functional_group_index == functional)[0])
        record = dataset[row]
        if record.functional != name or not all(
            bool(torch.isfinite(value).all())
            for value in (
                record.fractional_coordinates,
                record.lattice,
                record.energy_per_atom_ev,
                record.forces_ev_per_angstrom,
                record.stress_kelvin_gpa,
            )
        ):
            raise AssertionError(f"LeMat {name} random-access record is invalid")
        random_access_checked.append(name)
    weights = [1.0] * len(dataset.functional_names)
    streams = [
        BalancedRankShardedStream(
            dataset.functional_group_index,
            weights,
            args.global_batch_size,
            rank=rank,
            world_size=args.world_size,
            seed=args.seed,
        )
        for rank in range(args.world_size)
    ]
    source_counts = torch.zeros(len(weights), dtype=torch.long)
    rank_counts = torch.zeros(args.world_size, dtype=torch.long)
    resume_state: dict[str, object] | None = None
    resume_reference: torch.Tensor | None = None
    for batch in range(args.batches):
        local = [stream.next_indices() for stream in streams]
        global_indices = torch.empty(sum(value.numel() for value in local), dtype=torch.long)
        for rank, value in enumerate(local):
            global_indices[rank :: args.world_size] = value
            rank_counts[rank] += value.numel()
        source_counts += torch.bincount(
            dataset.functional_group_index[global_indices], minlength=len(weights)
        )
        states = [stream.state for stream in streams]
        if len({(state.source_epochs, state.source_offsets) for state in states}) != 1:
            raise AssertionError("LeMat rank streams disagree on global source cursors")
        if batch == args.batches // 2:
            resume_state = streams[0].state_dict()
        elif batch == args.batches // 2 + 1:
            resume_reference = local[0]
    if resume_state is None or resume_reference is None:
        raise AssertionError("balanced LeMat resume checkpoint was not exercised")
    resumed = BalancedRankShardedStream(
        dataset.functional_group_index,
        weights,
        args.global_batch_size,
        rank=0,
        world_size=args.world_size,
        seed=args.seed,
    )
    resumed.load_state_dict(resume_state)
    if not torch.equal(resumed.next_indices(), resume_reference):
        raise AssertionError("balanced LeMat stream did not resume exactly")
    fractions = source_counts.double() / source_counts.sum()
    target = torch.full_like(fractions, 1.0 / len(weights))
    result = {
        "schema": "gaugeflow.lemat_balanced_stream_smoke.v1",
        "index": str(args.index.resolve()),
        "functional_names": list(dataset.functional_names),
        "dataset_functional_rows": {
            name: int((dataset.functional_group_index == index).sum())
            for index, name in enumerate(dataset.functional_names)
        },
        "functional_random_access_checked": random_access_checked,
        "batches": args.batches,
        "global_batch_size": args.global_batch_size,
        "world_size": args.world_size,
        "rank_examples": rank_counts.tolist(),
        "sampled_functional_fraction": {
            name: float(fractions[index])
            for index, name in enumerate(dataset.functional_names)
        },
        "maximum_absolute_balance_error": float((fractions - target).abs().max()),
        "exact_resume": True,
        "finite": True,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
