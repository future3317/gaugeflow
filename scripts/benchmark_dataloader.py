"""Benchmark preprocessed-cache DataLoader settings on a frozen split."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gaugeflow.data import PiezoCrystalDataset, collate_crystals


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_config(dataset, config, args, device):
    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    if config["num_workers"]:
        kwargs.update(
            persistent_workers=config["persistent_workers"],
            prefetch_factor=config["prefetch_factor"],
        )
    construction_start = time.perf_counter()
    loader = DataLoader(**kwargs)
    construction_seconds = time.perf_counter() - construction_start
    iterator = iter(loader)
    data_times = []
    copy_times = []
    first_batch_seconds = None
    for step in range(args.batches):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_elapsed = time.perf_counter() - start
        if first_batch_seconds is None:
            first_batch_seconds = data_elapsed
        copy_start = time.perf_counter()
        batch = batch.to(device, non_blocking=config["pin_memory"])
        synchronize(device)
        copy_elapsed = time.perf_counter() - copy_start
        if step >= args.discard_batches:
            data_times.append(data_elapsed)
            copy_times.append(copy_elapsed)
    return {
        **config,
        "construction_seconds": construction_seconds,
        "first_batch_seconds": first_batch_seconds,
        "measured_batches": len(data_times),
        "mean_data_seconds": statistics.mean(data_times),
        "median_data_seconds": statistics.median(data_times),
        "p95_data_seconds": sorted(data_times)[max(0, int(0.95 * len(data_times)) - 1)],
        "mean_host_to_device_seconds": statistics.mean(copy_times),
        "batches_per_second_data_only": 1.0 / statistics.mean(data_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/dataloader_benchmark.json"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--discard-batches", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batches <= args.discard_batches:
        parser.error("batches must exceed discard-batches")
    device = torch.device(args.device)
    init_start = time.perf_counter()
    dataset = PiezoCrystalDataset(
        args.csv_dir,
        split_manifest=args.split_manifest,
        split=args.split,
        target_cache_dir=args.target_cache_dir,
        preprocessed_cache=args.preprocessed_cache,
    )
    cache_load_seconds = time.perf_counter() - init_start
    configs = [
        {"num_workers": workers, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2}
        for workers in (0, 2, 4, 8)
    ]
    configs.append(
        {"num_workers": 0, "pin_memory": True, "persistent_workers": False, "prefetch_factor": 2}
    )
    configs.extend(
        {
            "num_workers": workers,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": prefetch,
        }
        for workers in (2, 4, 8)
        for prefetch in (2, 4)
    )
    results = []
    for config in configs:
        result = run_config(dataset, config, args, device)
        results.append(result)
        print(result)
    payload = {
        "schema": 1,
        "device": str(device),
        "split": args.split,
        "records": len(dataset),
        "batch_size": args.batch_size,
        "cache_load_seconds": cache_load_seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
