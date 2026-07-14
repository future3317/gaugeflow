"""Profile Gate A conditioning modes without changing their scientific code path."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from gaugeflow.conditioning import randomize_tensor_orbit_representative
from gaugeflow.data import PiezoCrystalDataset, collate_crystals
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.stabilizer import proper_unimodular_candidates
from gaugeflow.tensor import isotypic_slices, normalize_isotypic


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(callable_, device: torch.device):
    synchronize(device)
    start = time.perf_counter()
    value = callable_()
    synchronize(device)
    return value, time.perf_counter() - start


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def build_panel(args, protocol: dict):
    started = time.perf_counter()
    dataset = PiezoCrystalDataset(
        args.train_csv,
        split_manifest=args.split_manifest,
        split="train",
        target_cache_dir=args.target_cache_dir,
        preprocessed_cache=args.preprocessed_cache,
    )
    lookup = {str(value): index for index, value in enumerate(dataset.frame.material_id)}
    ids = [str(value) for value in protocol["material_ids"]]
    missing = [value for value in ids if value not in lookup]
    if missing:
        raise ValueError(f"Gate A IDs missing from dataset: {missing}")
    indices = [lookup[value] for value in ids]
    conditions = torch.stack([dataset._condition_for_index(index)[0] for index in indices])
    scales = torch.stack(
        [conditions[:, block].square().mean().sqrt().clamp_min(1e-8) for block in isotypic_slices()]
    )
    records = [dataset[index] for index in indices]
    preprocessing_seconds = time.perf_counter() - started
    loader = DataLoader(
        records,
        batch_size=protocol["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=0,
    )
    return loader, scales, preprocessing_seconds, ids


def run_step(model, matcher, optimizer, batch, base_condition, mode):
    with record_function("profile.host_to_device"):
        batch.piezo_irreps = base_condition
        if mode == "direct_irrep":
            batch.piezo_irreps = randomize_tensor_orbit_representative(base_condition)
    optimizer.zero_grad(set_to_none=True)
    with record_function("profile.neural_score_forward"):
        terms = matcher.loss(model, batch)
    with record_function("profile.backward"):
        terms["loss"].backward()
    with record_function("profile.optimizer"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return terms


def table_text(profiler, sort_by: str) -> str:
    return profiler.key_averages().table(sort_by=sort_by, row_limit=20)


def profile_mode(mode, args, protocol, loader, scales, output_dir, device):
    training = protocol["training"]
    torch.manual_seed(training["seed"])
    model = GaugeFlowVectorField(
        training["hidden_dim"], training["layers"], training["orbit_frames"],
        conditioning_mode=mode,
    )
    normalized_records = []
    for source in loader.dataset:
        record = source.clone()
        record.piezo_irreps = normalize_isotypic(record.piezo_irreps, scales)
        if mode in {"stabilizer_pooling", "orbit_alignment"}:
            with torch.no_grad():
                record.condition_orbit = model.response.precompute_condition_orbit(
                    record.piezo_irreps
                )
        normalized_records.append(record)
    model = model.to(device)
    batch = collate_crystals(normalized_records).to(device)
    base_condition = batch.piezo_irreps.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])
    matcher = RiemannianCrystalFlowMatcher(uncertainty_weight=0.0)

    for _ in range(args.warmup_steps):
        run_step(model, matcher, optimizer, batch, base_condition, mode)
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    step_seconds = []
    cuda_step_milliseconds = []
    data_seconds = []
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
    ) as profiler:
        for _ in range(args.profile_steps):
            start = time.perf_counter()
            cuda_start = cuda_end = None
            if device.type == "cuda":
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            with record_function("profile.data_loading"):
                data_start = time.perf_counter()
                batch.piezo_irreps = base_condition
                data_seconds.append(time.perf_counter() - data_start)
            run_step(model, matcher, optimizer, batch, base_condition, mode)
            if cuda_end is not None:
                cuda_end.record()
            synchronize(device)
            step_seconds.append(time.perf_counter() - start)
            if cuda_start is not None and cuda_end is not None:
                cuda_step_milliseconds.append(cuda_start.elapsed_time(cuda_end))
            profiler.step()

    trace_path = output_dir / f"{mode}_{args.phase}_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    cpu_table = table_text(profiler, "self_cpu_time_total")
    (output_dir / f"{mode}_{args.phase}_top20_cpu.txt").write_text(cpu_table, encoding="utf-8")
    cuda_table = "CUDA profiling unavailable on CPU device."
    if device.type == "cuda":
        cuda_table = table_text(profiler, "self_cuda_time_total")
    (output_dir / f"{mode}_{args.phase}_top20_cuda.txt").write_text(cuda_table, encoding="utf-8")

    keys = profiler.key_averages()
    copies = sum(
        event.count for event in keys
        if any(token in event.key.lower() for token in ("memcpy", "_to_copy", "copy_"))
    )
    return {
        "mode": mode,
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
        "mean_step_seconds": statistics.mean(step_seconds),
        "median_step_seconds": statistics.median(step_seconds),
        "min_step_seconds": min(step_seconds),
        "max_step_seconds": max(step_seconds),
        "mean_data_seconds": statistics.mean(data_seconds),
        "mean_cuda_event_milliseconds": (
            statistics.mean(cuda_step_milliseconds) if cuda_step_milliseconds else None
        ),
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2 if device.type == "cuda" else 0.0
        ),
        "gpu_peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024.0**2 if device.type == "cuda" else 0.0
        ),
        "profiled_copy_ops": copies,
        "trace": str(trace_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a_v1.json"))
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/profiler_after"))
    parser.add_argument("--phase", choices=("before", "after"), default="after")
    parser.add_argument("--modes", nargs="+", default=["stabilizer_pooling", "orbit_alignment"])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.profile_steps < 1:
        parser.error("warmup/profile steps must be positive")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader, scales, preprocessing_seconds, material_ids = build_panel(args, protocol)

    proper_unimodular_candidates.cache_clear()
    _, cold_catalogue_seconds = timed(proper_unimodular_candidates, torch.device("cpu"))
    catalogue = proper_unimodular_candidates()
    results = []
    for mode in args.modes:
        results.append(profile_mode(mode, args, protocol, loader, scales, args.output_dir, device))
    payload = {
        "schema": 1,
        "phase": args.phase,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "material_ids": material_ids,
        "candidate_count": int(catalogue.shape[0]),
        "cold_catalogue_seconds": cold_catalogue_seconds,
        "panel_preprocessing_seconds": preprocessing_seconds,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
