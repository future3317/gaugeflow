"""Unified Gate A throughput benchmark on the frozen eight-record panel."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

from gaugeflow.conditioning import randomize_tensor_orbit_representative
from gaugeflow.data import PiezoCrystalDataset, collate_crystals
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.stabilizer import proper_unimodular_candidates
from gaugeflow.tensor import isotypic_slices, normalize_isotypic


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ResourceSampler:
    """Best-effort process CPU and nvidia-smi sampling outside timed steps."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.cpu: list[float] = []
        self.gpu: list[float] = []
        self.vram: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        try:
            import psutil

            process = psutil.Process()
            process.cpu_percent(None)
        except ImportError:
            process = None
        while not self._stop.wait(self.interval):
            if process is not None:
                self.cpu.append(process.cpu_percent(None))
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).splitlines()[0]
                utilization, memory = (float(value.strip()) for value in output.split(","))
                self.gpu.append(utilization)
                self.vram.append(memory)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)

    def summary(self) -> dict[str, float | int | None]:
        return {
            "resource_samples": max(len(self.cpu), len(self.gpu)),
            "process_cpu_percent_mean": statistics.mean(self.cpu) if self.cpu else None,
            "gpu_utilization_percent_mean": statistics.mean(self.gpu) if self.gpu else None,
            "nvidia_smi_vram_mib_peak": max(self.vram) if self.vram else None,
        }


def load_panel(args, protocol: dict):
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
    records = [dataset[lookup[value]] for value in ids]
    conditions = torch.stack([record.piezo_irreps[0] for record in records])
    scales = torch.stack(
        [conditions[:, block].square().mean().sqrt().clamp_min(1e-8) for block in isotypic_slices()]
    )
    return records, scales, time.perf_counter() - started, ids, dataset.preprocessed_manifest


def prepare_batch(records, scales, model, mode, device):
    normalized_records = []
    for source in records:
        record = source.clone()
        record.piezo_irreps = normalize_isotypic(record.piezo_irreps, scales)
        if mode in {"stabilizer_pooling", "orbit_alignment"}:
            with torch.no_grad():
                record.condition_orbit = model.response.precompute_condition_orbit(record.piezo_irreps)
        normalized_records.append(record)
    batch = collate_crystals(normalized_records).to(device)
    return batch, batch.piezo_irreps.detach().clone()


def train_step(model, matcher, optimizer, batch, base_condition, mode):
    batch.piezo_irreps = base_condition
    if mode == "direct_irrep":
        batch.piezo_irreps = randomize_tensor_orbit_representative(batch.piezo_irreps)
    optimizer.zero_grad(set_to_none=True)
    terms = matcher.loss(model, batch)
    terms["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return terms


def benchmark_mode(mode, args, protocol, records, scales, device):
    training = protocol["training"]
    torch.manual_seed(training["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["seed"])
    model = GaugeFlowVectorField(
        training["hidden_dim"],
        training["layers"],
        training["orbit_frames"],
        conditioning_mode=mode,
    ).to(device)
    batch, base_condition = prepare_batch(records, scales, model, mode, device)
    compile_invocation_seconds = None
    if args.torch_compile:
        compile_started = time.perf_counter()
        model = torch.compile(model, dynamic=True)
        compile_invocation_seconds = time.perf_counter() - compile_started
    matcher = RiemannianCrystalFlowMatcher(uncertainty_weight=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])

    warmup_started = time.perf_counter()
    for _ in range(args.warmup_steps):
        train_step(model, matcher, optimizer, batch, base_condition, mode)
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wall_seconds: list[float] = []
    cuda_milliseconds: list[float] = []
    with ResourceSampler(args.resource_interval) as sampler:
        for _ in range(args.benchmark_steps):
            cuda_start = cuda_end = None
            if device.type == "cuda":
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            started = time.perf_counter()
            train_step(model, matcher, optimizer, batch, base_condition, mode)
            if cuda_end is not None:
                cuda_end.record()
            synchronize(device)
            wall_seconds.append(time.perf_counter() - started)
            if cuda_start is not None and cuda_end is not None:
                cuda_milliseconds.append(cuda_start.elapsed_time(cuda_end))

        sample_started = time.perf_counter()
        with torch.no_grad():
            for repeat in range(args.sample_repeats):
                torch.manual_seed(args.sample_seed + repeat)
                matcher.sample(model, batch, steps=args.sample_steps)
        synchronize(device)
        sampling_seconds = time.perf_counter() - sample_started
    utilization = sampler.summary()

    mean_step = statistics.mean(wall_seconds)
    return {
        "mode": mode,
        "warmup_steps": args.warmup_steps,
        "benchmark_steps": args.benchmark_steps,
        "torch_compile": args.torch_compile,
        "compile_invocation_seconds": compile_invocation_seconds,
        "warmup_seconds_including_lazy_compile": warmup_seconds,
        "mean_step_seconds": mean_step,
        "median_step_seconds": statistics.median(wall_seconds),
        "min_step_seconds": min(wall_seconds),
        "max_step_seconds": max(wall_seconds),
        "mean_cuda_event_milliseconds": statistics.mean(cuda_milliseconds) if cuda_milliseconds else None,
        "measured_benchmark_seconds": sum(wall_seconds),
        "projected_400_step_seconds": mean_step * training["steps"],
        "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024.0**2 if device.type == "cuda" else 0.0
        ),
        "gpu_peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024.0**2 if device.type == "cuda" else 0.0
        ),
        "sampling_seconds": sampling_seconds,
        "sample_graphs": batch.num_graphs * args.sample_repeats,
        "sampling_graphs_per_second": batch.num_graphs * args.sample_repeats / sampling_seconds,
        "sampling_integration_graph_steps_per_second": (
            batch.num_graphs * args.sample_repeats * args.sample_steps / sampling_seconds
        ),
        **utilization,
    }


def add_comparisons(payload: dict, before_summary: Path | None) -> None:
    results = {row["mode"]: row for row in payload["results"]}
    direct = results["direct_irrep"]["mean_step_seconds"]
    for row in results.values():
        row["slowdown_vs_direct_irrep"] = row["mean_step_seconds"] / direct
    if "orbit_alignment" in results:
        results["orbit_alignment"]["incremental_seconds_per_candidate_vs_direct"] = max(
            results["orbit_alignment"]["mean_step_seconds"] - direct, 0.0
        ) / payload["candidate_count"]
    if before_summary and before_summary.exists():
        before = json.loads(before_summary.read_text(encoding="utf-8"))
        old = {row["mode"]: row for row in before["results"]}
        for mode, row in results.items():
            if mode in old:
                row["speedup_vs_before"] = old[mode]["mean_step_seconds"] / row["mean_step_seconds"]


def write_markdown(payload: dict, path: Path) -> None:
    rows = []
    for value in payload["results"]:
        display = dict(value)
        display["process_cpu_percent_mean"] = (
            f"{value['process_cpu_percent_mean']:.1f}%"
            if value["process_cpu_percent_mean"] is not None else "n/a"
        )
        display["gpu_utilization_percent_mean"] = (
            f"{value['gpu_utilization_percent_mean']:.1f}%"
            if value["gpu_utilization_percent_mean"] is not None else "n/a"
        )
        display["speedup"] = (
            f"{value['speedup_vs_before']:.1f}x" if "speedup_vs_before" in value else "n/a"
        )
        rows.append(
            "| {mode} | {mean_step_seconds:.4f} | {projected_400_step_seconds:.1f} | "
            "{process_cpu_percent_mean} | {gpu_utilization_percent_mean} | {cpu_peak_rss_mib:.1f} | "
            "{gpu_peak_allocated_mib:.1f} | {slowdown_vs_direct_irrep:.2f}x | {speedup} |".format(**display)
        )
    path.write_text(
        "# GaugeFlow unified performance benchmark\n\n"
        "Ten warm-up steps and twenty measured optimizer steps use the frozen Gate A panel, "
        "resident GPU batch, seed, capacity, and complete 792-candidate definition. Projected "
        "time is a linear 400-step estimate, not a replacement for completed training wall time.\n\n"
        "| method | sec/step | projected 400-step s | process CPU | GPU | RSS MiB | torch VRAM MiB | vs direct | vs before |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\nSampling throughput, CUDA-event timing, utilization sample counts, and exact definitions are in "
        "`reports/performance_benchmark_after.json`. Short utilization windows are noisy and are "
        "reported as diagnostics rather than scientific outcomes.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a_v1.json"))
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/performance_benchmark_after.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/performance_benchmark_after.md"))
    parser.add_argument("--before-summary", type=Path, default=Path("reports/profiler_before/summary.json"))
    parser.add_argument("--modes", nargs="+", default=["raw_tensor", "direct_irrep", "stabilizer_pooling", "orbit_alignment"])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--sample-repeats", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=73021)
    parser.add_argument("--resource-interval", type=float, default=0.2)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    records, scales, panel_seconds, ids, cache_manifest = load_panel(args, protocol)
    proper_unimodular_candidates.cache_clear()
    started = time.perf_counter()
    catalogue = proper_unimodular_candidates()
    cold_catalogue_seconds = time.perf_counter() - started
    results = [
        benchmark_mode(mode, args, protocol, records, scales, device) for mode in args.modes
    ]
    payload = {
        "schema": 1,
        "phase": "after_optional_compile" if args.torch_compile else "after",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "material_ids": ids,
        "candidate_count": int(catalogue.shape[0]),
        "cold_catalogue_seconds": cold_catalogue_seconds,
        "panel_preprocessing_seconds": panel_seconds,
        "preprocessed_manifest": cache_manifest,
        "results": results,
    }
    add_comparisons(payload, args.before_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
