"""Run the versioned S0.4.1 performance-only Cartesian-atlas qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from gaugeflow.production.cartesian_gauge_atlas import StratifiedCartesianGaugeAtlas
from gaugeflow.tensor import piezo_from_irreps

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paper_s0_4_1_cartesian_atlas_runtime_v1.json"
OUT = ROOT / "reports" / "paper_s0_4_1_cartesian_atlas_runtime_v1"
PREDECESSOR = ROOT / "reports" / "paper_s0_4_cartesian_atlas_prior_v1"


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run(arguments: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"arguments": arguments, "returncode": completed.returncode, "output": completed.stdout}


def generic_reference_equivalence() -> dict[str, float | int]:
    """Compare the production fast path with the original deduplicated measure."""
    torch.manual_seed(10411)
    atlas = StratifiedCartesianGaugeAtlas(16).double().eval()
    covariance = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    frame = atlas._frame_data(covariance, directional=True)
    raw_rotations, raw_prior = atlas._raw_candidate_measure(frame, frame)
    reference = atlas._deduplicate_measure(raw_rotations, raw_prior)
    optimized = atlas._candidate_measure(frame, frame)
    tensor = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64))[0]
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)

    def evaluate(measure):
        rotated = atlas._rotate_rank_three(tensor, measure.rotations)
        score = torch.einsum("fijk,cijk,c->f", rotated, query, atlas.score_channel)
        posterior = torch.softmax(score + measure.prior.log(), dim=0)
        aligned = torch.einsum("f,fijk->ijk", posterior, rotated)
        return posterior, aligned

    reference_posterior, reference_aligned = evaluate(reference)
    optimized_posterior, optimized_aligned = evaluate(optimized)
    scale = torch.linalg.vector_norm(reference_aligned).clamp_min(1e-15)
    return {
        "raw_candidates": optimized.raw_count,
        "reference_unique_candidates": int(reference.rotations.shape[0]),
        "optimized_unique_candidates": int(optimized.rotations.shape[0]),
        "aligned_relative_error": float(torch.linalg.vector_norm(reference_aligned - optimized_aligned) / scale),
        "sorted_posterior_l1_error": float(
            (reference_posterior.sort().values - optimized_posterior.sort().values).abs().sum()
        ),
        "prior_l1_error": float((reference.prior.sort().values - optimized.prior.sort().values).abs().sum()),
    }


def _timed_cuda(operation: Callable[[], object], repeats: int) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def candidate_profile() -> dict[str, float | str]:
    if not torch.cuda.is_available():
        return {"status": "no_cuda"}
    device = torch.device("cuda")
    atlas = StratifiedCartesianGaugeAtlas(16).to(device).eval()
    covariance = torch.diag(torch.tensor([0.0, 0.2, 1.0], device=device))
    frame = atlas._frame_data(covariance, directional=True)

    def reference_deduplication():
        rotations, prior = atlas._raw_candidate_measure(frame, frame)
        return atlas._deduplicate_measure(rotations, prior)

    return {
        "status": torch.cuda.get_device_name(device),
        "reference_raw_plus_dedup_ms": _timed_cuda(reference_deduplication, 20),
        "optimized_candidate_measure_ms": _timed_cuda(lambda: atlas._candidate_measure(frame, frame), 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", action="store_true")
    arguments = parser.parse_args()
    if not arguments.official:
        raise SystemExit("S0.4.1 requires --official and writes a versioned immutable result")
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_not_run":
        raise SystemExit(f"S0.4.1 is already frozen with status {protocol['status']}")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    predecessor_hash_before = _tree_sha256(PREDECESSOR)
    thresholds = protocol["frozen_thresholds"]

    equivalence = generic_reference_equivalence()
    profile = candidate_profile()
    old_runner = runpy.run_path(str(ROOT / "scripts" / "run_paper_s0_3_cartesian_atlas_audit.py"))
    benchmark = old_runner["cuda_benchmark"]()
    commands = {
        "pytest": _run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"]),
        "ruff": _run([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
        "mypy": _run([sys.executable, "-m", "mypy", "src/gaugeflow/production"]),
    }
    predecessor_hash_after = _tree_sha256(PREDECESSOR)
    checks = {
        "predecessor_immutable": predecessor_hash_before == predecessor_hash_after,
        "generic_raw_candidates": equivalence["raw_candidates"] == thresholds["generic_raw_candidates"],
        "generic_unique_candidates": equivalence["optimized_unique_candidates"]
        == thresholds["generic_unique_candidates"],
        "reference_unique_candidates": equivalence["reference_unique_candidates"]
        == thresholds["generic_unique_candidates"],
        "reference_aligned": equivalence["aligned_relative_error"]
        <= thresholds["reference_aligned_relative_error_max"],
        "reference_posterior": equivalence["sorted_posterior_l1_error"]
        <= thresholds["reference_sorted_posterior_l1_error_max"],
        "reference_prior": equivalence["prior_l1_error"] <= thresholds["reference_prior_l1_error_max"],
        "cuda_latency": benchmark.get("atlas_ms_per_forward", float("inf"))
        <= thresholds["cuda_atlas_latency_ms_max"],
        "cuda_memory": benchmark.get("atlas_peak_memory_mb", float("inf"))
        <= thresholds["cuda_atlas_peak_memory_mb_max"],
        "finite": bool(benchmark.get("finite", False)) == thresholds["all_outputs_finite"],
        "pytest": commands["pytest"]["returncode"] == 0,
        "ruff": commands["ruff"]["returncode"] == 0,
        "mypy": commands["mypy"]["returncode"] == 0,
    }
    passed = all(checks.values())
    result = {
        "protocol_id": protocol["protocol_id"],
        "decision": "passed_runtime_qualification" if passed else "failed_no_advance",
        "checks": checks,
        "equivalence": equivalence,
        "candidate_profile": profile,
        "cuda_benchmark": benchmark,
        "predecessor_report_tree_sha256": predecessor_hash_before,
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "s0_4_1_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, command in commands.items():
        (OUT / f"{name}.txt").write_text(command["output"], encoding="utf-8")
    failed = [name for name, passed_check in checks.items() if not passed_check]
    report = (
        "# S0.4.1 Cartesian-atlas runtime qualification\n\n"
        f"Decision: **{result['decision']}**.\n\n"
        "This performance-only successor preserves the frozen 4,032-candidate weighted prior. "
        "It does not rewrite the failed S0.4-v1 result and does not itself start S1a.\n\n"
        f"Failed checks: {failed if failed else 'none'}.\n\n"
        "```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    (OUT / "s0_4_1_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
