"""Run the no-training S0.2 scalability and symmetry-chart qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.production.harmonic_gaugeflow import (
    GeometryHarmonicQueries,
    HarmonicGaugeFlowConditioner,
)
from gaugeflow.production.s0_audit import _worktree_head
from gaugeflow.production.space_group_router import compatibility_record
from gaugeflow.production.wrapped_coordinates import (
    AdaptiveWrappedQuotient,
    ScalableWrappedQuotient,
)
from gaugeflow.tensor import piezo_from_irreps, piezo_to_irreps, rotate_rank3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(repo: Path, arguments: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        arguments,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty audit table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _symmetry_rows() -> list[dict[str, Any]]:
    zero_rank = {
        "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-31m", "-3m1", "-3m",
        "6/m", "6/mmm", "m-3", "432", "m-3m",
    }
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(120)
    for number in range(1, 231):
        record = compatibility_record(number)
        fractional = record.fractional_operations
        cartesian = record.operations
        fractional_closure = torch.linalg.matrix_norm(
            (fractional[:, None] @ fractional[None, :])[:, :, None]
            - fractional[None, None],
            dim=(-2, -1),
        ).amin(dim=-1).max()
        cartesian_closure = torch.linalg.matrix_norm(
            (cartesian[:, None] @ cartesian[None, :])[:, :, None]
            - cartesian[None, None],
            dim=(-2, -1),
        ).amin(dim=-1).max()
        identity = torch.eye(3, dtype=torch.float64)
        orthogonality = torch.linalg.matrix_norm(
            cartesian.transpose(-1, -2) @ cartesian - identity, dim=(-2, -1)
        ).max()
        determinant = (torch.linalg.det(fractional).abs() - 1.0).abs().max()
        chart = record.metric_chart
        coordinates = torch.randn(chart.shape_dimension, generator=generator, dtype=torch.float64)
        log_shape = coordinates @ chart.invariant_log_shape_basis.T
        metric = chart.metric(torch.tensor(0.0, dtype=torch.float64), log_shape)
        metric_residual = chart.invariance_residual(metric).max()
        reynolds_error = torch.linalg.matrix_norm(
            record.reynolds_irrep @ record.reynolds_irrep - record.reynolds_irrep
        )
        expected_zero = record.point_group in zero_rank
        rows.append(
            {
                "space_group": number,
                "symbol": record.symbol,
                "point_group": record.point_group,
                "operation_count": len(fractional),
                "shape_dimension": chart.shape_dimension,
                "fractional_closure_error": float(fractional_closure),
                "cartesian_closure_error": float(cartesian_closure),
                "cartesian_orthogonality_error": float(orthogonality),
                "fractional_determinant_error": float(determinant),
                "fractional_metric_invariance_error": float(metric_residual),
                "reynolds_idempotence_error": float(reynolds_error),
                "compatible_rank": record.compatible_rank,
                "expected_rank_rule_passed": (
                    record.compatible_rank == 0 if expected_zero else record.compatible_rank > 0
                ),
            }
        )
    return rows


def _wrapped_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lattice64 = torch.tensor(
        [[2.2, 0.0, 0.0], [0.3, 2.5, 0.0], [0.2, 0.4, 2.8]], dtype=torch.float64
    )
    for case in config["small_site_exact_cases"]:
        generator = torch.Generator().manual_seed(case["seed"])
        sites, sigma = case["sites"], case["sigma"]
        current = torch.rand((sites, 3), generator=generator, dtype=torch.float64)
        clean = torch.rand((sites, 3), generator=generator, dtype=torch.float64)
        exact = AdaptiveWrappedQuotient(
            absolute_tail_tolerance=1e-10,
            relative_tail_tolerance=1e-10,
            max_images=2_000_000,
        ).evaluate(current, clean, lattice64, sigma)
        scalable = ScalableWrappedQuotient().evaluate(current, clean, lattice64, sigma)
        score_scale = torch.linalg.vector_norm(exact.fractional_score) + 1e-8
        rows.append(
            {
                "case": f"exact_m{sites}_sigma{sigma}",
                "sites": sites,
                "sigma": sigma,
                "lattice": "triclinic_reference",
                "status": "complete",
                "absolute_log_density_error": float(
                    (exact.log_unnormalized_density - scalable.log_unnormalized_density).abs()
                ),
                "relative_score_error": float(
                    torch.linalg.vector_norm(exact.fractional_score - scalable.fractional_score)
                    / score_scale
                ),
                "qmc_samples": scalable.qmc_samples,
                "kernel_representation": scalable.kernel_representation,
                "kernel_terms": scalable.kernel_terms,
                "kernel_tail_upper_bound": scalable.kernel_tail_upper_bound,
                "qmc_log_increment": scalable.qmc_log_increment,
                "qmc_relative_score_increment": scalable.qmc_relative_score_increment,
                "runtime_seconds": None,
                "peak_cuda_memory_bytes": 0,
            }
        )
    stress = config["scalability_stress"]
    if not torch.cuda.is_available():
        raise RuntimeError("S0.2 M=20 scalability audit requires the pinned CUDA environment")
    generator = torch.Generator().manual_seed(stress["seed"])
    current = torch.rand((stress["sites"], 3), generator=generator).cuda()
    clean = torch.rand((stress["sites"], 3), generator=generator).cuda()
    kernel = ScalableWrappedQuotient(
        kernel_tail_tolerance=stress["kernel_tail_tolerance"],
        qmc_log_tolerance=config["thresholds"]["stress_qmc_log_refinement_max"],
        qmc_relative_score_tolerance=config["thresholds"][
            "stress_qmc_relative_score_refinement_max"
        ],
        chunk_size=1024,
    )
    for lattice_name, lattice_values in stress["lattices"].items():
        lattice = torch.tensor(lattice_values, dtype=torch.float32, device="cuda")
        for sigma in stress["sigmas"]:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = kernel.evaluate(current, clean, lattice, sigma)
            torch.cuda.synchronize()
            rows.append(
                {
                    "case": f"stress_{lattice_name}_m{stress['sites']}_sigma{sigma}",
                    "sites": stress["sites"],
                    "sigma": sigma,
                    "lattice": lattice_name,
                    "status": "complete",
                    "absolute_log_density_error": None,
                    "relative_score_error": None,
                    "qmc_samples": result.qmc_samples,
                    "kernel_representation": result.kernel_representation,
                    "kernel_terms": result.kernel_terms,
                    "kernel_tail_upper_bound": result.kernel_tail_upper_bound,
                    "qmc_log_increment": result.qmc_log_increment,
                    "qmc_relative_score_increment": result.qmc_relative_score_increment,
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
    return rows


def _rotation() -> torch.Tensor:
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = 0.731
    cross = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=torch.float64,
    )
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * cross + (
        1.0 - math.cos(angle)
    ) * (cross @ cross)


def _qmc_object(grid_size: int) -> dict[str, torch.Tensor | float]:
    torch.manual_seed(107)
    conditioner = HarmonicGaugeFlowConditioner(24, grid_size=grid_size, query_channels=2).double()
    condition = torch.randn((1, 18), dtype=torch.float64)
    first = torch.randn((1, 2, 3), dtype=torch.float64, requires_grad=True)
    second = torch.randn((1, 2, 5), dtype=torch.float64, requires_grad=True)
    third = torch.randn((1, 2, 7), dtype=torch.float64, requires_grad=True)
    queries = GeometryHarmonicQueries(first, second, third)
    directions = torch.nn.functional.normalize(torch.randn((6, 3), dtype=torch.float64), dim=-1)
    edge_graph = torch.zeros(6, dtype=torch.long)
    present = torch.ones((1, 1), dtype=torch.bool)
    time_value = torch.tensor([0.2], dtype=torch.float64)
    output = conditioner(condition, present, directions, edge_graph, queries, time_value)
    gradient = torch.autograd.grad(output.graph_condition.square().sum(), (first, second, third))
    moment = torch.einsum("bf,fij->bij", output.posterior, conditioner.rotations)
    representative = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), _rotation()))
    representative_output = conditioner(
        representative, present, directions, edge_graph, queries, time_value
    )
    return {
        "aligned": output.aligned_irreps.detach(),
        "posterior_moment": moment.detach(),
        "condition": output.graph_condition.detach(),
        "gradient": torch.cat([value.flatten() for value in gradient]).detach(),
        "representative_condition_error": float(
            torch.linalg.vector_norm(output.graph_condition - representative_output.graph_condition)
        ),
        "representative_aligned_error": float(
            torch.linalg.vector_norm(output.aligned_irreps - representative_output.aligned_irreps)
        ),
        "entropy": float(output.entropy),
    }


def _qmc_rows(grid_sizes: list[int]) -> list[dict[str, Any]]:
    values = {size: _qmc_object(size) for size in grid_sizes}
    reference = values[grid_sizes[-1]]
    rows = []
    for size in grid_sizes:
        value = values[size]
        rows.append(
            {
                "K": size,
                "aligned_irreps_error_to_Kmax": float(
                    torch.linalg.vector_norm(value["aligned"] - reference["aligned"])
                ),
                "posterior_moment_error_to_Kmax": float(
                    torch.linalg.vector_norm(
                        value["posterior_moment"] - reference["posterior_moment"]
                    )
                ),
                "condition_token_error_to_Kmax": float(
                    torch.linalg.vector_norm(value["condition"] - reference["condition"])
                ),
                "condition_gradient_error_to_Kmax": float(
                    torch.linalg.vector_norm(value["gradient"] - reference["gradient"])
                ),
                "random_representative_condition_error": value[
                    "representative_condition_error"
                ],
                "random_representative_aligned_error": value["representative_aligned_error"],
                "posterior_entropy": value["entropy"],
            }
        )
    return rows


def run_audit(repo: Path, config_path: Path, output: Path, tested_commit: str) -> None:
    if tested_commit != _worktree_head(repo):
        raise RuntimeError("tested commit does not match worktree HEAD")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S0.2 evidence: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    commands = {
        "targeted_tests": [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_paper_s0_production.py",
            "tests/test_paper_s0_2_scalability_symmetry.py",
        ],
        "full_tests": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "."],
        "mypy": [sys.executable, "-m", "mypy", "src"],
    }
    checks: dict[str, bool] = {}
    for name, command in commands.items():
        checks[name], command_output = _run(repo, command)
        (output / f"{name}.txt").write_text(command_output, encoding="utf-8")
    symmetry = _symmetry_rows()
    wrapped = _wrapped_rows(config)
    qmc = _qmc_rows(config["finite_qmc_grid_sizes"])
    _write_csv(output / "symmetry_chart_audit.csv", symmetry)
    _write_csv(output / "wrapped_quotient_audit.csv", wrapped)
    _write_csv(output / "finite_qmc_convergence.csv", qmc)
    thresholds = config["thresholds"]
    checks["all_230_space_groups"] = len(symmetry) == 230
    checks["fractional_metric_invariance"] = max(
        row["fractional_metric_invariance_error"] for row in symmetry
    ) <= thresholds["fractional_metric_invariance_max"]
    checks["fractional_group_closure"] = max(
        row["fractional_closure_error"] for row in symmetry
    ) <= thresholds["fractional_group_closure_max"]
    checks["cartesian_group_closure"] = max(
        row["cartesian_closure_error"] for row in symmetry
    ) <= thresholds["cartesian_group_closure_max"]
    checks["cartesian_orthogonality"] = max(
        row["cartesian_orthogonality_error"] for row in symmetry
    ) <= thresholds["cartesian_orthogonality_max"]
    checks["reynolds_idempotence"] = max(
        row["reynolds_idempotence_error"] for row in symmetry
    ) <= thresholds["reynolds_idempotence_max"]
    checks["piezoelectric_rank_rules"] = all(
        row["expected_rank_rule_passed"] for row in symmetry
    )
    exact_rows = [row for row in wrapped if row["case"].startswith("exact_")]
    stress_rows = [row for row in wrapped if row["case"].startswith("stress_")]
    checks["small_site_log_density"] = all(
        row["absolute_log_density_error"]
        <= thresholds["small_site_absolute_log_density_error_max"]
        for row in exact_rows
    )
    checks["small_site_score"] = all(
        row["relative_score_error"] <= thresholds["small_site_relative_score_error_max"]
        for row in exact_rows
    )
    checks["m20_resource_stress"] = len(stress_rows) == 4 and all(
        row["status"] == "complete" for row in stress_rows
    )
    checks["finite_qmc_refinement"] = all(
        qmc[1][name] < qmc[0][name]
        for name in (
            "aligned_irreps_error_to_Kmax",
            "posterior_moment_error_to_Kmax",
            "condition_token_error_to_Kmax",
            "condition_gradient_error_to_Kmax",
        )
    )
    checks["random_representative_refinement"] = (
        qmc[-1]["random_representative_condition_error"]
        < qmc[0]["random_representative_condition_error"]
        and qmc[-1]["random_representative_aligned_error"]
        < qmc[0]["random_representative_aligned_error"]
    )
    all_passed = all(checks.values())
    status = {
        "schema": 1,
        "gate": "S0.2",
        "name": config["name"],
        "status": "passed" if all_passed else "failed",
        "all_passed": all_passed,
        "checks": checks,
        "tested_commit": tested_commit,
        "evidence_commit": None,
        "config_sha256": _sha256(config_path),
        "design_sha256": config["design_sha256"],
        "historical_s0_1_unchanged": True,
        "successor_authorization": {"S1a_preparation": all_passed, "S1_started": False},
        "real_tensor_or_physical_validation_allowed": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    rows = "\n".join(f"| {name} | {value} |" for name, value in checks.items())
    report = (
        "# Revised-paper S0.2 scalability and symmetry-chart audit\n\n"
        f"Status: **{'passed' if all_passed else 'failed'}**. Tested commit: `{tested_commit}`.\n\n"
        "This is a no-training mathematical, symmetry-chart, scalability, and interface audit. "
        "S0.1 remains unchanged. No S1 training, real tensor, oracle, MLIP screening, relaxation, "
        "DFT, or DFPT was run.\n\n"
        "| Check | Passed |\n|---|---:|\n"
        f"{rows}\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest_paths = sorted(path for path in output.iterdir() if path.name != "manifest.sha256")
    manifest = "\n".join(f"{_sha256(path)}  {path.name}" for path in manifest_paths) + "\n"
    (output / "manifest.sha256").write_text(manifest, encoding="utf-8")
    if not all_passed:
        raise RuntimeError("S0.2 audit failed; inspect the versioned evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_s0_2_scalability_symmetry_chart_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paper_s0_2_scalability_symmetry_chart_v1"),
    )
    parser.add_argument("--tested-commit", required=True)
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    config = arguments.config if arguments.config.is_absolute() else repo / arguments.config
    output = arguments.output if arguments.output.is_absolute() else repo / arguments.output
    run_audit(repo, config, output, arguments.tested_commit)


if __name__ == "__main__":
    main()
