"""Run the checkpoint-preserving Q0 diagnosis of the frozen P5-C0 path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml  # type: ignore[import-untyped]
from torch_geometric.data import Batch, Data

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.geometry import GaussianRadialBasis, periodic_closest_image_edges
from gaugeflow.vnext.diagnostics import (
    adaptive_rk4,
    analytic_endpoint_jacobians,
    audit_representation_collisions,
    euler_integrate,
    knn_conditional_variance,
    rk4_integrate,
    variational_flow_jacobian,
)
from gaugeflow.vnext.legacy import load_manifest, verify_manifest
from gaugeflow.vnext.processes import translation_horizontal_basis

ROOT = Path(__file__).resolve().parents[4]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _endpoint_batch(count: int, device: torch.device) -> Batch:
    endpoint = Data(
        atom_types=torch.tensor((5, 7, 14, 32), dtype=torch.long, device=device),
        frac_coords=torch.tensor(
            ((0.06, 0.11, 0.19), (0.34, 0.22, 0.31), (0.72, 0.48, 0.41), (0.21, 0.79, 0.67)),
            dtype=torch.float32,
            device=device,
        ),
        lattice=torch.tensor(
            ((3.9, 0.2, 0.1), (0.3, 4.3, 0.4), (0.1, 0.4, 5.1)),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        num_nodes=4,
    )
    return Batch.from_data_list([endpoint.clone() for _ in range(count)]).to(device)


def _fixed_sources(batch: Batch, seed: int) -> torch.Tensor:
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    torch.manual_seed(seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(seed)
    return matcher.random_state(batch).frac_coords


def _parse_int_vector(value: str, *, rows: int) -> torch.Tensor:
    numbers = [int(item) for item in value.split(",")]
    if len(numbers) != rows * 3:
        raise ValueError("frozen integer-lift row has the wrong length")
    return torch.tensor(numbers, dtype=torch.float64).reshape(rows, 3)


def _load_frozen_couplings(
    path: Path,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    lattice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    graphs, sites = source.shape[:2]
    if len(rows) != graphs:
        raise ValueError("frozen coupling count does not match the registered sources")
    endpoint = torch.empty_like(source, dtype=torch.float64, device="cpu")
    maximum_cost_error = 0.0
    for graph, row in enumerate(rows):
        if int(row["source"]) != graph:
            raise ValueError("frozen coupling sources are not in canonical order")
        assignment = torch.tensor([int(item) for item in row["assignment"].split(",")], dtype=torch.long)
        if assignment.shape != (sites,) or torch.unique(assignment).numel() != sites:
            raise ValueError("frozen assignment is not a permutation")
        integer_lift = _parse_int_vector(row["integer_lift"], rows=sites)
        translation = torch.tensor(
            [float(row["translation_x"]), float(row["translation_y"]), float(row["translation_z"])],
            dtype=torch.float64,
        )
        endpoint[graph] = target[graph, assignment].double().cpu() + integer_lift + translation
        residual = source[graph].double().cpu() - endpoint[graph]
        centered = residual - residual.mean(dim=0, keepdim=True)
        cost = (centered @ lattice[graph].double().cpu()).square().sum()
        maximum_cost_error = max(maximum_cost_error, abs(float(cost) - float(row["coupling_cost"])))
    if maximum_cost_error > 2.0e-4:
        raise RuntimeError(
            "regenerated source noise does not reproduce the frozen coupling costs; "
            f"maximum absolute error={maximum_cost_error:.6g}"
        )
    source_cpu = source.double().cpu()
    return endpoint, endpoint - source_cpu, maximum_cost_error


def _reduced(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1) @ basis


def _wrapped_representation(value: torch.Tensor) -> torch.Tensor:
    relative = value[:, 1:, :] - value[:, :1, :]
    angle = 2.0 * torch.pi * relative
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(value.shape[0], -1)


def _input_representation(value: torch.Tensor, lattice: torch.Tensor, rbf_count: int, cutoff: float) -> torch.Tensor:
    rbf = GaussianRadialBasis(rbf_count, cutoff).to(dtype=torch.float64)
    rows = []
    for graph in range(value.shape[0]):
        graph_batch = torch.zeros(value.shape[1], dtype=torch.long)
        geometry = periodic_closest_image_edges(value[graph], lattice[graph : graph + 1], graph_batch)
        features = torch.cat((geometry.direction, rbf(geometry.distance)), dim=-1)
        rows.append(features.reshape(-1))
    return torch.stack(rows)


def _conditional_rows(
    times: torch.Tensor,
    states: torch.Tensor,
    velocity: torch.Tensor,
    lattice: torch.Tensor,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    basis = translation_horizontal_basis(states.shape[2], dtype=torch.float64)
    target = _reduced(velocity, basis)
    variance_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    settings = config["diagnostics"]
    for time_index, time in enumerate(times):
        state = states[time_index]
        representations = {
            "universal_cover_X": _reduced(state, basis),
            "wrapped_torus": _wrapped_representation(state),
            "production_input": _input_representation(state, lattice, 16, 8.0),
        }
        for name, representation in representations.items():
            collision = audit_representation_collisions(
                representation,
                target,
                near_quantile=float(settings["near_collision_quantile"]),
                target_ratio_min=float(settings["near_collision_target_ratio_min"]),
                distance_floor=float(settings["local_lipschitz_distance_floor"]),
            )
            collision_rows.append(
                {
                    "time": float(time),
                    "representation": name,
                    **{
                        key: float(value) if isinstance(value, torch.Tensor) else value
                        for key, value in asdict(collision).items()
                    },
                }
            )
            for neighbors in settings["knn_neighbors"]:
                estimate = knn_conditional_variance(representation, target, neighbors=int(neighbors))
                variance_rows.append(
                    {
                        "time": float(time),
                        "representation": name,
                        "neighbors": int(neighbors),
                        "trace_variance": float(estimate.trace_variance),
                        "target_trace_variance": float(estimate.target_trace_variance),
                        "normalized_trace_variance": float(estimate.normalized_trace_variance),
                    }
                )
    return variance_rows, collision_rows


def _jacobian_rows(times: torch.Tensor, dimension: int, steps: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time in times:
        result = analytic_endpoint_jacobians(dimension, time)
        terminal = float(time) == 1.0
        row: dict[str, Any] = {
            "time": float(time),
            "dimension": dimension,
            "vector_jacobian_spectral_norm": None
            if result.vector_jacobian is None
            else float(torch.linalg.matrix_norm(result.vector_jacobian, ord=2)),
            "flow_singular_value_min": float(result.singular_values.min()),
            "flow_singular_value_max": float(result.singular_values.max()),
            "flow_log_abs_det": float(result.log_abs_det),
            "endpoint_singular": terminal,
        }
        rows.append(row)
    end_time = 1.0 - 1.0e-6
    exact = analytic_endpoint_jacobians(dimension, torch.tensor(end_time, dtype=torch.float64)).flow_jacobian
    for count in steps:
        integrated = variational_flow_jacobian(
            lambda time: -torch.eye(dimension, dtype=torch.float64) / (1.0 - time),
            dimension=dimension,
            end_time=end_time,
            steps=int(count),
        )
        rows.append(
            {
                "time": end_time,
                "dimension": dimension,
                "variational_steps": int(count),
                "variational_flow_jacobian_max_error": float((integrated - exact).abs().max()),
            }
        )
    return rows


def _solver_rows(
    source: torch.Tensor,
    endpoint: torch.Tensor,
    basis: torch.Tensor,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    source_reduced = _reduced(source, basis)
    endpoint_reduced = _reduced(endpoint, basis)
    end_time = 1.0 - 1.0e-6

    def field(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (endpoint_reduced - value) / (1.0 - time)

    expected = (1.0 - end_time) * source_reduced + end_time * endpoint_reduced
    rows: list[dict[str, Any]] = []
    for method, counts, solver in (
        ("euler", settings["euler_steps"], euler_integrate),
        ("rk4", settings["rk4_steps"], rk4_integrate),
    ):
        for count in counts:
            result = solver(field, source_reduced, start=0.0, end=end_time, steps=int(count))
            rows.append(
                {
                    "method": method,
                    "steps": int(count),
                    "integration_end_time": end_time,
                    "endpoint_rms": float((result.state - expected).square().mean().sqrt()),
                    "accepted_steps": result.accepted_steps,
                    "rejected_steps": result.rejected_steps,
                    "evaluations": result.evaluations,
                }
            )
    adaptive = adaptive_rk4(
        field,
        source_reduced,
        start=0.0,
        end=end_time,
        rtol=float(settings["adaptive_rtol"]),
        atol=float(settings["adaptive_atol"]),
    )
    rows.append(
        {
            "method": "adaptive_rk4_step_doubling",
            "steps": adaptive.accepted_steps,
            "integration_end_time": end_time,
            "endpoint_rms": float((adaptive.state - expected).square().mean().sqrt()),
            "accepted_steps": adaptive.accepted_steps,
            "rejected_steps": adaptive.rejected_steps,
            "evaluations": adaptive.evaluations,
        }
    )
    return rows


def _environment(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else platform.processor(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_manifest(directory: Path) -> None:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "manifest.sha256")
    lines = [f"{_sha256_file(path)}  {path.relative_to(directory).as_posix()}" for path in files]
    (directory / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, *, device: torch.device) -> Path:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config.get("gate") != "Q0" or config.get("status") != "pre_registered":
        raise ValueError("Q0 requires its frozen pre-registered configuration")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("exact frozen-source regeneration requires the registered CUDA environment")
    legacy_manifest_path = ROOT / config["legacy_evidence_manifest"]
    legacy_failures = verify_manifest(ROOT, load_manifest(legacy_manifest_path))
    if legacy_failures:
        raise RuntimeError("legacy evidence changed: " + "; ".join(legacy_failures))
    config_hash = _sha256_bytes(config_bytes)
    commit = _git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "runs" / "Q0" / f"{timestamp}_{config_hash[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "plots").mkdir()
    (output / "config.yaml").write_bytes(config_bytes)
    (output / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(_environment(device), indent=2) + "\n", encoding="utf-8")
    seeds = {"source_noise": int(config["inputs"]["source_noise_seed"]), "bootstrap": None}
    (output / "seeds.json").write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")

    count = int(config["inputs"]["source_count"])
    batch = _endpoint_batch(count, device)
    source = _fixed_sources(batch, int(config["inputs"]["source_noise_seed"])).reshape(count, 4, 3)
    target = batch.frac_coords.reshape(count, 4, 3)
    lattice = batch.lattice.reshape(count, 3, 3)
    endpoint, velocity, coupling_cost_error = _load_frozen_couplings(
        ROOT / config["inputs"]["fixed_lift_rows"], source=source, target=target, lattice=lattice
    )
    source_cpu = source.double().cpu()
    lattice_cpu = lattice.double().cpu()
    times = torch.linspace(0.0, 1.0, int(config["inputs"]["time_grid_count"]), dtype=torch.float64)
    states = source_cpu.unsqueeze(0) + times[:, None, None, None] * velocity.unsqueeze(0)
    variance_rows, collision_rows = _conditional_rows(times, states, velocity, lattice_cpu, config)
    basis = translation_horizontal_basis(4, dtype=torch.float64)
    jacobian_rows = _jacobian_rows(times, basis.shape[1], config["diagnostics"]["flow_jacobian"]["variational_steps"])
    solver_rows = _solver_rows(source_cpu, endpoint, basis, config["diagnostics"]["solver_audit"])

    checkpoint = config["inputs"]["legacy_checkpoint"]
    checkpoint_path = checkpoint.get("path")
    checkpoint_available = bool(checkpoint_path) and (ROOT / checkpoint_path).is_file()
    unavailable = []
    if not checkpoint_available:
        unavailable = [
            "R_embed(t)",
            "learned reduced vector-field Jacobian",
            "learned flow Jacobian and log determinant",
            "legacy checkpoint Euler/RK4/adaptive rollout convergence",
        ]
    status = "complete" if not unavailable else "blocked"
    failed_rules = [] if not unavailable else ["required frozen P5-C0 checkpoint is missing; Q0 forbids retraining"]
    pd.DataFrame(variance_rows).to_csv(output / "conditional_variance.csv", index=False)
    pd.DataFrame(collision_rows).to_csv(output / "representation_collisions.csv", index=False)
    pd.DataFrame(jacobian_rows).to_csv(output / "jacobian_diagnostics.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(output / "solver_convergence.csv", index=False)
    metrics = {
        "source_count": count,
        "time_grid_count": len(times),
        "frozen_coupling_cost_reproduction_max_abs_error": coupling_cost_error,
        "checkpoint_available": checkpoint_available,
        "checkpoint_independent_diagnostics_complete": True,
        "unavailable_diagnostics": unavailable,
        "scientific_pass_applicable": False,
        "root_cause_classification": (
            "blocked_before_complete_classification" if unavailable else "see_diagnostic_tables"
        ),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary_rows = [
        {"metric": "source_count", "value": count},
        {"metric": "time_grid_count", "value": len(times)},
        {"metric": "coupling_cost_reproduction_max_abs_error", "value": coupling_cost_error},
        {"metric": "checkpoint_available", "value": checkpoint_available},
        {"metric": "status", "value": status},
    ]
    pd.DataFrame(summary_rows).to_csv(output / "metrics.csv", index=False)
    status_payload = {
        "gate": "Q0",
        "status": status,
        "passed_all_seeds": False,
        "failed_rules": failed_rules,
        "config_hash": config_hash,
        "git_commit": commit,
        "Q1_authorized": status == "complete",
    }
    (output / "status.json").write_text(json.dumps(status_payload, indent=2) + "\n", encoding="utf-8")
    failures = (
        [
            {
                "kind": "missing_required_legacy_checkpoint",
                "path": checkpoint_path,
                "action": checkpoint["missing_action"],
                "prohibited_recovery": "retraining",
            }
        ]
        if unavailable
        else []
    )
    (output / "failures.jsonl").write_text("".join(json.dumps(item) + "\n" for item in failures), encoding="utf-8")
    report = (
        "# Q0 legacy C0 no-training root-cause audit\n\n"
        f"Status: **{status}**. Q0 has no scientific pass label. Q1 authorized: **{status == 'complete'}**.\n\n"
        "The frozen 64 CUDA source noises and 64 fixed-lift couplings were reconstructed without training. "
        f"Their maximum coupling-cost reproduction error is `{coupling_cost_error:.6g}`. "
        "Checkpoint-independent conditional-variance, representation-collision, analytic Jacobian, and "
        "analytic singular-field solver tables are complete.\n\n"
        "The historical P5-C0/D0.4--D0.8 runners did not persist model weights. Consequently the learned "
        "embedding, learned vector/flow Jacobians, and old-checkpoint solver convergence cannot be measured. "
        "Q0 explicitly prohibits retraining, so this run is blocked rather than silently substituting a new model.\n\n"
        "## Missing required diagnostics\n\n"
        + "".join(f"- {item}\n" for item in unavailable)
        + "\n## Decision\n\nStop at Q0. Do not run Q1 or any later gate. "
        "Historical reports and thresholds remain unchanged.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "checkpoints" / ".gitkeep").write_text("", encoding="utf-8")
    (output / "plots" / ".gitkeep").write_text("", encoding="utf-8")
    _write_manifest(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gates/q0_c0_audit.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = run(config_path, device=torch.device(args.device))
    print(output)


if __name__ == "__main__":
    main()
