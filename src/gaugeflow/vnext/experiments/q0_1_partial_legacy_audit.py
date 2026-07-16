"""Versioned Q0.1 audit with corrected metrics and immutable missing evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml  # type: ignore[import-untyped]

from gaugeflow.coupling import fixed_lift_coupling
from gaugeflow.vnext.diagnostics import (
    adaptive_rk4,
    audit_representation_collisions,
    euler_integrate,
    exact_equivalence_risk,
    knn_local_target_dispersion,
    rk4_integrate,
)
from gaugeflow.vnext.experiments.gate_status import require_gate_status
from gaugeflow.vnext.experiments.q0_c0_audit import (
    ROOT,
    _endpoint_batch,
    _environment,
    _fixed_sources,
    _git_commit,
    _input_representation,
    _jacobian_rows,
    _parse_int_vector,
    _reduced,
    _sha256_bytes,
    _wrapped_representation,
    _write_manifest,
)
from gaugeflow.vnext.processes import translation_horizontal_basis


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def _replay_couplings(
    path: Path,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    lattice: torch.Tensor,
    tolerances: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, str]]:
    frozen_rows = list(csv.DictReader(path.open(encoding="utf-8")))
    graphs, sites = source.shape[:2]
    if len(frozen_rows) != graphs:
        raise ValueError("frozen coupling count does not match source count")
    assignments = []
    integer_lifts = []
    translations = []
    gaps = []
    endpoints = []
    velocities = []
    audit_rows = []
    for graph, frozen in enumerate(frozen_rows):
        if int(frozen["source"]) != graph:
            raise ValueError("frozen coupling sources are not canonical")
        recomputed = fixed_lift_coupling(
            source[graph],
            target[graph],
            lattice[graph],
            source_types=torch.arange(sites, device=source.device),
            target_types=torch.arange(sites, device=source.device),
        )
        frozen_assignment = torch.tensor([int(item) for item in frozen["assignment"].split(",")], device=source.device)
        frozen_lift = _parse_int_vector(frozen["integer_lift"], rows=sites).to(device=source.device, dtype=torch.long)
        frozen_translation = torch.tensor(
            [float(frozen["translation_x"]), float(frozen["translation_y"]), float(frozen["translation_z"])],
            device=source.device,
            dtype=source.dtype,
        )
        frozen_endpoint = target[graph, frozen_assignment] + frozen_lift.to(source.dtype) + frozen_translation
        frozen_velocity = frozen_endpoint - source[graph]
        assignment_equal = torch.equal(recomputed.assignment, frozen_assignment)
        lift_equal = torch.equal(recomputed.integer_lift, frozen_lift)
        translation_error = float((recomputed.translation - frozen_translation).abs().max())
        optimum_error = abs(float(recomputed.cost) - float(frozen["coupling_cost"]))
        second_error = abs(float(recomputed.second_cost) - float(frozen["second_coupling_cost"]))
        endpoint_error = float((recomputed.endpoint_lift - frozen_endpoint).abs().max())
        velocity_error = float((recomputed.velocity - frozen_velocity).abs().max())
        passed = bool(
            assignment_equal
            and lift_equal
            and translation_error <= float(tolerances["translation_abs_tolerance"])
            and optimum_error <= float(tolerances["optimum_cost_abs_tolerance"])
            and second_error <= float(tolerances["second_cost_abs_tolerance"])
            and endpoint_error <= float(tolerances["endpoint_abs_tolerance"])
            and velocity_error <= float(tolerances["velocity_abs_tolerance"])
        )
        audit_rows.append(
            {
                "source": graph,
                "assignment_exact": assignment_equal,
                "integer_lift_exact": lift_equal,
                "translation_max_abs_error": translation_error,
                "optimum_cost_abs_error": optimum_error,
                "second_cost_abs_error": second_error,
                "endpoint_max_abs_error": endpoint_error,
                "velocity_max_abs_error": velocity_error,
                "coupling_gap": float(recomputed.second_cost - recomputed.cost),
                "passed": passed,
            }
        )
        assignments.append(recomputed.assignment)
        integer_lifts.append(recomputed.integer_lift)
        translations.append(recomputed.translation)
        gaps.append(recomputed.second_cost - recomputed.cost)
        endpoints.append(recomputed.endpoint_lift)
        velocities.append(recomputed.velocity)
    assignment_tensor = torch.stack(assignments)
    lift_tensor = torch.stack(integer_lifts)
    translation_tensor = torch.stack(translations)
    gap_tensor = torch.stack(gaps)
    endpoint_tensor = torch.stack(endpoints)
    velocity_tensor = torch.stack(velocities)
    basis = translation_horizontal_basis(sites, dtype=torch.float64, device=source.device)
    gauge_fixed = torch.stack((_reduced(source.double(), basis), _reduced(endpoint_tensor.double(), basis)), dim=1)
    hashes = {
        "source_noise": _tensor_hash(source),
        "assignment": _tensor_hash(assignment_tensor),
        "integer_lift": _tensor_hash(lift_tensor),
        "translation": _tensor_hash(translation_tensor),
        "coupling_gap": _tensor_hash(gap_tensor),
        "gauge_fixed_state": _tensor_hash(gauge_fixed),
        "endpoint": _tensor_hash(endpoint_tensor),
        "velocity": _tensor_hash(velocity_tensor),
    }
    return endpoint_tensor, velocity_tensor, audit_rows, hashes


def _representation_rows(
    *,
    times: torch.Tensor,
    states: torch.Tensor,
    velocity: torch.Tensor,
    endpoint: torch.Tensor,
    lattice: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    basis = translation_horizontal_basis(states.shape[2], dtype=torch.float64)
    target = _reduced(velocity, basis)
    lift_representation = _reduced(endpoint, basis)
    dispersion_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    for time_index, time in enumerate(times):
        state = states[time_index]
        representations = {
            "universal_cover_X": _reduced(state, basis),
            "wrapped_torus": _wrapped_representation(state),
            "production_input": _input_representation(state, lattice, 16, 8.0),
        }
        for name, representation in representations.items():
            risk = exact_equivalence_risk(
                representation,
                target,
                absolute_tolerance=float(settings["exact_representation_abs_tolerance"]),
            )
            risk_rows.append(
                {
                    "time": float(time),
                    "representation": name,
                    **{
                        key: float(value) if isinstance(value, torch.Tensor) else value
                        for key, value in asdict(risk).items()
                    },
                }
            )
            collision, witnesses = audit_representation_collisions(
                representation,
                target,
                exact_absolute_tolerance=float(settings["exact_representation_abs_tolerance"]),
                near_quantile=float(settings["near_pair_quantile"]),
                alias_target_distance_min=float(settings["alias_target_distance_min"]),
                distance_floor=float(settings["local_distance_floor"]),
                lift_representation=lift_representation,
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
            witness_rows.extend(
                {"time": float(time), "representation": name, **asdict(witness)} for witness in witnesses
            )
            for neighbors in settings["knn_neighbors"]:
                dispersion = knn_local_target_dispersion(representation, target, neighbors=int(neighbors))
                dispersion_rows.append(
                    {
                        "time": float(time),
                        "representation": name,
                        "neighbors": int(neighbors),
                        "trace_dispersion": float(dispersion.trace_dispersion),
                        "target_trace_variance": float(dispersion.target_trace_variance),
                        "normalized_trace_dispersion": float(dispersion.normalized_trace_dispersion),
                    }
                )
    return dispersion_rows, risk_rows, collision_rows, witness_rows


def _corrected_solver_rows(
    source: torch.Tensor,
    endpoint: torch.Tensor,
    basis: torch.Tensor,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    source_reduced = _reduced(source, basis)
    endpoint_reduced = _reduced(endpoint, basis)
    end_time = 1.0 - float(settings["solver_terminal_epsilon"])

    def field(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (endpoint_reduced - value) / (1.0 - time)

    exact = (1.0 - end_time) * source_reduced + end_time * endpoint_reduced
    rows = []
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
                    "solution_error_rms": float((result.state - exact).square().mean().sqrt()),
                    "target_residual_rms": float((result.state - endpoint_reduced).square().mean().sqrt()),
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
            "solution_error_rms": float((adaptive.state - exact).square().mean().sqrt()),
            "target_residual_rms": float((adaptive.state - endpoint_reduced).square().mean().sqrt()),
            "accepted_steps": adaptive.accepted_steps,
            "rejected_steps": adaptive.rejected_steps,
            "evaluations": adaptive.evaluations,
        }
    )
    return rows


def run(config_path: Path, *, device: torch.device) -> Path:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config.get("gate") != "Q0.1" or config.get("status") != "pre_registered_not_run":
        raise ValueError("Q0.1 requires its pre-registered versioned config")
    predecessor = config["requires"]
    require_gate_status(ROOT / predecessor["original_q0_status"], gate="Q0", accepted=frozenset({"blocked"}))
    p0_status_path = ROOT / "reports" / "vnext_p0_release_v1" / "status.json"
    if not p0_status_path.is_file():
        raise RuntimeError("Q0.1 must run after the pre-registered P0 release audit")
    p0_status = json.loads(p0_status_path.read_text(encoding="utf-8"))
    p0_passed = bool(p0_status.get("all_passed"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Q0.1 source replay requires the registered CUDA environment")
    config_hash = _sha256_bytes(config_bytes)
    commit = _git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "runs" / "Q0.1" / f"{timestamp}_{config_hash[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "plots").mkdir()
    (output / "config.yaml").write_bytes(config_bytes)
    (output / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(_environment(device), indent=2) + "\n", encoding="utf-8")
    (output / "seeds.json").write_text(
        json.dumps({"source_noise": int(config["inputs"]["source_noise_seed"])}, indent=2) + "\n",
        encoding="utf-8",
    )
    count = int(config["inputs"]["source_count"])
    batch = _endpoint_batch(count, device)
    source = _fixed_sources(batch, int(config["inputs"]["source_noise_seed"])).reshape(count, 4, 3)
    target = batch.frac_coords.reshape(count, 4, 3)
    lattice = batch.lattice.reshape(count, 3, 3)
    endpoint, velocity, coupling_rows, coupling_hashes = _replay_couplings(
        ROOT / config["inputs"]["fixed_lift_rows"],
        source=source,
        target=target,
        lattice=lattice,
        tolerances=config["coupling_replay"],
    )
    coupling_passed = all(bool(row["passed"]) for row in coupling_rows)
    source_cpu = source.double().cpu()
    endpoint_cpu = endpoint.double().cpu()
    velocity_cpu = velocity.double().cpu()
    lattice_cpu = lattice.double().cpu()
    times = torch.linspace(0.0, 1.0, int(config["inputs"]["time_grid_count"]), dtype=torch.float64)
    states = source_cpu.unsqueeze(0) + times[:, None, None, None] * velocity_cpu.unsqueeze(0)
    dispersion, risks, collisions, witnesses = _representation_rows(
        times=times,
        states=states,
        velocity=velocity_cpu,
        endpoint=endpoint_cpu,
        lattice=lattice_cpu,
        settings=config["diagnostics"],
    )
    basis = translation_horizontal_basis(4, dtype=torch.float64)
    solver_rows = _corrected_solver_rows(source_cpu, endpoint_cpu, basis, config["diagnostics"])
    jacobian_rows = _jacobian_rows(times, basis.shape[1], config["diagnostics"]["variational_steps"])
    pd.DataFrame(dispersion).to_csv(output / "knn_local_target_dispersion.csv", index=False)
    pd.DataFrame(risks).to_csv(output / "exact_equivalence_risk.csv", index=False)
    pd.DataFrame(collisions).to_csv(output / "representation_collisions.csv", index=False)
    pd.DataFrame(witnesses).to_csv(output / "collision_witnesses.csv", index=False)
    pd.DataFrame(coupling_rows).to_csv(output / "coupling_replay.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(output / "solver_convergence.csv", index=False)
    pd.DataFrame(jacobian_rows).to_csv(output / "jacobian_diagnostics.csv", index=False)
    (output / "coupling_hashes.json").write_text(
        json.dumps(coupling_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = {
        "source_count": count,
        "time_grid_count": len(times),
        "coupling_replay_passed": coupling_passed,
        "p0_release_checks_passed": p0_passed,
        "historical_checkpoint_available": False,
        "historical_learned_field_classified": False,
        "scientific_pass_applicable": False,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        output / "metrics.csv", index=False
    )
    q1_authorized = coupling_passed and p0_passed
    status = {
        "gate": "Q0.1",
        "execution_status": "complete_partial_legacy",
        "scientific_verdict": "legacy_learned_field_unclassified",
        "successor_authorization": {"Q1v2": q1_authorized},
        "missing_immutable_artifacts": [
            "historical_model_checkpoint",
            "historical_optimizer_state",
        ],
        "completed_diagnostics": [
            "exact_coupling_replay",
            "knn_local_target_dispersion",
            "exact_equivalence_risk",
            "exact_and_near_collision_witnesses",
            "reference_affine_jacobian",
            "reference_solver_convergence",
        ],
        "config_hash": config_hash,
        "git_commit": commit,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    failures = [] if q1_authorized else [{"kind": "q1_authorization_withheld", "p0": p0_passed}]
    (output / "failures.jsonl").write_text("".join(json.dumps(item) + "\n" for item in failures), encoding="utf-8")
    endpoint_risk = [row for row in risks if row["time"] == 1.0 and row["representation"] == "production_input"][0]
    report = (
        "# Q0.1 corrected partial-legacy audit\n\n"
        "Execution status: `complete_partial_legacy`. Scientific verdict: "
        "`legacy_learned_field_unclassified`. This is not a scientific pass.\n\n"
        f"Exact coupling replay passed: `{coupling_passed}`. P0 release passed: `{p0_passed}`. "
        f"Q1v2 authorized: `{q1_authorized}`.\n\n"
        "kNN values are reported only as local target dispersion. Exact finite-sample risk is computed over "
        "fixed-tolerance equivalence classes. At the production-input endpoint, exact collision risk has "
        f"normalized value `{endpoint_risk['normalized_trace_risk']}` across "
        f"`{endpoint_risk['exact_collision_count']}` colliding pairs.\n\n"
        "Solver `solution_error_rms` is relative to the analytic state at `1-epsilon`; "
        "`target_residual_rms` is relative to the limiting point target.\n\n"
        "The historical learned field remains unclassified because its immutable checkpoint does not exist. "
        "No reenactment was trained.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "checkpoints" / ".gitkeep").write_text("", encoding="utf-8")
    (output / "plots" / ".gitkeep").write_text("", encoding="utf-8")
    _write_manifest(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gates/q0_1_partial_legacy_audit.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(run(config_path, device=torch.device(args.device)))


if __name__ == "__main__":
    main()
