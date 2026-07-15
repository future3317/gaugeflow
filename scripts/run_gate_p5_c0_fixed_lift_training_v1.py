"""Run the authorized P5-C0 fixed-lift universal-cover qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from gaugeflow.coupling import fixed_lift_coupling, remove_graphwise_translation
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_sources(matcher, batch, *, seed: int):
    """Solve one immutable endpoint lift for every source before training."""
    torch.manual_seed(seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(seed)
    random = matcher.random_state(batch)
    target = matcher.target_state(batch)
    endpoint_lift = torch.empty_like(batch.frac_coords)
    velocity = torch.empty_like(batch.frac_coords)
    rows = []
    for graph in range(batch.num_graphs):
        nodes = torch.nonzero(batch.batch == graph, as_tuple=False).flatten()
        coupling = fixed_lift_coupling(
            random.frac_coords[nodes], target.frac_coords[nodes], batch.lattice[graph],
            source_types=batch.atom_types[nodes], target_types=batch.atom_types[nodes],
        )
        endpoint_lift[nodes] = coupling.endpoint_lift
        velocity[nodes] = coupling.velocity
        rows.append({
            "source": graph,
            "assignment": ",".join(str(int(value)) for value in coupling.assignment.tolist()),
            "integer_lift": ",".join(str(int(value)) for value in coupling.integer_lift.flatten().tolist()),
            "translation_x": float(coupling.translation[0]),
            "translation_y": float(coupling.translation[1]),
            "translation_z": float(coupling.translation[2]),
            "coupling_cost": float(coupling.cost),
            "second_coupling_cost": float(coupling.second_cost),
            "coupling_margin": float(coupling.second_cost - coupling.cost),
        })
    source = CrystalFlowState(target.type_state, random.frac_coords, target.lattice_log)
    return source, endpoint_lift, velocity, pd.DataFrame(rows)


def universal_interpolant(source: CrystalFlowState, velocity: torch.Tensor, batch, time: torch.Tensor) -> CrystalFlowState:
    """Linear path on the frozen universal cover; deliberately never wraps."""
    return CrystalFlowState(source.type_state, source.frac_coords + time[batch.batch, None] * velocity, source.lattice_log)


def predicted_velocity(model, matcher, state, batch, time):
    return matcher._coordinate_velocity(model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)[1], batch)


def quotient_rms(value: torch.Tensor, batch) -> torch.Tensor:
    return remove_graphwise_translation(value, batch.batch, batch.num_graphs).square().mean().sqrt()


def metrics_at_time(model, matcher, batch, source, endpoint_lift, target_velocity, time):
    state = universal_interpolant(source, target_velocity, batch, time)
    predicted = predicted_velocity(model, matcher, state, batch, time)
    velocity_mse = (predicted - target_velocity).square().mean()
    remaining = (1.0 - time[batch.batch])[:, None]
    map_mse = (remaining * (predicted - target_velocity)).square().mean()
    predicted_endpoint = state.frac_coords + remaining * predicted
    teacher_rms = quotient_rms(predicted_endpoint - endpoint_lift, batch)
    return velocity_mse, map_mse, teacher_rms


@torch.no_grad()
def sample_universal_cover(model, matcher, batch, initial: CrystalFlowState, steps: int) -> CrystalFlowState:
    """Euler integration without a runtime torus-branch fallback."""
    state = CrystalFlowState(initial.type_state, initial.frac_coords.clone(), initial.lattice_log)
    for step in range(steps):
        time = torch.full((batch.num_graphs,), step / steps, device=batch.frac_coords.device)
        velocity = predicted_velocity(model, matcher, state, batch, time)
        state = CrystalFlowState(state.type_state, state.frac_coords + velocity / steps, state.lattice_log)
    return state


def periodic_terminal_rms(value: torch.Tensor, endpoint_lift: torch.Tensor, batch) -> torch.Tensor:
    displacement = torus_logmap(wrap01(value), wrap01(endpoint_lift))
    return quotient_rms(displacement, batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_c0_periodic_path_fixed_lift_v1.json"))
    parser.add_argument("--audit-manifest", type=Path, default=Path("reports/gate_p5_c0_periodic_path_fixed_lift_v1/audit/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_c0_periodic_path_fixed_lift_v1/training"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    audit_path = ROOT / args.audit_manifest
    output = ROOT / args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if protocol["status"] != "pre_registered_not_started" or not audit["training_authorized"] or output.exists():
        raise ValueError("P5-C0 fixed-lift training requires its positive frozen audit and a fresh output")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-C0 reported training requires CUDA")
    training = protocol["training_if_triggered"]
    evaluation = protocol["evaluation_if_triggered"]
    # Importing the frozen D0.4 builder does not import its dynamic path.
    import runpy
    d04 = runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_4_fixed_source_full_trajectory_v1.py"))
    batch = d04["build_repeated_endpoint"](training["fixed_sources"]["count"], device=device)
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    source, endpoint_lift, target_velocity, coupling_rows = fixed_sources(matcher, batch, seed=training["fixed_sources"]["source_noise_seed"])
    torch.manual_seed(training["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["model_seed"])
    model = GaugeFlowVectorField(
        hidden_dim=training["hidden_dim"], layers=training["layers"], conditioning_mode="unconditional",
        coordinate_rbf_dim=training["coordinate_rbf_dim"], coordinate_rbf_cutoff=training["coordinate_rbf_cutoff_angstrom"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    time_generator = torch.Generator(device=device).manual_seed(training["time_sampling"]["seed"])
    trace = []
    model.train()
    for step in range(1, training["steps"] + 1):
        time = torch.rand((batch.num_graphs,), device=device, generator=time_generator)
        optimizer.zero_grad(set_to_none=True)
        velocity_mse, map_mse, _ = metrics_at_time(model, matcher, batch, source, endpoint_lift, target_velocity, time)
        if not torch.isfinite(velocity_mse):
            raise FloatingPointError("P5-C0 encountered non-finite fixed-lift loss")
        velocity_mse.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == training["steps"]:
            trace.append({"step": step, "training_velocity_mse": float(velocity_mse.detach()), "training_map_mse": float(map_mse.detach())})
    model.eval()
    grid_rows = []
    with torch.no_grad():
        for value in torch.linspace(0.0, 1.0, evaluation["time_grid"]["count"], device=device):
            time = torch.full((batch.num_graphs,), value, device=device)
            velocity_mse, map_mse, teacher_rms = metrics_at_time(model, matcher, batch, source, endpoint_lift, target_velocity, time)
            grid_rows.append({"time": float(value), "velocity_mse": float(velocity_mse), "map_mse": float(map_mse), "teacher_forced_fixed_lift_rms": float(teacher_rms)})
        sampling_rows = []
        for steps in evaluation["sampling_steps"]:
            sampled = sample_universal_cover(model, matcher, batch, source, steps)
            fixed_rms = quotient_rms(sampled.frac_coords - endpoint_lift, batch)
            wrapped_rms = periodic_terminal_rms(sampled.frac_coords, endpoint_lift, batch)
            sampling_rows.append({"sampling_steps": steps, "free_running_fixed_lift_rms": float(fixed_rms), "terminal_wrapped_periodic_rms": float(wrapped_rms), "sampling_failures": int(not torch.isfinite(sampled.frac_coords).all())})
    grid = pd.DataFrame(grid_rows)
    sampling = pd.DataFrame(sampling_rows)
    criteria = protocol["pass_criteria"]
    velocity_mse = float(grid["velocity_mse"].mean())
    map_mse = float(grid["map_mse"].mean())
    teacher_rms = float(grid["teacher_forced_fixed_lift_rms"].mean())
    final = sampling.loc[sampling["sampling_steps"] == 100].iloc[0]
    free_rms = float(final["free_running_fixed_lift_rms"])
    failures = int(sampling["sampling_failures"].sum())
    passed = bool(velocity_mse <= criteria["all_time_grid_velocity_mse_max"] and teacher_rms <= criteria["teacher_forced_translation_aligned_rms_max"] and free_rms <= criteria["free_running_translation_aligned_rms_max"] and failures <= criteria["sampling_failures_max"])
    results = {"model_seed": training["model_seed"], "fixed_source_count": batch.num_graphs, "all_time_grid_velocity_mse": velocity_mse, "all_time_grid_map_mse": map_mse, "teacher_forced_fixed_lift_rms": teacher_rms, "free_running_fixed_lift_rms": free_rms, "terminal_wrapped_periodic_rms": float(final["terminal_wrapped_periodic_rms"]), "sampling_failures": failures, "passed": passed, "attribution": "fixed_lift_qualified" if passed else "fixed_lift_not_qualified"}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([results]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "learning_curve.csv", index=False)
    grid.to_csv(output / "time_grid_metrics.csv", index=False)
    sampling.to_csv(output / "sampling_step_curve.csv", index=False)
    coupling_rows.to_csv(output / "frozen_couplings.csv", index=False)
    manifest = {"schema": 1, "status": "passed_p5_c0" if passed else "not_passed_p5_c0", "attribution": results["attribution"], "next_gate_allowed": False, "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path), "audit_manifest_sha256": _sha256(audit_path), "runner_sha256": _sha256(Path(__file__)), "coupling_sha256": _sha256(ROOT / "src/gaugeflow/coupling.py"), "geometry_sha256": _sha256(ROOT / "src/gaugeflow/geometry.py"), "historical_results_modified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("# P5-C0 fixed-lift universal-cover training\n\n" + pd.DataFrame([results]).to_markdown(index=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
