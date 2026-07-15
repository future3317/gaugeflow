"""Run P5-D0.4: complete quotient trajectories on the fixed D0.3 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_repeated_endpoint(count: int, *, device: torch.device) -> Batch:
    if count < 1:
        raise ValueError("fixed source count must be positive")
    endpoint = Data(
        atom_types=torch.tensor((5, 7, 14, 32), dtype=torch.long, device=device),
        frac_coords=torch.tensor(
            ((0.06, 0.11, 0.19), (0.34, 0.22, 0.31), (0.72, 0.48, 0.41), (0.21, 0.79, 0.67)),
            dtype=torch.float32, device=device,
        ),
        lattice=torch.tensor(
            ((3.9, 0.2, 0.1), (0.3, 4.3, 0.4), (0.1, 0.4, 5.1)),
            dtype=torch.float32, device=device,
        ).unsqueeze(0),
        num_nodes=4,
    )
    return Batch.from_data_list([endpoint.clone() for _ in range(count)]).to(device)


def fixed_sources(matcher: RiemannianCrystalFlowMatcher, batch: Batch, *, seed: int) -> tuple[CrystalFlowState, torch.Tensor]:
    torch.manual_seed(seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(seed)
    source = matcher.random_state(batch)
    target = matcher.target_state(batch)
    velocity = matcher._coordinate_velocity(torus_logmap(source.frac_coords, target.frac_coords), batch)
    return CrystalFlowState(target.type_state, source.frac_coords, target.lattice_log), velocity


def interpolant(source: CrystalFlowState, velocity: torch.Tensor, batch: Batch, time: torch.Tensor) -> CrystalFlowState:
    return CrystalFlowState(
        source.type_state,
        wrap01(source.frac_coords + time[batch.batch].unsqueeze(-1) * velocity),
        source.lattice_log,
    )


def _aligned_rms(value: torch.Tensor, target: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    displacement = torus_logmap(value, target)
    graph_mean = scatter(displacement, batch, dim=0, dim_size=graphs, reduce="mean")
    return (displacement - graph_mean[batch]).square().mean().sqrt()


def metrics_at_time(
    model: GaugeFlowVectorField,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    source: CrystalFlowState,
    target_velocity: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = interpolant(source, target_velocity, batch, time)
    predicted = matcher._coordinate_velocity(
        model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)[1], batch
    )
    velocity_mse = (predicted - target_velocity).square().mean()
    endpoint = wrap01(state.frac_coords + (1.0 - time[batch.batch]).unsqueeze(-1) * predicted)
    return velocity_mse, _aligned_rms(endpoint, batch.frac_coords, batch.batch, batch.num_graphs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_4_fixed_source_full_trajectory_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_4_fixed_source_full_trajectory_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "pre_registered_not_started":
        raise ValueError("P5-D0.4 requires its matching pre-registered contract")
    if output.exists():
        raise FileExistsError("P5-D0.4 output exists; the fixed test must not be silently rerun")
    if protocol["state"]["coordinate_gauge"] != "translation_quotient_no_drift":
        raise ValueError("P5-D0.4 requires translation-quotient coordinates")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-D0.4 is a reported learning test and requires CUDA")

    training, evaluation = protocol["training"], protocol["evaluation"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = build_repeated_endpoint(training["fixed_sources"]["count"], device=device)
    source, target_velocity = fixed_sources(matcher, batch, seed=training["fixed_sources"]["source_noise_seed"])
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
        velocity_mse, _ = metrics_at_time(model, matcher, batch, source, target_velocity, time)
        if not torch.isfinite(velocity_mse):
            raise FloatingPointError("P5-D0.4 encountered a non-finite training loss")
        velocity_mse.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == training["steps"]:
            trace.append({"step": step, "training_velocity_mse": float(velocity_mse.detach())})

    model.eval()
    grid_rows = []
    with torch.no_grad():
        for value in torch.linspace(0.0, 1.0, evaluation["time_grid"]["count"], device=device):
            time = torch.full((batch.num_graphs,), value, device=device)
            velocity_mse, aligned_rms = metrics_at_time(model, matcher, batch, source, target_velocity, time)
            grid_rows.append({"time": float(value), "velocity_mse": float(velocity_mse), "teacher_forced_translation_aligned_rms": float(aligned_rms)})
        grid = pd.DataFrame(grid_rows)
        initial = CrystalFlowState(source.type_state, source.frac_coords.clone(), source.lattice_log)
        sampled = matcher.sample(
            model, batch, steps=evaluation["free_running"]["sampler_steps"],
            guidance_scale=evaluation["free_running"]["guidance_scale"], initial_state=initial,
        )
        if not isinstance(sampled, CrystalFlowState):
            raise RuntimeError("P5-D0.4 does not request uncertainty sampling")
        free_rms = _aligned_rms(sampled.frac_coords, batch.frac_coords, batch.batch, batch.num_graphs)
        failures = int(not torch.isfinite(sampled.frac_coords).all())

    criteria = protocol["pass_criteria"]
    mean_velocity_mse = float(grid["velocity_mse"].mean())
    mean_teacher_rms = float(grid["teacher_forced_translation_aligned_rms"].mean())
    passed = bool(
        mean_velocity_mse <= criteria["all_time_grid_velocity_mse_max"]
        and mean_teacher_rms <= criteria["teacher_forced_translation_aligned_rms_max"]
        and float(free_rms) <= criteria["free_running_translation_aligned_rms_max"]
        and failures <= criteria["sampling_failures_max"]
    )
    if passed:
        attribution = "complete_fixed_source_trajectory_learned"
    elif mean_velocity_mse > criteria["all_time_grid_velocity_mse_max"]:
        attribution = "time_conditioning_or_vector_field_expression_failure"
    else:
        attribution = "trajectory_stability_failure"
    results = {
        "model_seed": training["model_seed"],
        "fixed_source_count": batch.num_graphs,
        "time_grid_count": evaluation["time_grid"]["count"],
        "all_time_grid_velocity_mse": mean_velocity_mse,
        "teacher_forced_translation_aligned_rms": mean_teacher_rms,
        "free_running_translation_aligned_rms": float(free_rms),
        "sampling_failures": failures,
        "passed": passed,
        "attribution": attribution,
    }
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([results]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "learning_curve.csv", index=False)
    grid.to_csv(output / "time_grid_metrics.csv", index=False)
    manifest = {
        "schema": 1,
        "status": "passed_fixed_source_trajectory" if passed else "not_passed_fixed_source_trajectory",
        "attribution": attribution,
        "next_step_allowed": "versioned_unseen_source_generalization_only" if passed else "none",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "results": "results.csv",
        "learning_curve": "learning_curve.csv",
        "time_grid_metrics": "time_grid_metrics.csv",
        "historical_results_modified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5-D0.4 fixed-source full-trajectory coordinate-flow learning test\n\n"
        f"Passed: `{passed}`. Attribution: `{attribution}`. No subsequent gate is authorized by this run.\n\n"
        "The 64 D0.3 source noises are fixed. Each optimization update draws a new independent Uniform[0,1] time for every source. No unseen source, tensor, endpoint ID, CFG, or harmonic condition is evaluated.\n\n"
        + pd.DataFrame([results]).to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
