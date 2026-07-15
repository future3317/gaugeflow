"""Run P5-D0.6 Quotient Rollout-Corrected Flow Map exactly as registered."""

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
from gaugeflow.model import QuotientRolloutFlowMap


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
    source_to_endpoint = matcher._coordinate_velocity(torus_logmap(source.frac_coords, target.frac_coords), batch)
    return CrystalFlowState(target.type_state, source.frac_coords, target.lattice_log), source_to_endpoint


def analytic_state(source: CrystalFlowState, source_to_endpoint: torch.Tensor, batch: Batch, time: torch.Tensor) -> CrystalFlowState:
    return CrystalFlowState(
        source.type_state,
        wrap01(source.frac_coords + time[batch.batch].unsqueeze(-1) * source_to_endpoint),
        source.lattice_log,
    )


def quotient_logmap(matcher: RiemannianCrystalFlowMatcher, source: torch.Tensor, target: torch.Tensor, batch: Batch) -> torch.Tensor:
    return matcher._coordinate_velocity(torus_logmap(source, target), batch)


def aligned_rms(value: torch.Tensor, target: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    displacement = torus_logmap(value, target)
    mean = scatter(displacement, batch, dim=0, dim_size=graphs, reduce="mean")
    return (displacement - mean[batch]).square().mean().sqrt()


def map_prediction(
    model: QuotientRolloutFlowMap,
    matcher: RiemannianCrystalFlowMatcher,
    state: CrystalFlowState,
    batch: Batch,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    return matcher._coordinate_velocity(
        model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, start, end), batch
    )


def losses_for_triples(
    model: QuotientRolloutFlowMap,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    source: CrystalFlowState,
    source_to_endpoint: torch.Tensor,
    start: torch.Tensor,
    middle: torch.Tensor,
    end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_start = analytic_state(source, source_to_endpoint, batch, start)
    state_middle = analytic_state(source, source_to_endpoint, batch, middle)
    delta_target = quotient_logmap(matcher, state_start.frac_coords, state_middle.frac_coords, batch)
    delta_on = map_prediction(model, matcher, state_start, batch, start, middle)
    loss_on = (delta_on - delta_target).square().mean()
    # The rollout is intentionally detached: correction teaches the model on
    # states that its own first map creates without backpropagating through a
    # second-order target construction.
    state_hat = CrystalFlowState(
        state_start.type_state,
        wrap01(state_start.frac_coords + delta_on.detach()),
        state_start.lattice_log,
    )
    state_end = analytic_state(source, source_to_endpoint, batch, end)
    correction_target = quotient_logmap(matcher, state_hat.frac_coords, state_end.frac_coords, batch)
    delta_correction = map_prediction(model, matcher, state_hat, batch, middle, end)
    loss_correction = (delta_correction - correction_target).square().mean()
    return loss_on, loss_correction


@torch.no_grad()
def sample_maps(
    model: QuotientRolloutFlowMap,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    initial: CrystalFlowState,
    steps: int,
) -> CrystalFlowState:
    state = CrystalFlowState(initial.type_state.clone(), initial.frac_coords.clone(), initial.lattice_log.clone())
    for step in range(steps):
        start = torch.full((batch.num_graphs,), step / steps, device=batch.frac_coords.device)
        end = torch.full((batch.num_graphs,), (step + 1) / steps, device=batch.frac_coords.device)
        delta = map_prediction(model, matcher, state, batch, start, end)
        state = CrystalFlowState(state.type_state, wrap01(state.frac_coords + delta), state.lattice_log)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_6_quotient_rollout_corrected_flow_map_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_6_quotient_rollout_corrected_flow_map_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "pre_registered_not_started" or output.exists():
        raise ValueError("P5-D0.6 requires a fresh matching pre-registered contract")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-D0.6 requires CUDA")
    training, model_settings, evaluation = protocol["training"], protocol["model"], protocol["evaluation"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = build_repeated_endpoint(training["fixed_sources"]["count"], device=device)
    source, source_to_endpoint = fixed_sources(matcher, batch, seed=training["fixed_sources"]["source_noise_seed"])
    torch.manual_seed(training["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["model_seed"])
    model = QuotientRolloutFlowMap(
        hidden_dim=model_settings["hidden_dim"], layers=model_settings["layers"],
        coordinate_rbf_dim=model_settings["coordinate_rbf_dim"], coordinate_rbf_cutoff=model_settings["coordinate_rbf_cutoff_angstrom"],
        fourier_frequencies=model_settings["fourier_frequencies"], time_epsilon=model_settings["time_epsilon"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(training["triple_time_sampling"]["seed"])
    trace = []
    model.train()
    for step in range(1, training["steps"] + 1):
        triples = torch.rand((batch.num_graphs, 3), device=device, generator=generator).sort(dim=-1).values
        start, middle, end = triples.unbind(dim=-1)
        optimizer.zero_grad(set_to_none=True)
        loss_on, loss_correction = losses_for_triples(
            model, matcher, batch, source, source_to_endpoint, start, middle, end
        )
        loss = loss_on + loss_correction
        if not torch.isfinite(loss):
            raise FloatingPointError("P5-D0.6 encountered non-finite map loss")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == training["steps"]:
            trace.append({"step": step, "loss_on": float(loss_on.detach()), "loss_correction": float(loss_correction.detach()), "loss": float(loss.detach())})

    model.eval()
    grid_values = torch.linspace(0.0, 1.0, evaluation["time_grid"]["count"], device=device)
    grid_rows = []
    teacher_rms = []
    with torch.no_grad():
        for index in range(grid_values.numel() - 1):
            start = torch.full((batch.num_graphs,), grid_values[index], device=device)
            end = torch.full((batch.num_graphs,), grid_values[index + 1], device=device)
            state_start = analytic_state(source, source_to_endpoint, batch, start)
            state_end = analytic_state(source, source_to_endpoint, batch, end)
            prediction = map_prediction(model, matcher, state_start, batch, start, end)
            target = quotient_logmap(matcher, state_start.frac_coords, state_end.frac_coords, batch)
            grid_rows.append({"start": float(grid_values[index]), "end": float(grid_values[index + 1]), "flow_map_mse": float((prediction - target).square().mean())})
        for value in grid_values:
            start = torch.full((batch.num_graphs,), value, device=device)
            end = torch.ones((batch.num_graphs,), device=device)
            if bool((end <= start).any()):
                # The analytic endpoint map is identity at t=1, so no model
                # call with an invalid zero interval is made.
                teacher_rms.append(0.0)
                continue
            state = analytic_state(source, source_to_endpoint, batch, start)
            endpoint_prediction = wrap01(state.frac_coords + map_prediction(model, matcher, state, batch, start, end))
            teacher_rms.append(float(aligned_rms(endpoint_prediction, batch.frac_coords, batch.batch, batch.num_graphs)))
        sampled = sample_maps(model, matcher, batch, source, evaluation["free_running"]["sampler_steps"])
        free_rms = aligned_rms(sampled.frac_coords, batch.frac_coords, batch.batch, batch.num_graphs)
        failures = int(not torch.isfinite(sampled.frac_coords).all())
    grid = pd.DataFrame(grid_rows)
    criteria = protocol["pass_criteria"]
    flow_map_mse = float(grid["flow_map_mse"].mean())
    teacher_forced_rms = float(sum(teacher_rms) / len(teacher_rms))
    passed = bool(
        flow_map_mse <= criteria["all_time_flow_map_mse_max"]
        and teacher_forced_rms <= criteria["teacher_forced_translation_aligned_rms_max"]
        and float(free_rms) <= criteria["free_running_translation_aligned_rms_max"]
        and failures <= criteria["sampling_failures_max"]
    )
    attribution = "quotient_rollout_map_qualified" if passed else (
        "on_path_flow_map_fit_failure" if flow_map_mse > criteria["all_time_flow_map_mse_max"] else "rollout_or_free_running_failure"
    )
    results = {
        "model_seed": training["model_seed"], "fixed_source_count": batch.num_graphs,
        "time_grid_count": evaluation["time_grid"]["count"], "all_time_flow_map_mse": flow_map_mse,
        "teacher_forced_translation_aligned_rms": teacher_forced_rms,
        "free_running_translation_aligned_rms": float(free_rms), "sampling_failures": failures,
        "passed": passed, "attribution": attribution,
    }
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([results]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "learning_curve.csv", index=False)
    grid.to_csv(output / "time_grid_metrics.csv", index=False)
    (output / "teacher_forced_rms.csv").write_text(
        "time,translation_aligned_rms\n" + "\n".join(f"{float(time)},{rms}" for time, rms in zip(grid_values.cpu(), teacher_rms)) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": 1, "status": "passed_quotient_rollout_map" if passed else "not_passed_quotient_rollout_map",
        "attribution": attribution, "next_step_allowed": "versioned_unseen_source_flow_map_only" if passed else "none",
        "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)), "device": str(device), "results": "results.csv",
        "learning_curve": "learning_curve.csv", "time_grid_metrics": "time_grid_metrics.csv",
        "teacher_forced_rms": "teacher_forced_rms.csv", "historical_results_modified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5-D0.6 Quotient Rollout-Corrected Flow Map\n\n"
        f"Passed: `{passed}`. Attribution: `{attribution}`. No subsequent gate is automatically authorized.\n\n"
        "The model predicts finite quotient maps for `(s,u)` with Fourier interval features and FiLM in every message block. Training is exactly L_on + L_corr with a detached first rollout; sampling composes finite maps directly. No velocity Euler update, endpoint bridge coefficient, tensor, harmonic, or unseen source appears in this protocol.\n\n"
        + pd.DataFrame([results]).to_markdown(index=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
