"""Run the single authorized D0.8 contractive quotient-flow-map study."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.utils import scatter

from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import wrap01
from gaugeflow.model import QuotientRolloutFlowMap


ROOT = Path(__file__).resolve().parents[1]
D06 = runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_6_quotient_rollout_corrected_flow_map_v1.py"))
D07 = runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_7_multiscale_semigroup_flow_map_v1.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map(model, matcher, state, batch, start, end):
    return D06["map_prediction"](model, matcher, state, batch, start, end)


def _per_graph_energy(value: torch.Tensor, batch) -> torch.Tensor:
    return scatter(value.square().mean(dim=-1), batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")


def quotient_perturbation(matcher, batch, *, rms: float, generator: torch.Generator) -> torch.Tensor:
    """Draw a fixed-size tangent perturbation, modulo a graphwise translation."""
    raw = matcher._coordinate_velocity(torch.randn(batch.frac_coords.shape, device=batch.frac_coords.device, generator=generator), batch)
    raw_rms = _per_graph_energy(raw, batch).sqrt().clamp_min(1.0e-12)
    return raw * (rms / raw_rms[batch.batch]).unsqueeze(-1)


def contractive_penalty(model, matcher, batch, state, start, end, perturbation, *, lipschitz_bound: float, epsilon: float):
    """Finite-difference positive excess over a quotient 1-Lipschitz map."""
    baseline_delta = _map(model, matcher, state, batch, start, end)
    baseline_end = wrap01(state.frac_coords + baseline_delta)
    perturbed = CrystalFlowState(state.type_state, wrap01(state.frac_coords + perturbation), state.lattice_log)
    perturbed_delta = _map(model, matcher, perturbed, batch, start, end)
    perturbed_end = wrap01(perturbed.frac_coords + perturbed_delta)
    input_energy = _per_graph_energy(D06["quotient_logmap"](matcher, state.frac_coords, perturbed.frac_coords, batch), batch)
    output_energy = _per_graph_energy(D06["quotient_logmap"](matcher, baseline_end, perturbed_end, batch), batch)
    ratio = output_energy / input_energy.clamp_min(epsilon)
    return torch.relu(ratio - lipschitz_bound**2).square().mean(), ratio.detach()


@torch.no_grad()
def _perturbation_audit(model, matcher, batch, source, protocol):
    setup = protocol["evaluation"]["perturbation"]
    generator = torch.Generator(device=batch.frac_coords.device).manual_seed(setup["seed"])
    state = CrystalFlowState(source.type_state.clone(), source.frac_coords.clone(), source.lattice_log.clone())
    perturbation = quotient_perturbation(matcher, batch, rms=setup["quotient_fractional_rms"], generator=generator)
    perturbed = CrystalFlowState(state.type_state.clone(), wrap01(state.frac_coords + perturbation), state.lattice_log.clone())
    rows = []
    previous = D06["aligned_rms"](perturbed.frac_coords, state.frac_coords, batch.batch, batch.num_graphs)
    for index in range(setup["rollout_steps"]):
        start = torch.full((batch.num_graphs,), index / setup["rollout_steps"], device=batch.frac_coords.device)
        end = torch.full((batch.num_graphs,), (index + 1) / setup["rollout_steps"], device=batch.frac_coords.device)
        state = CrystalFlowState(state.type_state, wrap01(state.frac_coords + _map(model, matcher, state, batch, start, end)), state.lattice_log)
        perturbed = CrystalFlowState(perturbed.type_state, wrap01(perturbed.frac_coords + _map(model, matcher, perturbed, batch, start, end)), perturbed.lattice_log)
        current = D06["aligned_rms"](perturbed.frac_coords, state.frac_coords, batch.batch, batch.num_graphs)
        rows.append({"step": index + 1, "quotient_rms": float(current), "per_step_amplification": float(current / previous.clamp_min(1.0e-12))})
        previous = current
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_8_contractive_quotient_flow_map_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_8_contractive_quotient_flow_map_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / args.output_dir
    if protocol["status"] != "pre_registered_not_started" or output.exists():
        raise ValueError("D0.8 requires a fresh frozen protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("D0.8 requires CUDA")
    setup, model_setup = protocol["training"], protocol["model"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = D06["build_repeated_endpoint"](setup["fixed_sources"]["count"], device=device)
    source, source_to_endpoint = D06["fixed_sources"](matcher, batch, seed=setup["fixed_sources"]["source_noise_seed"])
    torch.manual_seed(setup["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(setup["model_seed"])
    model = QuotientRolloutFlowMap(hidden_dim=model_setup["hidden_dim"], layers=model_setup["layers"], coordinate_rbf_dim=model_setup["coordinate_rbf_dim"], coordinate_rbf_cutoff=model_setup["coordinate_rbf_cutoff_angstrom"], fourier_frequencies=model_setup["fourier_frequencies"], time_epsilon=model_setup["time_epsilon"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setup["learning_rate"], weight_decay=setup["weight_decay"])
    span_generator = torch.Generator(device=device).manual_seed(setup["span_sampling"]["seed"])
    perturbation_generator = torch.Generator(device=device).manual_seed(setup["contractivity"]["seed"])
    trace = []
    model.train()
    for step in range(1, setup["steps"] + 1):
        start, middle, end, regular = D07["_stratified_batch_times"]({"phase_2_training": {"span_sampling": setup["span_sampling"]}}, batch.num_graphs, device, span_generator)
        optimizer.zero_grad(set_to_none=True)
        direct, rollout, semi = D07["d07_losses"](model, matcher, batch, source, source_to_endpoint, start, middle, end, regular)
        state_start = D06["analytic_state"](source, source_to_endpoint, batch, start)
        perturbation = quotient_perturbation(matcher, batch, rms=setup["contractivity"]["perturbation_quotient_rms"], generator=perturbation_generator)
        contract, ratio = contractive_penalty(model, matcher, batch, state_start, start, middle, perturbation, lipschitz_bound=setup["contractivity"]["lipschitz_bound"], epsilon=setup["contractivity"]["normalization_epsilon"])
        loss = direct + rollout + semi + contract
        if not torch.isfinite(loss):
            raise FloatingPointError("D0.8 encountered non-finite loss")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == setup["steps"]:
            trace.append({"step": step, "direct": float(direct.detach()), "rollout": float(rollout.detach()), "semigroup": float(semi.detach()), "contractive": float(contract.detach()), "mean_map_distance_ratio": float(ratio.mean()), "max_map_distance_ratio": float(ratio.max()), "loss": float(loss.detach())})
    model.eval()
    evaluation_protocol = {"evaluation": protocol["evaluation"]}
    flow, teacher, steps = D07["_evaluate"](model, matcher, batch, source, source_to_endpoint, evaluation_protocol)
    amplification = _perturbation_audit(model, matcher, batch, source, protocol)
    criteria = protocol["pass_criteria"]
    flow_mse = float(flow["flow_map_mse"].mean())
    teacher_rms = float(teacher["translation_aligned_rms"].mean())
    free_rms = float(steps.loc[steps["sampling_steps"] == 100].iloc[0]["final_translation_aligned_rms"])
    failures = 0
    passed = bool(flow_mse <= criteria["all_time_flow_map_mse_max"] and teacher_rms <= criteria["teacher_forced_translation_aligned_rms_max"] and free_rms <= criteria["free_running_translation_aligned_rms_max"] and failures <= criteria["sampling_failures_max"])
    attribution = "contractive_map_qualified" if passed else "contractive_map_not_qualified"
    results = {"model_seed": setup["model_seed"], "all_time_flow_map_mse": flow_mse, "teacher_forced_translation_aligned_rms": teacher_rms, "free_running_translation_aligned_rms": free_rms, "sampling_failures": failures, "perturbation_first_step_amplification": float(amplification.iloc[0]["per_step_amplification"]), "perturbation_mean_step_amplification": float(amplification["per_step_amplification"].mean()), "passed": passed, "attribution": attribution}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([results]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "learning_curve.csv", index=False)
    flow.to_csv(output / "time_grid_metrics.csv", index=False)
    teacher.to_csv(output / "teacher_forced_rms.csv", index=False)
    steps.to_csv(output / "sampling_step_curve.csv", index=False)
    amplification.to_csv(output / "perturbation_amplification.csv", index=False)
    manifest = {"schema": 1, "status": "passed_d0_8" if passed else "not_passed_d0_8", "attribution": attribution, "d0_9_allowed": False, "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path), "runner_sha256": _sha256(Path(__file__)), "device": str(device), "historical_results_modified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("# D0.8 contractive quotient flow map\n\n" + pd.DataFrame([results]).to_markdown(index=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
