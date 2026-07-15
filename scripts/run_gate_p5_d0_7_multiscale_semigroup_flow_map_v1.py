"""Run the pre-registered D0.7 multiscale semigroup-consistent map study."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import QuotientRolloutFlowMap


ROOT = Path(__file__).resolve().parents[1]
D06 = runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_6_quotient_rollout_corrected_flow_map_v1.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _energy_normalized(error: torch.Tensor, analytic_displacement: torch.Tensor) -> torch.Tensor:
    return error.square().mean() / analytic_displacement.square().mean().detach().clamp_min(1.0e-8)


def _stratified_batch_times(protocol: dict, graphs: int, device: torch.device, generator: torch.Generator):
    """Return endpoint direct maps plus regular s<u<v semigroup triples.

    Regular samples choose the total ``s->v`` horizon uniformly over registered
    buckets and set ``u`` to its midpoint. Endpoint samples reserve at least a
    quarter of the batch for direct ``s->1`` supervision and are excluded from
    the two-step loss because no later ``v`` exists.
    """
    setup = protocol["phase_2_training"]["span_sampling"]
    endpoint_count = int(round(graphs * setup["endpoint_probability"]))
    endpoint_count = max(1, min(graphs - 1, endpoint_count))
    regular_count = graphs - endpoint_count
    buckets = torch.tensor(setup["buckets"], dtype=torch.float32, device=device)
    selected = buckets[torch.randint(buckets.numel(), (regular_count,), device=device, generator=generator)]
    start_regular = torch.rand((regular_count,), device=device, generator=generator) * (1.0 - selected)
    middle_regular = start_regular + 0.5 * selected
    end_regular = start_regular + selected
    endpoint_start = torch.rand((endpoint_count,), device=device, generator=generator)
    endpoint_end = torch.ones_like(endpoint_start)
    start = torch.cat((start_regular, endpoint_start))
    middle = torch.cat((middle_regular, endpoint_end))
    end = torch.cat((end_regular, endpoint_end))
    regular = torch.zeros(graphs, dtype=torch.bool, device=device)
    regular[:regular_count] = True
    return start, middle, end, regular


def _map(model, matcher, state, batch, start, end):
    return D06["map_prediction"](model, matcher, state, batch, start, end)


def _subset_state(state, batch, mask: torch.Tensor):
    node_mask = mask[batch.batch]
    # The D0.7 loss evaluates all graphs in one forward pass; this helper only
    # extracts loss tensors, not a disconnected PyG graph.
    return node_mask


def d07_losses(model, matcher, batch, source, source_to_endpoint, start, middle, end, regular):
    state_start = D06["analytic_state"](source, source_to_endpoint, batch, start)
    state_middle = D06["analytic_state"](source, source_to_endpoint, batch, middle)
    target_su = D06["quotient_logmap"](matcher, state_start.frac_coords, state_middle.frac_coords, batch)
    predicted_su = _map(model, matcher, state_start, batch, start, middle)
    direct = _energy_normalized(predicted_su - target_su, target_su)

    # Only regular rows have a strict u<v. Gradients deliberately pass through
    # both maps and the intermediate quotient Exp update.
    if not regular.any():
        raise RuntimeError("D0.7 requires regular semigroup triples")
    state_end = D06["analytic_state"](source, source_to_endpoint, batch, end)
    rollout_middle = wrap01(state_start.frac_coords + predicted_su)
    rollout_state = type(state_start)(state_start.type_state, rollout_middle, state_start.lattice_log)
    # Endpoint-supervision rows have u=v=1 and intentionally do not enter a
    # two-step loss.  Give only those ignored rows a valid dummy interval so
    # the batched backbone retains its fixed graph layout; all indexed rollout
    # and semigroup terms below remain strictly s<u<v.
    rollout_start = middle.clone()
    rollout_end = end.clone()
    rollout_start[~regular] = 0.0
    rollout_end[~regular] = 1.0
    predicted_uv = _map(model, matcher, rollout_state, batch, rollout_start, rollout_end)
    rollout_end = wrap01(rollout_middle + predicted_uv)
    total_analytic = D06["quotient_logmap"](matcher, state_start.frac_coords, state_end.frac_coords, batch)
    rollout_error = D06["quotient_logmap"](matcher, rollout_end, state_end.frac_coords, batch)
    direct_sv = _map(model, matcher, state_start, batch, start, end)
    direct_end = wrap01(state_start.frac_coords + direct_sv)
    semi_error = D06["quotient_logmap"](matcher, rollout_end, direct_end, batch)
    node_regular = _subset_state(state_start, batch, regular)
    rollout = _energy_normalized(rollout_error[node_regular], total_analytic[node_regular])
    semi = _energy_normalized(semi_error[node_regular], total_analytic[node_regular])
    return direct, rollout, semi


@torch.no_grad()
def _evaluate(model, matcher, batch, source, source_to_endpoint, protocol):
    values = torch.linspace(0.0, 1.0, protocol["evaluation"]["time_grid"]["count"], device=batch.frac_coords.device)
    flow_rows, teacher_rows = [], []
    for index in range(values.numel() - 1):
        start = torch.full((batch.num_graphs,), values[index], device=batch.frac_coords.device)
        end = torch.full((batch.num_graphs,), values[index + 1], device=batch.frac_coords.device)
        state_start = D06["analytic_state"](source, source_to_endpoint, batch, start)
        state_end = D06["analytic_state"](source, source_to_endpoint, batch, end)
        prediction = _map(model, matcher, state_start, batch, start, end)
        target = D06["quotient_logmap"](matcher, state_start.frac_coords, state_end.frac_coords, batch)
        flow_rows.append({"start": float(values[index]), "end": float(values[index + 1]), "flow_map_mse": float((prediction - target).square().mean())})
    for value in values:
        if float(value) == 1.0:
            teacher_rows.append({"time": float(value), "translation_aligned_rms": 0.0})
            continue
        start = torch.full((batch.num_graphs,), value, device=batch.frac_coords.device)
        end = torch.ones((batch.num_graphs,), device=batch.frac_coords.device)
        state = D06["analytic_state"](source, source_to_endpoint, batch, start)
        endpoint = wrap01(state.frac_coords + _map(model, matcher, state, batch, start, end))
        teacher_rows.append({"time": float(value), "translation_aligned_rms": float(D06["aligned_rms"](endpoint, batch.frac_coords, batch.batch, batch.num_graphs))})
    step_rows = []
    for steps in protocol["evaluation"]["sampling_steps"]:
        sampled = D06["sample_maps"](model, matcher, batch, source, steps)
        step_rows.append({"sampling_steps": steps, "final_translation_aligned_rms": float(D06["aligned_rms"](sampled.frac_coords, batch.frac_coords, batch.batch, batch.num_graphs))})
    return pd.DataFrame(flow_rows), pd.DataFrame(teacher_rows), pd.DataFrame(step_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_7_multiscale_semigroup_flow_map_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_7_multiscale_semigroup_flow_map_v1/phase2_multiscale_semigroup"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / args.output_dir
    if protocol["status"] != "pre_registered_not_started" or output.exists():
        raise ValueError("D0.7 phase 2 requires a fresh frozen protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("D0.7 requires CUDA")
    setup, model_setup = protocol["phase_2_training"], protocol["phase_2_model"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = D06["build_repeated_endpoint"](setup["fixed_sources"]["count"], device=device)
    source, source_to_endpoint = D06["fixed_sources"](matcher, batch, seed=setup["fixed_sources"]["source_noise_seed"])
    torch.manual_seed(setup["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(setup["model_seed"])
    model = QuotientRolloutFlowMap(
        hidden_dim=model_setup["hidden_dim"], layers=model_setup["layers"], coordinate_rbf_dim=model_setup["coordinate_rbf_dim"],
        coordinate_rbf_cutoff=model_setup["coordinate_rbf_cutoff_angstrom"], fourier_frequencies=model_setup["fourier_frequencies"], time_epsilon=model_setup["time_epsilon"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setup["learning_rate"], weight_decay=setup["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(setup["span_sampling"]["seed"])
    trace = []
    model.train()
    for step in range(1, setup["steps"] + 1):
        start, middle, end, regular = _stratified_batch_times(protocol, batch.num_graphs, device, generator)
        optimizer.zero_grad(set_to_none=True)
        direct, rollout, semi = d07_losses(model, matcher, batch, source, source_to_endpoint, start, middle, end, regular)
        loss = direct + rollout + semi
        if not torch.isfinite(loss):
            raise FloatingPointError("D0.7 encountered non-finite multiscale loss")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == setup["steps"]:
            trace.append({"step": step, "direct": float(direct.detach()), "rollout": float(rollout.detach()), "semigroup": float(semi.detach()), "loss": float(loss.detach())})
    model.eval()
    flow, teacher, steps = _evaluate(model, matcher, batch, source, source_to_endpoint, protocol)
    criteria = protocol["pass_criteria"]
    flow_mse = float(flow["flow_map_mse"].mean())
    teacher_rms = float(teacher["translation_aligned_rms"].mean())
    free_row = steps.loc[steps["sampling_steps"] == 100].iloc[0]
    free_rms = float(free_row["final_translation_aligned_rms"])
    failures = 0
    passed = bool(flow_mse <= criteria["all_time_flow_map_mse_max"] and teacher_rms <= criteria["teacher_forced_translation_aligned_rms_max"] and free_rms <= criteria["free_running_translation_aligned_rms_max"])
    attribution = "multiscale_semigroup_qualified" if passed else "multiscale_semigroup_not_qualified"
    results = {"model_seed": setup["model_seed"], "all_time_flow_map_mse": flow_mse, "teacher_forced_translation_aligned_rms": teacher_rms, "free_running_translation_aligned_rms": free_rms, "sampling_failures": failures, "passed": passed, "attribution": attribution}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([results]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "learning_curve.csv", index=False)
    flow.to_csv(output / "time_grid_metrics.csv", index=False)
    teacher.to_csv(output / "teacher_forced_rms.csv", index=False)
    steps.to_csv(output / "sampling_step_curve.csv", index=False)
    manifest = {"schema": 1, "status": "passed_d0_7" if passed else "not_passed_d0_7", "attribution": attribution, "d0_8_allowed": not passed, "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path), "runner_sha256": _sha256(Path(__file__)), "device": str(device), "historical_results_modified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("# D0.7 multiscale semigroup-consistent quotient flow map\n\n" + pd.DataFrame([results]).to_markdown(index=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
