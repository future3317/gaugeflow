"""Reconstruct archived D0.6 once and perform read-only long-horizon diagnostics."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import pandas as pd
import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.manifold import wrap01
from gaugeflow.model import QuotientRolloutFlowMap


ROOT = Path(__file__).resolve().parents[1]
D06 = runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_6_quotient_rollout_corrected_flow_map_v1.py"))


def _build_reconstructed_model(protocol: dict, device: torch.device):
    settings = protocol["phase_2_model"]
    torch.manual_seed(5201)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(5201)
    return QuotientRolloutFlowMap(
        hidden_dim=settings["hidden_dim"], layers=settings["layers"],
        coordinate_rbf_dim=settings["coordinate_rbf_dim"], coordinate_rbf_cutoff=settings["coordinate_rbf_cutoff_angstrom"],
        fourier_frequencies=settings["fourier_frequencies"], time_epsilon=settings["time_epsilon"],
    ).to(device)


def _train_d06_reconstruction(model, matcher, batch, source, source_to_endpoint, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(520106)
    model.train()
    for _ in range(5000):
        triples = torch.rand((batch.num_graphs, 3), device=device, generator=generator).sort(dim=-1).values
        start, middle, end = triples.unbind(dim=-1)
        optimizer.zero_grad(set_to_none=True)
        on, correction = D06["losses_for_triples"](
            model, matcher, batch, source, source_to_endpoint, start, middle, end
        )
        (on + correction).backward()
        optimizer.step()
    model.eval()


def _start_times(span: float, device: torch.device) -> torch.Tensor:
    count = 16 if span < 1.0 else 1
    return torch.linspace(0.0, 1.0 - span, count, device=device)


@torch.no_grad()
def _diagnose(model, matcher, batch, source, source_to_endpoint, protocol: dict):
    rows = []
    for span in protocol["phase_1_diagnostics"]["span_buckets"]:
        direct_errors, semigroup = [], []
        for start_value in _start_times(span, batch.frac_coords.device):
            start = torch.full((batch.num_graphs,), start_value, device=batch.frac_coords.device)
            end = torch.full((batch.num_graphs,), start_value + span, device=batch.frac_coords.device)
            middle = (start + end) / 2.0
            state_start = D06["analytic_state"](source, source_to_endpoint, batch, start)
            state_end = D06["analytic_state"](source, source_to_endpoint, batch, end)
            direct = D06["map_prediction"](model, matcher, state_start, batch, start, end)
            target = D06["quotient_logmap"](matcher, state_start.frac_coords, state_end.frac_coords, batch)
            direct_errors.append((direct - target).square().mean())
            direct_state = wrap01(state_start.frac_coords + direct)
            first = D06["map_prediction"](model, matcher, state_start, batch, start, middle)
            middle_state = wrap01(state_start.frac_coords + first)
            second = D06["map_prediction"](model, matcher, type(state_start)(state_start.type_state, middle_state, state_start.lattice_log), batch, middle, end)
            composed_state = wrap01(middle_state + second)
            semigroup.append(D06["aligned_rms"](direct_state, composed_state, batch.batch, batch.num_graphs))
        rows.append({"span": span, "direct_map_mse": float(torch.stack(direct_errors).mean()), "semigroup_defect_rms": float(torch.stack(semigroup).mean())})
    step_rows = []
    for steps in protocol["phase_1_diagnostics"]["sampling_steps"]:
        sampled = D06["sample_maps"](model, matcher, batch, source, steps)
        step_rows.append({"sampling_steps": steps, "final_translation_aligned_rms": float(D06["aligned_rms"](sampled.frac_coords, batch.frac_coords, batch.batch, batch.num_graphs))})
    perturb = protocol["phase_1_diagnostics"]["perturbation"]
    torch.manual_seed(perturb["seed"])
    raw = torch.randn_like(source.frac_coords)
    noise = matcher._coordinate_velocity(raw, batch)
    noise = noise * (perturb["quotient_fractional_rms"] / noise.square().mean().sqrt())
    clean = source.frac_coords.clone()
    noisy = wrap01(source.frac_coords + noise)
    errors = [float(D06["aligned_rms"](clean, noisy, batch.batch, batch.num_graphs))]
    steps = perturb["rollout_steps"]
    for step in range(steps):
        start = torch.full((batch.num_graphs,), step / steps, device=clean.device)
        end = torch.full((batch.num_graphs,), (step + 1) / steps, device=clean.device)
        clean_state = type(source)(source.type_state, clean, source.lattice_log)
        noisy_state = type(source)(source.type_state, noisy, source.lattice_log)
        clean = wrap01(clean + D06["map_prediction"](model, matcher, clean_state, batch, start, end))
        noisy = wrap01(noisy + D06["map_prediction"](model, matcher, noisy_state, batch, start, end))
        errors.append(float(D06["aligned_rms"](clean, noisy, batch.batch, batch.num_graphs)))
    ratios = [right / max(left, 1.0e-12) for left, right in zip(errors[:-1], errors[1:])]
    perturb_rows = [{"step": index, "quotient_state_distance": value, "local_amplification": None if index == 0 else ratios[index - 1]} for index, value in enumerate(errors)]
    return pd.DataFrame(rows), pd.DataFrame(step_rows), pd.DataFrame(perturb_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_7_multiscale_semigroup_flow_map_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_7_multiscale_semigroup_flow_map_v1/phase1_diagnostics"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = json.loads((ROOT / args.protocol).read_text(encoding="utf-8"))
    output = ROOT / args.output_dir
    if protocol["status"] != "pre_registered_not_started" or output.exists():
        raise ValueError("D0.7 phase 1 requires a fresh frozen protocol")
    device = torch.device(args.device)
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = D06["build_repeated_endpoint"](64, device=device)
    source, source_to_endpoint = D06["fixed_sources"](matcher, batch, seed=520101)
    model = _build_reconstructed_model(protocol, device)
    _train_d06_reconstruction(model, matcher, batch, source, source_to_endpoint, device)
    spans, steps, perturbation = _diagnose(model, matcher, batch, source, source_to_endpoint, protocol)
    output.mkdir(parents=True, exist_ok=False)
    spans.to_csv(output / "span_and_semigroup.csv", index=False)
    steps.to_csv(output / "sampling_step_curve.csv", index=False)
    perturbation.to_csv(output / "perturbation_amplification.csv", index=False)
    amplification = perturbation["local_amplification"].dropna()
    report = {
        "baseline": "D0.6 deterministic reconstruction; no checkpoint persisted",
        "max_span_direct_map_mse": float(spans.loc[spans["span"].idxmax(), "direct_map_mse"]),
        "max_semigroup_defect_rms": float(spans["semigroup_defect_rms"].max()),
        "mean_local_amplification": float(amplification.mean()),
        "max_local_amplification": float(amplification.max()),
        "rationale_sources": protocol["phase_1_diagnostics"]["literature"],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "rationale.md").write_text(
        "# D0.7 phase-1 rationale\n\n"
        "D0.6 learned adjacent maps but failed direct long maps and repeated composition. "
        "Flow Map Matching frames two-time maps and their composition as the primary object; "
        "Consistency Models motivate agreement across temporal resolutions; improved consistency training motivates explicit long-horizon and self-consistency diagnostics. "
        "D0.7 therefore uses equal span strata, endpoint-map mass, an un-detached two-step rollout target, and a semigroup loss.\n\n"
        + "\n".join(f"- {item['citation']}: {item['url']}" for item in protocol["phase_1_diagnostics"]["literature"])
        + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
