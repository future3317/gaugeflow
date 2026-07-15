"""Run the authorized P5-D0.3 quotient-coordinate fixed-batch qualification."""

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


def _endpoint(device: torch.device) -> Data:
    return Data(
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


def build_repeated_endpoint(count: int, *, device: torch.device) -> Batch:
    if count < 1:
        raise ValueError("fixed source count must be positive")
    endpoint = _endpoint(device)
    return Batch.from_data_list([endpoint.clone() for _ in range(count)]).to(device)


def fixed_examples(
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    *,
    source_noise_seed: int,
    time_seed: int,
) -> tuple[CrystalFlowState, torch.Tensor, torch.Tensor]:
    """Freeze sources and quotient-coordinate targets without resampling."""
    torch.manual_seed(source_noise_seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(source_noise_seed)
    source = matcher.random_state(batch)
    torch.manual_seed(time_seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(time_seed)
    time = torch.rand((batch.num_graphs,), dtype=batch.frac_coords.dtype, device=batch.frac_coords.device)
    target = matcher.target_state(batch)
    velocity = matcher._coordinate_velocity(torus_logmap(source.frac_coords, target.frac_coords), batch)
    state = CrystalFlowState(
        type_state=target.type_state,
        frac_coords=wrap01(source.frac_coords + time[batch.batch].unsqueeze(-1) * velocity),
        lattice_log=target.lattice_log,
    )
    return state, velocity, time


def _translation_aligned_rms(value: torch.Tensor, target: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    displacement = torus_logmap(value, target)
    mean = scatter(displacement, batch, dim=0, dim_size=graphs, reduce="mean")
    return (displacement - mean[batch]).square().mean().sqrt()


def fixed_batch_metrics(
    model: GaugeFlowVectorField,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    state: CrystalFlowState,
    target_velocity: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = matcher._coordinate_velocity(
        model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)[1], batch
    )
    velocity_mse = (predicted - target_velocity).square().mean()
    endpoint = wrap01(state.frac_coords + (1.0 - time[batch.batch]).unsqueeze(-1) * predicted)
    aligned_rms = _translation_aligned_rms(endpoint, batch.frac_coords, batch.batch, batch.num_graphs)
    absolute_rms = torus_logmap(endpoint, batch.frac_coords).square().mean().sqrt()
    return velocity_mse, aligned_rms, absolute_rms


def _initial_from_first_training_source(
    matcher: RiemannianCrystalFlowMatcher, batch: Batch, train_state: CrystalFlowState
) -> CrystalFlowState:
    target = matcher.target_state(batch)
    nodes = batch.atom_types.numel()
    return CrystalFlowState(target.type_state, train_state.frac_coords[:nodes].clone(), target.lattice_log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_3_translation_quotient_metric_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_3_translation_quotient_metric_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "prepared_not_started":
        raise ValueError("P5-D0.3 requires its frozen prepared contract")
    if output.exists():
        raise FileExistsError("P5-D0.3 output already exists; do not silently rerun the fixed audit")
    if protocol["state"]["coordinate_gauge"] != "translation_quotient_no_drift":
        raise ValueError("P5-D0.3 requires the translation quotient")
    if protocol["backbone"]["periodic_geometry"] != "closest_image_cartesian_distance_and_direction":
        raise ValueError("P5-D0.3 requires metric closest-image geometry")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-D0.3 is a reported ML audit and requires CUDA")
    training, evaluation = protocol["training"], protocol["evaluation"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    train_batch = build_repeated_endpoint(training["fixed_examples"]["count"], device=device)
    train_state, train_velocity, train_time = fixed_examples(
        matcher,
        train_batch,
        source_noise_seed=training["fixed_examples"]["source_noise_seed"],
        time_seed=training["fixed_examples"]["time_seed"],
    )
    torch.manual_seed(training["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["model_seed"])
    model = GaugeFlowVectorField(
        hidden_dim=training["hidden_dim"],
        layers=training["layers"],
        conditioning_mode="unconditional",
        coordinate_rbf_dim=protocol["backbone"]["coordinate_rbf_dim"],
        coordinate_rbf_cutoff=protocol["backbone"]["coordinate_rbf_cutoff_angstrom"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    trace = []
    model.train()
    for step in range(1, training["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        velocity_mse, _, _ = fixed_batch_metrics(model, matcher, train_batch, train_state, train_velocity, train_time)
        if not torch.isfinite(velocity_mse):
            raise FloatingPointError("P5-D0.3 encountered non-finite fixed-batch loss")
        velocity_mse.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == training["steps"]:
            trace.append({"step": step, "fixed_batch_velocity_mse": float(velocity_mse.detach())})

    model.eval()
    with torch.no_grad():
        train_mse, train_aligned_rms, train_absolute_rms = fixed_batch_metrics(
            model, matcher, train_batch, train_state, train_velocity, train_time
        )
        unseen = evaluation["unseen_examples"]
        unseen_batch = build_repeated_endpoint(unseen["count"], device=device)
        unseen_state, unseen_velocity, unseen_time = fixed_examples(
            matcher, unseen_batch, source_noise_seed=unseen["source_noise_seed"], time_seed=unseen["time_seed"]
        )
        unseen_mse, unseen_aligned_rms, unseen_absolute_rms = fixed_batch_metrics(
            model, matcher, unseen_batch, unseen_state, unseen_velocity, unseen_time
        )
        sample_batch = build_repeated_endpoint(1, device=device)
        sampled = matcher.sample(
            model,
            sample_batch,
            steps=evaluation["free_running"]["sampler_steps"],
            guidance_scale=evaluation["free_running"]["guidance_scale"],
            initial_state=_initial_from_first_training_source(matcher, sample_batch, train_state),
        )
        if not isinstance(sampled, CrystalFlowState):
            raise RuntimeError("P5-D0.3 does not request uncertainty sampling")
        free_aligned_rms = _translation_aligned_rms(
            sampled.frac_coords, sample_batch.frac_coords, sample_batch.batch, sample_batch.num_graphs
        )
        free_absolute_rms = torus_logmap(sampled.frac_coords, sample_batch.frac_coords).square().mean().sqrt()
        failures = int(not torch.isfinite(sampled.frac_coords).all())

    criteria = protocol["pass_criteria"]
    passed = bool(
        train_mse <= criteria["fixed_batch_velocity_mse_max"]
        and train_aligned_rms <= criteria["fixed_batch_translation_aligned_endpoint_rms_max"]
        and failures <= criteria["sampling_failures_max"]
    )
    if not passed:
        attribution = "fixed_batch_fit_failure"
    elif unseen_aligned_rms > criteria["fixed_batch_translation_aligned_endpoint_rms_max"]:
        attribution = "source_coupling_generalization_failure"
    elif free_aligned_rms > criteria["fixed_batch_translation_aligned_endpoint_rms_max"]:
        attribution = "free_running_integration_failure"
    else:
        attribution = "fixed_and_unseen_fit_with_free_running_closure"
    row = {
        "model_seed": training["model_seed"],
        "fixed_batch_velocity_mse": float(train_mse),
        "fixed_batch_translation_aligned_endpoint_rms": float(train_aligned_rms),
        "fixed_batch_absolute_origin_rms_diagnostic": float(train_absolute_rms),
        "unseen_velocity_mse": float(unseen_mse),
        "unseen_translation_aligned_endpoint_rms": float(unseen_aligned_rms),
        "unseen_absolute_origin_rms_diagnostic": float(unseen_absolute_rms),
        "free_running_translation_aligned_endpoint_rms": float(free_aligned_rms),
        "free_running_absolute_origin_rms_diagnostic": float(free_absolute_rms),
        "sampling_failures": failures,
        "passed": passed,
        "attribution": attribution,
    }
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([row]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "learning_curve.csv", index=False)
    manifest = {
        "schema": 1,
        "status": "passed_fixed_batch_qualification" if passed else "not_passed_fixed_batch_qualification",
        "attribution": attribution,
        "p5_d1_allowed": False,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "results": "results.csv",
        "learning_curve": "learning_curve.csv",
        "historical_results_modified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5-D0.3 translation-quotient metric coordinate-flow qualification\n\n"
        f"Fixed-batch qualification passed: `{passed}`. Attribution: `{attribution}`. P5-D1 allowed: `False`.\n\n"
        "This authorized one-endpoint test uses no tensor, endpoint ID, CFG, or harmonic input. "
        "It does not modify historical D0/D0.1/D0.2/P5 evidence and does not authorize a subsequent Gate.\n\n"
        + pd.DataFrame([row]).to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
