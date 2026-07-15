"""Run P5-D0.2: exact single-example overfit trace for coordinate flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import torch

from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        canonical = value.detach().cpu().contiguous()
        digest.update(str(tuple(canonical.shape)).encode("ascii"))
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


def _d0_1_runner() -> dict[str, object]:
    return runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_1_fixed_batch_overfit_v1.py"))


def selected_example(*, device: torch.device) -> tuple[object, CrystalFlowState, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Recreate D0.1's frozen data, then select its first graph exactly."""
    parent = _d0_1_runner()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    train_batch = parent["build_repeated_endpoint"](64, device=device)
    state64, velocity64, time64 = parent["fixed_examples"](
        matcher, train_batch, source_noise_seed=520101, time_seed=520102
    )
    batch = parent["build_repeated_endpoint"](1, device=device)
    nodes = batch.atom_types.numel()
    target = matcher.target_state(batch)
    state = CrystalFlowState(
        type_state=target.type_state,
        frac_coords=state64.frac_coords[:nodes].clone(),
        lattice_log=target.lattice_log,
    )
    velocity = velocity64[:nodes].clone()
    time = time64[:1].clone()
    audit = {
        "parent_fixed_batch_sha256": parent["_tensor_sha256"](state64.frac_coords, velocity64, time64),
        "selected_index": 0,
        "selected_state_velocity_time_sha256": _tensor_sha256(state.frac_coords, velocity, time),
        "source_noise_seed": 520101,
        "time_seed": 520102,
    }
    return batch, state, velocity, time, audit


def _metrics(
    model: GaugeFlowVectorField,
    batch,
    state: CrystalFlowState,
    target_velocity: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)[1]
    mse = (predicted - target_velocity).square().mean()
    endpoint = wrap01(state.frac_coords + (1.0 - time[batch.batch]).unsqueeze(-1) * predicted)
    endpoint_rms = torch.sqrt(torus_logmap(endpoint, batch.frac_coords).square().mean())
    return mse, endpoint_rms, predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_2_single_sample_overfit_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_2_single_sample_overfit_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("name") != "GaugeFlow P5-D0.2 single-sample unconditional coordinate-flow overfit test v1":
        raise ValueError("P5-D0.2 runner requires its matching versioned protocol")
    if protocol.get("status") != "pre_registered_not_started":
        raise ValueError("P5-D0.2 protocol must remain pre-registered")
    if output.exists():
        raise FileExistsError("P5-D0.2 output already exists; the frozen test must not be rerun")
    training = protocol["training"]
    if training["conditioning_mode"] != "unconditional" or training["steps"] != 5000:
        raise ValueError("P5-D0.2 freezes the unconditional model and exactly 5,000 steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-D0.2 is a reported audit and requires the declared CUDA environment")
    batch, state, target_velocity, time, selected_audit = selected_example(device=device)
    torch.manual_seed(training["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["model_seed"])
    model = GaugeFlowVectorField(
        hidden_dim=training["hidden_dim"], layers=training["layers"], conditioning_mode="unconditional"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    trace: list[dict[str, float | int]] = []
    model.train()
    for step in range(training["steps"]):
        optimizer.zero_grad(set_to_none=True)
        mse, endpoint_rms, predicted = _metrics(model, batch, state, target_velocity, time)
        if not torch.isfinite(mse):
            raise FloatingPointError("P5-D0.2 encountered a non-finite single-sample loss")
        mse.backward()
        head_grad = model.coord_out.weight.grad
        if head_grad is None:
            raise RuntimeError("coordinate-head gradient is absent")
        trace.append({
            "step": step + 1,
            "velocity_mse": float(mse.detach()),
            "endpoint_periodic_rms": float(endpoint_rms.detach()),
            "coordinate_head_gradient_norm": float(torch.linalg.vector_norm(head_grad)),
            "coordinate_head_output_norm": float(torch.linalg.vector_norm(predicted.detach())),
            "target_velocity_norm": float(torch.linalg.vector_norm(target_velocity)),
        })
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_mse, final_rms, final_prediction = _metrics(model, batch, state, target_velocity, time)
    criteria = protocol["pass_criteria"]
    passed = bool(final_mse <= criteria["velocity_mse_max"] and final_rms <= criteria["endpoint_periodic_rms_max"])
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(trace).to_csv(output / "loss_curve.csv", index=False)
    components = []
    for node in range(final_prediction.shape[0]):
        for dimension in range(final_prediction.shape[1]):
            components.append({
                "node": node,
                "dimension": dimension,
                "predicted_coordinate_velocity": float(final_prediction[node, dimension]),
                "target_coordinate_velocity": float(target_velocity[node, dimension]),
                "difference": float(final_prediction[node, dimension] - target_velocity[node, dimension]),
            })
    pd.DataFrame(components).to_csv(output / "coordinate_components.csv", index=False)
    result = {
        "model_seed": training["model_seed"],
        "fixed_single_velocity_mse": float(final_mse),
        "fixed_single_endpoint_periodic_rms": float(final_rms),
        "single_sample_overfit_passed": passed,
        "failure_attribution": "passed" if passed else "model_forward_coordinate_head_loss_or_gradient_chain",
    }
    pd.DataFrame([result]).to_csv(output / "results.csv", index=False)
    (output / "selected_example_audit.json").write_text(json.dumps(selected_audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": 1,
        "status": "passed_single_sample_overfit" if passed else "not_passed_single_sample_overfit",
        "failure_attribution": result["failure_attribution"],
        "p5_d1_allowed": False,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "selected_example_audit": "selected_example_audit.json",
        "loss_curve": "loss_curve.csv",
        "coordinate_components": "coordinate_components.csv",
        "historical_p5_d0_1_modified": False,
        "results": "results.csv",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5-D0.2 single-sample unconditional coordinate-flow overfit\n\n"
        f"Single-sample overfit: `{passed}`. Attribution: `{result['failure_attribution']}`.\n\n"
        f"- Final velocity MSE: `{float(final_mse):.8e}` (required `<= {criteria['velocity_mse_max']:.1e}`)\n"
        f"- Final endpoint periodic RMS: `{float(final_rms):.8e}` (required `<= {criteria['endpoint_periodic_rms_max']:.3f}`)\n"
        "- `loss_curve.csv` records every update's loss, endpoint RMS, coordinate-head gradient norm, output norm, and target-velocity norm.\n"
        "- `coordinate_components.csv` records every final node/dimension prediction and target.\n\n"
        "No condition input, model-capacity change, extra training, harmonic module, oracle, or subsequent Gate is used. P5-D1 remains prohibited.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
