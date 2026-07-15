"""Run P5-D0.1: fixed-batch overfit audit for unconditional coordinate flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch, Data

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


def _endpoint(dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Exact P5-D0 endpoint; copied without changing the frozen runner."""
    lattice = torch.tensor(
        ((3.9, 0.2, 0.1), (0.3, 4.3, 0.4), (0.1, 0.4, 5.1)), dtype=dtype
    )
    return {
        "frac": torch.tensor(
            ((0.06, 0.11, 0.19), (0.34, 0.22, 0.31), (0.72, 0.48, 0.41), (0.21, 0.79, 0.67)),
            dtype=dtype,
        ),
        "atom_types": torch.tensor((5, 7, 14, 32), dtype=torch.long),
        "lattice": lattice,
    }, torch.tensor((-1.0, 0.35, 1.4, 2.1), dtype=dtype)


def build_repeated_endpoint(count: int, *, device: torch.device) -> Batch:
    """Construct ``count`` identical endpoint graphs without any condition field."""
    if count < 1:
        raise ValueError("fixed source count must be positive")
    spec, _ = _endpoint(torch.float32)
    records = [
        Data(
            atom_types=spec["atom_types"].clone(),
            frac_coords=spec["frac"].clone(),
            lattice=spec["lattice"].unsqueeze(0).clone(),
            num_nodes=spec["frac"].shape[0],
        )
        for _ in range(count)
    ]
    return Batch.from_data_list(records).to(device)


def fixed_examples(
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    *,
    source_noise_seed: int,
    time_seed: int,
) -> tuple[CrystalFlowState, torch.Tensor, torch.Tensor]:
    """Generate and freeze a batched source/noise and one scalar time per graph."""
    torch.manual_seed(source_noise_seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(source_noise_seed)
    source = matcher.random_state(batch)
    torch.manual_seed(time_seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(time_seed)
    time = torch.rand((batch.num_graphs,), dtype=batch.frac_coords.dtype, device=batch.frac_coords.device)
    target = matcher.target_state(batch)
    velocity = torus_logmap(source.frac_coords, target.frac_coords)
    state = CrystalFlowState(
        type_state=target.type_state,
        frac_coords=wrap01(source.frac_coords + time[batch.batch].unsqueeze(-1) * velocity),
        lattice_log=target.lattice_log,
    )
    return state, velocity, time


def fixed_batch_metrics(
    model: GaugeFlowVectorField,
    batch: Batch,
    state: CrystalFlowState,
    target_velocity: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return velocity MSE and torus-path endpoint RMS for one frozen batch."""
    predicted = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)[1]
    velocity_mse = (predicted - target_velocity).square().mean()
    endpoint = wrap01(state.frac_coords + (1.0 - time[batch.batch]).unsqueeze(-1) * predicted)
    endpoint_rms = torch.sqrt(torus_logmap(endpoint, batch.frac_coords).square().mean())
    return velocity_mse, endpoint_rms


def _single_graph_initial_state(
    matcher: RiemannianCrystalFlowMatcher,
    sample_batch: Batch,
    train_state: CrystalFlowState,
) -> CrystalFlowState:
    """Reuse source index zero from the fixed train set for the one sample."""
    target = matcher.target_state(sample_batch)
    nodes = sample_batch.atom_types.numel()
    return CrystalFlowState(target.type_state, train_state.frac_coords[:nodes].clone(), target.lattice_log)


def _classify(
    train_passed: bool,
    unseen_rms: float,
    free_running_rms: float,
    attribution_reference: dict[str, float],
) -> str:
    if not train_passed:
        return "model_or_loss_cannot_memorize"
    if unseen_rms > attribution_reference["unseen_teacher_forced_endpoint_rms_max"]:
        return "source_coupling_generalization"
    if free_running_rms > attribution_reference["free_running_endpoint_rms_max"]:
        return "integration_trajectory"
    return "fixed_batch_fit_and_references_pass"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_1_fixed_batch_overfit_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_1_fixed_batch_overfit_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("name") != "GaugeFlow P5-D0.1 fixed-batch unconditional coordinate-flow overfit audit v1":
        raise ValueError("P5-D0.1 runner requires its matching versioned protocol")
    if protocol.get("status") != "pre_registered_not_started":
        raise ValueError("P5-D0.1 protocol must remain a pre-registered contract")
    if output.exists():
        raise FileExistsError("P5-D0.1 output directory already exists; the fixed audit must not be rerun")
    training, evaluation = protocol["training"], protocol["evaluation"]
    if training["conditioning_mode"] != "unconditional" or training["steps"] != 5000:
        raise ValueError("P5-D0.1 freezes the unconditional model and exactly 5,000 steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-D0.1 is a reported ML audit and requires the declared CUDA environment")
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    fixed = training["fixed_examples"]
    train_batch = build_repeated_endpoint(fixed["count"], device=device)
    train_state, train_velocity, train_time = fixed_examples(
        matcher, train_batch, source_noise_seed=fixed["source_noise_seed"], time_seed=fixed["time_seed"]
    )
    fixed_hash = _tensor_sha256(train_state.frac_coords, train_velocity, train_time)
    torch.manual_seed(training["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["model_seed"])
    model = GaugeFlowVectorField(
        hidden_dim=training["hidden_dim"], layers=training["layers"], conditioning_mode="unconditional"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    model.train()
    for _ in range(training["steps"]):
        optimizer.zero_grad(set_to_none=True)
        velocity_mse, _ = fixed_batch_metrics(model, train_batch, train_state, train_velocity, train_time)
        if not torch.isfinite(velocity_mse):
            raise FloatingPointError("P5-D0.1 encountered a non-finite fixed-batch velocity loss")
        velocity_mse.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_mse, train_rms = fixed_batch_metrics(model, train_batch, train_state, train_velocity, train_time)
        unseen = evaluation["unseen_examples"]
        unseen_batch = build_repeated_endpoint(unseen["count"], device=device)
        unseen_state, unseen_velocity, unseen_time = fixed_examples(
            matcher, unseen_batch, source_noise_seed=unseen["source_noise_seed"], time_seed=unseen["time_seed"]
        )
        _, unseen_rms = fixed_batch_metrics(model, unseen_batch, unseen_state, unseen_velocity, unseen_time)
        sample_batch = build_repeated_endpoint(1, device=device)
        initial = _single_graph_initial_state(matcher, sample_batch, train_state)
        sampled = matcher.sample(
            model, sample_batch, steps=evaluation["free_running"]["sampler_steps"], guidance_scale=0.0,
            initial_state=initial,
        )
        if not isinstance(sampled, CrystalFlowState):
            raise RuntimeError("P5-D0.1 does not request uncertainty sampling")
        free_running_rms = torch.sqrt(torus_logmap(sampled.frac_coords, sample_batch.frac_coords).square().mean())
    criteria = protocol["pass_criteria"]
    train_passed = bool(
        train_mse <= criteria["fixed_batch_velocity_mse_max"]
        and train_rms <= criteria["fixed_batch_endpoint_rms_max"]
    )
    attribution = _classify(train_passed, float(unseen_rms), float(free_running_rms), protocol["attribution_reference"])
    output.mkdir(parents=True, exist_ok=False)
    row = {
        "model_seed": training["model_seed"],
        "fixed_batch_velocity_mse": float(train_mse),
        "fixed_batch_endpoint_rms": float(train_rms),
        "fixed_train_batch_overfit_passed": train_passed,
        "unseen_teacher_forced_endpoint_rms": float(unseen_rms),
        "free_running_endpoint_rms": float(free_running_rms),
        "failure_attribution": attribution,
    }
    pd.DataFrame([row]).to_csv(output / "results.csv", index=False)
    fixed_audit = {
        "fixed_example_count": fixed["count"],
        "source_noise_seed": fixed["source_noise_seed"],
        "time_seed": fixed["time_seed"],
        "state_velocity_time_sha256": fixed_hash,
        "resampling_during_training": False,
        "unseen_example_count": unseen["count"],
        "unseen_source_noise_seed": unseen["source_noise_seed"],
        "unseen_time_seed": unseen["time_seed"],
    }
    (output / "fixed_batch_audit.json").write_text(json.dumps(fixed_audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": 1,
        "status": "passed_fixed_batch_overfit" if train_passed else "not_passed_fixed_batch_overfit",
        "failure_attribution": attribution,
        "p5_d1_allowed": False,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "fixed_batch_audit": "fixed_batch_audit.json",
        "historical_p5_d0_modified": False,
        "results": "results.csv",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5-D0.1 fixed-batch unconditional coordinate-flow overfit audit\n\n"
        f"Fixed training batch overfit: `{train_passed}`. Attribution: `{attribution}`. P5-D1 allowed: `False`.\n\n"
        "The 64 train `(source noise, t)` pairs were generated before model construction, hashed, and repeated unchanged for exactly 5,000 updates. No tensor, condition mask, endpoint ID, harmonic module, CFG, or resampling is used.\n\n"
        + pd.DataFrame([row]).to_markdown(index=False)
        + "\n\nThe P5-D0 result remains immutable. This audit ends here and does not authorize P5-D1, P3, P4, P6, oracle, real tensor, relaxation, DFT, or DFPT.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
