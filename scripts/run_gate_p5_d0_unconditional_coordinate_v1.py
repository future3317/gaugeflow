"""Run P5-D0: a single-endpoint, genuinely unconditional coordinate-flow gate.

This is deliberately narrower than P5.  The batch and model forward receive no
tensor condition, condition mask, endpoint identifier, or harmonic alignment
object.  It qualifies only the coordinate substrate needed before P5-D1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.harmonic import deterministic_so3_grid
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.synthetic_teacher import directed_species_rank3_teacher
from gaugeflow.tensor import rotate_rank3


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _endpoint(dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """P5 v1 endpoint zero, copied verbatim to keep the path fixed."""
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


def _teacher_tensor(frac: torch.Tensor, lattice: torch.Tensor, species_scalar: torch.Tensor) -> torch.Tensor:
    batch = torch.zeros(frac.shape[0], dtype=torch.long, device=frac.device)
    return directed_species_rank3_teacher(
        frac, lattice.unsqueeze(0), batch, species_scalar.to(frac), distance_decay=1.0
    )[0]


def build_panel(*, device: torch.device) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """Return one fixed endpoint and its analytic tensor; no condition fields exist."""
    spec, species_scalar = _endpoint(torch.float32)
    target = _teacher_tensor(spec["frac"], spec["lattice"], species_scalar)
    record = Data(
        atom_types=spec["atom_types"].clone(),
        frac_coords=spec["frac"].clone(),
        lattice=spec["lattice"].unsqueeze(0).clone(),
        teacher_scalar=species_scalar.clone(),
        num_nodes=spec["frac"].shape[0],
    )
    return Batch.from_data_list([record]).to(device), target.to(device), species_scalar.to(device)


def _orbit_error(predicted: torch.Tensor, target: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    rotated = rotate_rank3(target.unsqueeze(0), grid)
    return torch.linalg.vector_norm((predicted.unsqueeze(0) - rotated).reshape(grid.shape[0], -1), dim=-1).amin()


def _periodic_rms(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torus_logmap(predicted, target).square().mean())


@torch.no_grad()
def _teacher_forced_metrics(
    model: GaugeFlowVectorField,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    *,
    time_points: int,
) -> tuple[float, float]:
    """Assess tangent fitting independently of free-running Euler integration."""
    if time_points < 2:
        raise ValueError("P5-D0 freezes at least two teacher-forced time points")
    target = matcher.target_state(batch)
    endpoint_errors, velocity_errors = [], []
    for time_value in torch.linspace(0.05, 0.95, time_points, device=batch.frac_coords.device):
        base = matcher.random_state(batch)
        velocity = torus_logmap(base.frac_coords, target.frac_coords)
        time = time_value.reshape(1)
        state = CrystalFlowState(
            type_state=target.type_state,
            frac_coords=wrap01(base.frac_coords + time_value * velocity),
            lattice_log=target.lattice_log,
        )
        predicted = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch, time
        )[1]
        endpoint = wrap01(state.frac_coords + (1.0 - time_value) * predicted)
        endpoint_errors.append(_periodic_rms(endpoint, target.frac_coords))
        velocity_errors.append((predicted - velocity).square().mean())
    return float(torch.stack(endpoint_errors).mean()), float(torch.stack(velocity_errors).mean())


@torch.no_grad()
def evaluate(
    model: GaugeFlowVectorField,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    target_tensor: torch.Tensor,
    *,
    sampler_steps: int,
    orbit_grid_size: int,
    thresholds: dict[str, float],
    teacher_forced_time_points: int,
) -> dict[str, float | int]:
    result = matcher.sample(model, batch, steps=sampler_steps, guidance_scale=0.0)
    if not isinstance(result, CrystalFlowState):
        raise RuntimeError("P5-D0 does not request uncertainty sampling")
    generated = _teacher_tensor(result.frac_coords, batch.lattice[0], batch.teacher_scalar)
    finite = bool(torch.isfinite(result.frac_coords).all() and torch.isfinite(generated).all())
    coordinate_rms = _periodic_rms(result.frac_coords, batch.frac_coords)
    orbit_error = _orbit_error(
        generated, target_tensor, deterministic_so3_grid(orbit_grid_size, dtype=generated.dtype, device=generated.device)
    )
    relative_orbit_error = orbit_error / torch.linalg.vector_norm(target_tensor).clamp_min(1e-8)
    teacher_forced_rms, teacher_forced_velocity_mse = _teacher_forced_metrics(
        model, matcher, batch, time_points=teacher_forced_time_points
    )
    retrieved = bool(
        finite
        and coordinate_rms <= thresholds["periodic_coordinate_rms_max"]
        and relative_orbit_error <= thresholds["analytic_teacher_target_orbit_relative_error_max"]
    )
    return {
        "periodic_coordinate_rms": float(coordinate_rms),
        "analytic_teacher_target_orbit_error": float(orbit_error),
        "analytic_teacher_target_orbit_relative_error": float(relative_orbit_error),
        "unique_endpoint_retrieval": float(retrieved),
        "teacher_forced_endpoint_rms": teacher_forced_rms,
        "teacher_forced_coordinate_velocity_mse": teacher_forced_velocity_mse,
        "sampling_failures": 0 if finite else 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_d0_unconditional_coordinate_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_d0_unconditional_coordinate_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("name") != "GaugeFlow P5-D0 single-endpoint unconditional coordinate-flow qualification v1":
        raise ValueError("P5-D0 runner requires its matching versioned protocol")
    if protocol.get("status") != "pre_registered_not_started":
        raise ValueError("P5-D0 protocol must remain a pre-registered contract")
    if output.exists():
        raise FileExistsError("P5-D0 output directory already exists; the frozen protocol must not be rerun")
    if protocol["training"]["conditioning_mode"] != "unconditional":
        raise ValueError("P5-D0 forbids every conditional encoder")
    if protocol["training"]["flow_heads"] != ["coord"] or protocol["training"]["CFG"] != 0.0:
        raise ValueError("P5-D0 freezes coordinate-only flow and CFG=0")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-D0 is a reported ML gate and requires the declared CUDA environment")
    settings, evaluation, thresholds = protocol["training"], protocol["evaluation"], protocol["pass_criteria"]
    batch, target_tensor, species_scalar = build_panel(device=device)
    # P5's exact teacher still has to be valid independently of the model.
    rotation = deterministic_so3_grid(12, dtype=target_tensor.dtype, device=device)[5]
    spec, _ = _endpoint(target_tensor.dtype)
    rotated = _teacher_tensor(spec["frac"].to(device), spec["lattice"].to(device) @ rotation.T, species_scalar)
    teacher_equivariance = float((rotated - rotate_rank3(target_tensor, rotation)).abs().max())
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, float | int]] = []
    for seed in settings["seeds"]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = GaugeFlowVectorField(
            hidden_dim=settings["hidden_dim"], layers=settings["layers"], conditioning_mode="unconditional"
        ).to(device)
        matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        final_loss = float("nan")
        model.train()
        for _ in range(settings["steps"]):
            optimizer.zero_grad(set_to_none=True)
            terms = matcher.loss(model, batch)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("P5-D0 encountered a non-finite flow objective")
            terms["loss"].backward()
            optimizer.step()
            final_loss = float(terms["loss"].detach())
        model.eval()
        torch.manual_seed(seed + evaluation["frozen_sample_seed_offset"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + evaluation["frozen_sample_seed_offset"])
        metrics = evaluate(
            model, matcher, batch, target_tensor,
            sampler_steps=evaluation["sampler_steps"], orbit_grid_size=evaluation["orbit_error_grid"],
            thresholds=thresholds, teacher_forced_time_points=evaluation["teacher_forced_time_points"],
        )
        rows.append({"seed": seed, "final_flow_loss": final_loss, **metrics})
    frame = pd.DataFrame(rows)
    teacher_forced_fit = frame.teacher_forced_endpoint_rms <= thresholds["teacher_forced_endpoint_rms_max"]
    sample_pass = (
        (frame.periodic_coordinate_rms <= thresholds["periodic_coordinate_rms_max"])
        & (frame.analytic_teacher_target_orbit_relative_error <= thresholds["analytic_teacher_target_orbit_relative_error_max"])
        & (frame.unique_endpoint_retrieval >= thresholds["unique_endpoint_retrieval_min"])
        & (frame.sampling_failures <= thresholds["sampling_failures_max"])
    )
    passed = bool(teacher_equivariance <= thresholds["teacher_SO3_equivariance_max"] and teacher_forced_fit.all() and sample_pass.all())
    if not teacher_forced_fit.all():
        attribution = "training_fit_failure"
    elif not sample_pass.all():
        attribution = "free_running_sampling_failure"
    else:
        attribution = "passed"
    csv_path = output / "results.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        "schema": 1,
        "status": "passed_unconditional_coordinate_substrate" if passed else "not_passed_unconditional_coordinate_substrate",
        "failure_attribution": attribution,
        "p5_d1_allowed": bool(passed),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "teacher_so3_equivariance_max_error": teacher_equivariance,
        "all_pre_registered_seed_criteria_pass": bool(teacher_forced_fit.all() and sample_pass.all()),
        "historical_p5_modified": False,
        "results": csv_path.name,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5-D0 single-endpoint unconditional coordinate-flow qualification\n\n"
        f"Status: `{manifest['status']}`. Failure attribution: `{attribution}`. P5-D1 allowed: `{bool(passed)}`.\n\n"
        "This model receives only the current flow state, graph index, and time. It receives no tensor, condition mask/null token, endpoint ID, harmonic alignment/grid, CFG, learned oracle, or real-material response.\n\n"
        f"- Analytic teacher SO(3) equivariance error: `{teacher_equivariance:.8e}`\n"
        f"- Every pre-registered seed passes: `{bool(teacher_forced_fit.all() and sample_pass.all())}`\n\n"
        + frame.to_markdown(index=False)
        + "\n\nThe frozen P5 conditional negative result is not modified. This report does not activate P5-D1 unless its manifest says `p5_d1_allowed: true`; it does not authorize P3, P4, oracle, real tensor, relaxation, DFT, or DFPT.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
