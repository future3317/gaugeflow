"""Run the pre-registered oracle-free exact rank-three conditional-control gate.

The target property is recomputed from generated coordinates by an analytic,
SO(3)-equivariant synthetic teacher.  It is intentionally a geometry-only
substrate control and cannot be cited as a real piezoelectric or joint-crystal
generation result.
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
from gaugeflow.manifold import log_vector_to_lattice, torus_logmap
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.synthetic_teacher import directed_species_rank3_teacher
from gaugeflow.tensor import normalize_isotypic, piezo_to_irreps, rotate_rank3


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _endpoint_specifications(dtype: torch.dtype) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    """Two fixed, non-congruent four-site coordinate endpoints."""
    # A generic triclinic metric avoids closest-image ties, which would make a
    # discontinuous tie-breaker masquerade as a failure of the analytic SO(3)
    # teacher under floating-point rotations.
    lattice = torch.tensor(
        ((3.9, 0.2, 0.1), (0.3, 4.3, 0.4), (0.1, 0.4, 5.1)), dtype=dtype
    )
    atom_types = torch.tensor((5, 7, 14, 32), dtype=torch.long)
    # These are synthetic chemical contrasts used only by the exact teacher;
    # model inputs are the usual fixed atom-type one-hot states.
    species_scalar = torch.tensor((-1.0, 0.35, 1.4, 2.1), dtype=dtype)
    return [
        {
            "frac": torch.tensor(
                ((0.06, 0.11, 0.19), (0.34, 0.22, 0.31), (0.72, 0.48, 0.41), (0.21, 0.79, 0.67)),
                dtype=dtype,
            ),
            "atom_types": atom_types,
            "lattice": lattice,
        },
        {
            "frac": torch.tensor(
                ((0.13, 0.07, 0.28), (0.43, 0.36, 0.14), (0.81, 0.57, 0.62), (0.28, 0.86, 0.54)),
                dtype=dtype,
            ),
            "atom_types": atom_types,
            "lattice": lattice,
        },
    ], species_scalar


def _teacher_tensor(frac: torch.Tensor, lattice: torch.Tensor, species_scalar: torch.Tensor) -> torch.Tensor:
    batch = torch.zeros(frac.shape[0], dtype=torch.long, device=frac.device)
    return directed_species_rank3_teacher(
        frac, lattice.unsqueeze(0), batch, species_scalar.to(frac), distance_decay=1.0
    )[0]


def build_panel(representatives: int, *, device: torch.device) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """Construct a two-orbit, multi-representative training/evaluation batch."""
    if representatives != 4:
        raise ValueError("P5 v1 freezes four representatives per tensor orbit")
    dtype = torch.float32
    endpoint_specs, species_scalar = _endpoint_specifications(dtype)
    frames = deterministic_so3_grid(12, dtype=dtype)[torch.tensor((0, 2, 5, 9))]
    target_tensors = torch.stack(
        [_teacher_tensor(spec["frac"], spec["lattice"], species_scalar) for spec in endpoint_specs]
    )
    raw_conditions = piezo_to_irreps(target_tensors)
    scales = torch.stack(
        [raw_conditions[:, block].square().mean().sqrt().clamp_min(1e-8) for block in (slice(0, 6), slice(6, 11), slice(11, 18))]
    )
    records: list[Data] = []
    for target_id, spec in enumerate(endpoint_specs):
        for representative_id, frame in enumerate(frames):
            condition = piezo_to_irreps(rotate_rank3(target_tensors[target_id], frame))
            records.append(
                Data(
                    atom_types=spec["atom_types"].clone(),
                    frac_coords=spec["frac"].clone(),
                    lattice=spec["lattice"].unsqueeze(0).clone(),
                    piezo_irreps=normalize_isotypic(condition.unsqueeze(0), scales).squeeze(0).unsqueeze(0),
                    condition_present=torch.ones(1, 1, dtype=torch.bool),
                    teacher_scalar=species_scalar.clone(),
                    target_id=torch.tensor([target_id], dtype=torch.long),
                    representative_id=torch.tensor([representative_id], dtype=torch.long),
                    num_nodes=spec["frac"].shape[0],
                )
            )
    return Batch.from_data_list(records).to(device), target_tensors.to(device), frames.to(device)


def _common_initial_state(matcher: RiemannianCrystalFlowMatcher, batch: Batch) -> CrystalFlowState:
    """Reuse each endpoint's base coordinate noise across its representatives."""
    state = matcher.random_state(batch)
    coordinates = state.frac_coords.clone()
    target_ids = batch.target_id.reshape(-1)
    for target_id in torch.unique(target_ids).tolist():
        graph_ids = torch.nonzero(target_ids == target_id, as_tuple=False).flatten()
        template = coordinates[batch.batch == graph_ids[0]].clone()
        for graph_id in graph_ids[1:].tolist():
            coordinates[batch.batch == graph_id] = template
    return CrystalFlowState(state.type_state, coordinates, state.lattice_log)


def _orbit_distances(predicted: torch.Tensor, targets: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Finite-grid evaluation approximation to distance from each target SO(3) orbit."""
    rotated_targets = rotate_rank3(targets.unsqueeze(1), grid.unsqueeze(0))
    delta = predicted[:, None, None] - rotated_targets.unsqueeze(0)
    return torch.linalg.vector_norm(delta.reshape(predicted.shape[0], targets.shape[0], grid.shape[0], -1), dim=-1).amin(dim=-1)


def _within_between_ratio(properties: torch.Tensor, target_ids: torch.Tensor) -> tuple[float, float, float]:
    distances = torch.cdist(properties.flatten(1), properties.flatten(1))
    eye = torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    same = target_ids[:, None] == target_ids[None, :]
    within = distances[same & ~eye].mean()
    between = distances[~same].mean()
    return float(between), float(within), float(between / within.clamp_min(1e-8))


@torch.no_grad()
def evaluate(
    model: GaugeFlowVectorField,
    matcher: RiemannianCrystalFlowMatcher,
    batch: Batch,
    targets: torch.Tensor,
    frames: torch.Tensor,
    *,
    sampler_steps: int,
) -> dict[str, float]:
    initial = _common_initial_state(matcher, batch)
    result = matcher.sample(model, batch, steps=sampler_steps, guidance_scale=0.0, initial_state=initial)
    if not isinstance(result, CrystalFlowState):
        raise RuntimeError("P5 does not request uncertainty sampling")
    lattice = log_vector_to_lattice(result.lattice_log)
    generated = directed_species_rank3_teacher(
        result.frac_coords, lattice, batch.batch, batch.teacher_scalar, distance_decay=1.0
    )
    finite = bool(torch.isfinite(result.frac_coords).all() and torch.isfinite(generated).all())
    orbit_grid = deterministic_so3_grid(240, dtype=generated.dtype, device=generated.device)
    distances = _orbit_distances(generated, targets, orbit_grid)
    predicted_target = distances.argmin(dim=-1)
    actual_target = batch.target_id.reshape(-1)
    retrieval = float((predicted_target == actual_target).float().mean())
    between, within, ratio = _within_between_ratio(generated, actual_target)
    representative_rms_values = []
    for target_id in torch.unique(actual_target).tolist():
        graph_ids = torch.nonzero(actual_target == target_id, as_tuple=False).flatten()
        reference = result.frac_coords[batch.batch == graph_ids[0]]
        for graph_id in graph_ids[1:].tolist():
            value = result.frac_coords[batch.batch == graph_id]
            representative_rms_values.append(torch.sqrt(torus_logmap(reference, value).square().mean()))
    representative_rms = float(torch.stack(representative_rms_values).mean())
    return {
        "exact_teacher_target_retrieval": retrieval,
        "target_orbit_distance_mean": float(distances[torch.arange(distances.shape[0]), actual_target].mean()),
        "target_orbit_distance_other_mean": float(distances[torch.arange(distances.shape[0]), 1 - actual_target].mean()),
        "between_exact_property": between,
        "within_exact_property": within,
        "between_within_exact_property_ratio": ratio,
        "common_noise_representative_coordinate_rms": representative_rms,
        "sampling_failures": 0 if finite else int(distances.shape[0]),
    }


def _teacher_equivariance(targets: torch.Tensor, frames: torch.Tensor, species_scalar: torch.Tensor) -> float:
    specs, _ = _endpoint_specifications(targets.dtype)
    rotation = frames[2]
    errors = []
    for spec, target in zip(specs, targets):
        frac = spec["frac"].to(target)
        lattice = spec["lattice"].to(target)
        rotated = _teacher_tensor(frac, lattice @ rotation.transpose(-1, -2), species_scalar)
        errors.append((rotated - rotate_rank3(target, rotation)).abs().max())
    return float(torch.stack(errors).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_exact_synthetic_tensor_control_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_exact_synthetic_tensor_control_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Write a missing manifest from an already complete results.csv; never retrains or resamples.",
    )
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("name") != "GaugeFlow P5 exact synthetic rank-three tensor-control gate v1":
        raise ValueError("P5 runner requires its exact versioned protocol")
    if protocol.get("status") != "pre_registered_not_started":
        raise ValueError("P5 protocol must remain a pre-registered unstarted contract")
    if (output / "manifest.json").exists():
        raise FileExistsError("P5 output already has a manifest; the fixed protocol must not be rerun")
    existing_results = output / "results.csv"
    if existing_results.exists() and not args.finalize_existing:
        raise FileExistsError(
            "P5 results.csv exists without a manifest. Use --finalize-existing to attest the completed run without retraining."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5 is a reported ML gate and requires the declared CUDA environment")
    settings = protocol["training"]
    if settings["conditioning_mode"] != "harmonic_alignment_v1" or settings["flow_heads"] != ["coord"]:
        raise ValueError("P5 v1 freezes harmonic alignment and the coordinate-only substrate")
    batch, targets, frames = build_panel(protocol["state"]["representatives_per_orbit"], device=device)
    species_scalar = batch.teacher_scalar[batch.batch == 0]
    teacher_equivariance = _teacher_equivariance(targets, frames, species_scalar)
    target_distance = float(_orbit_distances(targets, targets, deterministic_so3_grid(240, dtype=targets.dtype, device=device))[0, 1])
    output.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        frame = pd.read_csv(existing_results)
        if frame.seed.astype(int).tolist() != list(settings["seeds"]):
            raise ValueError("existing P5 results do not contain exactly the frozen seed order")
        csv_path = existing_results
    else:
        rows: list[dict[str, float | int]] = []
        for seed in settings["seeds"]:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = GaugeFlowVectorField(
                hidden_dim=settings["hidden_dim"], layers=settings["layers"], orbit_frames=settings["orbit_grid"],
                conditioning_mode=settings["conditioning_mode"],
            ).to(device)
            matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
            )
            final_loss = float("nan")
            model.train()
            for _ in range(settings["steps"]):
                optimizer.zero_grad(set_to_none=True)
                terms = matcher.loss(model, batch)
                if not torch.isfinite(terms["loss"]):
                    raise FloatingPointError("P5 encountered a non-finite flow objective")
                terms["loss"].backward()
                optimizer.step()
                final_loss = float(terms["loss"].detach())
            model.eval()
            torch.manual_seed(seed + protocol["evaluation"]["frozen_sample_seed_offset"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + protocol["evaluation"]["frozen_sample_seed_offset"])
            metrics = evaluate(
                model, matcher, batch, targets, frames, sampler_steps=protocol["evaluation"]["sampler_steps"]
            )
            rows.append({"seed": seed, "final_flow_loss": final_loss, **metrics})
        frame = pd.DataFrame(rows)
        csv_path = output / "results.csv"
        frame.to_csv(csv_path, index=False)
    thresholds = protocol["pass_criteria"]
    each_seed_pass = (
        (frame.exact_teacher_target_retrieval >= thresholds["exact_teacher_target_retrieval_min"])
        & (frame.between_within_exact_property_ratio >= thresholds["between_within_exact_property_ratio_min"])
        & (frame.common_noise_representative_coordinate_rms <= thresholds["common_noise_representative_coordinate_rms_max"])
        & (frame.sampling_failures <= thresholds["sampling_failures_max"])
    )
    passed = bool(
        teacher_equivariance <= thresholds["teacher_SO3_equivariance_max"]
        and target_distance >= thresholds["two_endpoint_teacher_orbit_distance_min"]
        and each_seed_pass.all()
    )
    manifest = {
        "schema": 1,
        "status": "passed_exact_synthetic_control" if passed else "not_passed_exact_synthetic_control",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "teacher_so3_equivariance_max_error": teacher_equivariance,
        "two_endpoint_teacher_orbit_distance": target_distance,
        "all_pre_registered_seed_criteria_pass": bool(each_seed_pass.all()),
        "real_tensor_oracle_used": False,
        "historical_gate_modified": False,
        "results": csv_path.name,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# P5 exact synthetic tensor-control gate\n\n"
        f"Status: `{manifest['status']}`. The property is recomputed analytically from generated coordinates; no learned oracle, real piezo label, relaxation, DFT, or DFPT is used.\n\n"
        f"- Teacher SO(3) equivariance error: `{teacher_equivariance:.8e}`\n"
        f"- Two-target finite-grid orbit distance: `{target_distance:.8e}`\n"
        f"- Every pre-registered seed passes: `{bool(each_seed_pass.all())}`\n\n"
        + frame.to_markdown(index=False)
        + "\n\nThis is only a coordinate-substrate test with fixed atom types and lattice metric. It cannot be used as evidence for joint crystal generation or real piezoelectric control.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
