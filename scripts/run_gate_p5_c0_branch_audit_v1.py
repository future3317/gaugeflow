"""Read-only P5-C0 audit of lift, permutation, and translation-gauge branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import torch

from gaugeflow.coupling import _type_preserving_assignments, fixed_lift_coupling
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01


ROOT = Path(__file__).resolve().parents[1]
D04 = runpy.run_path(str(ROOT / "scripts" / "run_gate_p5_d0_4_fixed_source_full_trajectory_v1.py"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_coupling(source: torch.Tensor, target: torch.Tensor, lattice: torch.Tensor, source_types: torch.Tensor, target_types: torch.Tensor):
    """Apply the production exact fixed-lift solver independently per graph."""
    assignments = _type_preserving_assignments(source_types, target_types)
    if len(assignments) != 1:
        raise ValueError("the frozen P5-C0 panel must have one type-preserving assignment")
    rows = [
        fixed_lift_coupling(
            source[graph], target[graph], lattice[graph],
            source_types=source_types, target_types=target_types,
        )
        for graph in range(source.shape[0])
    ]
    return {
        "cost": torch.stack([row.cost for row in rows]),
        "second_cost": torch.stack([row.second_cost for row in rows]),
        "integer_lift": torch.stack([row.integer_lift for row in rows]),
        "translation": torch.stack([row.translation for row in rows]),
        "residual": torch.stack([source[graph] - target[graph, row.assignment] - row.integer_lift - row.translation for graph, row in enumerate(rows)]),
        "assignment": torch.stack([row.assignment for row in rows]),
        "permutation_count": 1,
        "second_permutation_cost": torch.full_like(torch.stack([row.cost for row in rows]), torch.inf),
    }


def _signature(value: torch.Tensor) -> list[str]:
    return [",".join(str(int(item)) for item in row.flatten().tolist()) for row in value]


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean(dim=(-1, -2)).sqrt()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_p5_c0_periodic_path_fixed_lift_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_p5_c0_periodic_path_fixed_lift_v1/audit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    output = ROOT / args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "pre_registered_not_started" or output.exists():
        raise ValueError("P5-C0 requires a fresh frozen audit protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5-C0 audit requested CUDA for exact batched lift enumeration")
    audit = protocol["audit"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = D04["build_repeated_endpoint"](audit["fixed_sources"]["count"], device=device)
    source, current_velocity = D04["fixed_sources"](matcher, batch, seed=audit["fixed_sources"]["source_noise_seed"])
    target = batch.frac_coords.reshape(batch.num_graphs, 4, 3)
    source_coords = source.frac_coords.reshape(batch.num_graphs, 4, 3)
    lattice = batch.lattice.reshape(batch.num_graphs, 3, 3)
    atom_types = batch.atom_types.reshape(batch.num_graphs, 4)[0]
    perturb_generator = torch.Generator(device=device).manual_seed(audit["perturbation"]["seed"])
    perturbation = torch.randn(source_coords.shape, device=device, generator=perturb_generator)
    perturbation = perturbation - perturbation.mean(dim=1, keepdim=True)
    perturbation = perturbation * (audit["perturbation"]["translation_quotient_fractional_rms"] / _rms(perturbation).clamp_min(1e-12))[:, None, None]
    times = torch.linspace(0.0, 1.0, audit["time_grid"]["count"], device=device)
    rows = []
    previous = None
    thresholds = audit["branch_switch_rule"]
    confirmed_events = 0
    for time_index, time in enumerate(times):
        state = wrap01(source_coords + time * current_velocity.reshape(batch.num_graphs, 4, 3))
        perturbed_state = wrap01(state + perturbation)
        solved = solve_coupling(state, target, lattice, atom_types, atom_types)
        perturbed = solve_coupling(perturbed_state, target, lattice, atom_types, atom_types)
        displacement = -solved["residual"]
        perturbed_displacement = -perturbed["residual"]
        jump = _rms((perturbed_displacement - displacement) - (perturbed_displacement - displacement).mean(dim=1, keepdim=True))
        lift_signature = _signature(solved["integer_lift"])
        perturbed_lift_signature = _signature(perturbed["integer_lift"])
        permutation_signature = _signature(solved["assignment"][:, :, None])
        perturbed_permutation_signature = _signature(perturbed["assignment"][:, :, None])
        component_cut_margin = (torus_logmap(state, target).abs() - 0.5).abs().amin(dim=(-1, -2))
        temporal_lift_change = [False] * batch.num_graphs
        temporal_permutation_change = [False] * batch.num_graphs
        path_jump = torch.zeros(batch.num_graphs, device=device)
        if previous is not None and time_index < times.numel() - 1:
            temporal_lift_change = [left != right for left, right in zip(previous["lift_signature"], lift_signature)]
            temporal_permutation_change = [left != right for left, right in zip(previous["permutation_signature"], permutation_signature)]
            scale = float((1.0 - time) / (1.0 - times[time_index - 1]))
            consistency = displacement - scale * previous["displacement"]
            consistency = consistency - consistency.mean(dim=1, keepdim=True)
            path_jump = _rms(consistency)
        for graph in range(batch.num_graphs):
            perturb_label_change = lift_signature[graph] != perturbed_lift_signature[graph] or permutation_signature[graph] != perturbed_permutation_signature[graph]
            temporal_label_change = temporal_lift_change[graph] or temporal_permutation_change[graph]
            event = bool((perturb_label_change and jump[graph] >= thresholds["perturbation_target_jump_threshold"]) or (temporal_label_change and path_jump[graph] >= thresholds["path_consistency_jump_threshold"]))
            confirmed_events += int(event)
            rows.append({
                "source": graph, "time_index": time_index, "time": float(time),
                "lift_cost": float(solved["cost"][graph]), "second_lift_cost": float(solved["second_cost"][graph]),
                "lift_cost_margin": float(solved["second_cost"][graph] - solved["cost"][graph]),
                "permutation_count": solved["permutation_count"], "permutation_cost": float(solved["cost"][graph]),
                "second_permutation_cost": float(solved["second_permutation_cost"][graph]),
                "permutation_cost_margin": float(solved["second_permutation_cost"][graph] - solved["cost"][graph]),
                "component_cut_margin_fractional": float(component_cut_margin[graph]),
                "lift_signature": lift_signature[graph], "permutation_signature": permutation_signature[graph],
                "translation_x": float(torch.remainder(solved["translation"][graph, 0], 1.0)),
                "translation_y": float(torch.remainder(solved["translation"][graph, 1], 1.0)),
                "translation_z": float(torch.remainder(solved["translation"][graph, 2], 1.0)),
                "temporal_lift_change": temporal_lift_change[graph], "temporal_permutation_change": temporal_permutation_change[graph],
                "temporal_path_consistency_jump_rms": float(path_jump[graph]),
                "perturbation_lift_change": lift_signature[graph] != perturbed_lift_signature[graph],
                "perturbation_permutation_change": permutation_signature[graph] != perturbed_permutation_signature[graph],
                "perturbation_target_displacement_jump_rms": float(jump[graph]),
                "confirmed_branch_event": event,
            })
        previous = {"lift_signature": lift_signature, "permutation_signature": permutation_signature, "displacement": displacement}
    frame = pd.DataFrame(rows)
    summary = {
        "sources": batch.num_graphs,
        "time_grid_count": int(times.numel()),
        "type_preserving_permutations": int(frame["permutation_count"].max()),
        "temporal_lift_change_rows": int(frame["temporal_lift_change"].sum()),
        "temporal_permutation_change_rows": int(frame["temporal_permutation_change"].sum()),
        "perturbation_lift_change_rows": int(frame["perturbation_lift_change"].sum()),
        "perturbation_permutation_change_rows": int(frame["perturbation_permutation_change"].sum()),
        "near_cut_rows": int((frame["component_cut_margin_fractional"] <= audit["perturbation"]["translation_quotient_fractional_rms"]).sum()),
        "near_degenerate_lift_rows": int((frame["lift_cost_margin"] <= thresholds["near_degenerate_lift_cost_margin_angstrom2"]).sum()),
        "near_degenerate_permutation_rows": int((frame["permutation_cost_margin"] <= thresholds["near_degenerate_permutation_cost_margin_angstrom2"]).sum()),
        "max_temporal_path_consistency_jump_rms": float(frame["temporal_path_consistency_jump_rms"].max()),
        "max_perturbation_target_displacement_jump_rms": float(frame["perturbation_target_displacement_jump_rms"].max()),
        "confirmed_branch_events": confirmed_events,
        "branch_switching_confirmed": confirmed_events > 0,
        "training_authorized": confirmed_events > 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "branch_audit_rows.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    manifest = {"schema": 1, "status": "branch_switching_confirmed" if confirmed_events else "no_material_branch_switching", "training_authorized": confirmed_events > 0, "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path), "runner_sha256": _sha256(Path(__file__)), "coupling_sha256": _sha256(ROOT / "src/gaugeflow/coupling.py"), "geometry_sha256": _sha256(ROOT / "src/gaugeflow/geometry.py"), "historical_results_modified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("# P5-C0 periodic-path branch audit\n\n" + pd.DataFrame([summary]).to_markdown(index=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
