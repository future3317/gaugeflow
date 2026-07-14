"""Run the fixed Gate A5 quotient-substrate repair screen.

This is intentionally endpoint-ID-only.  It qualifies the generator substrate
after A4; it does not load tensor-conditioned checkpoints or modify a historic
gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audit_gate_a4_generator_substrate import (  # noqa: E402
    HEADS,
    _decoded_metrics,
    _finite_state,
    _generated_batch,
    _load_panel,
    _localize_types,
    _set_endpoint_ids,
    _state_copy,
    _rms,
)
from evaluate_gate_a import _distance_diagnostics, _state_features  # noqa: E402
from gaugeflow.coupling import periodic_assignment, periodic_assignment_cost, translation_aligned_torus_rms  # noqa: E402
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher  # noqa: E402
from gaugeflow.manifold import project_simplex, torus_logmap, wrap01  # noqa: E402
from gaugeflow.model import GaugeFlowVectorField  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(value: int) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _common_initial(matcher: RiemannianCrystalFlowMatcher, batch, atoms: int, replicates: int, seed: int):
    _seed(seed)
    device = batch.batch.device
    if matcher.type_path == "riemannian_simplex":
        type_template = torch.distributions.Dirichlet(
            torch.ones(matcher.atom_types, device=device)
        ).sample((replicates * atoms,)).reshape(replicates, atoms, matcher.atom_types)
    else:
        type_template = torch.randn((replicates, atoms, matcher.atom_types), device=device)
    coord_template = torch.rand((replicates, atoms, 3), device=device)
    lattice_template = torch.randn((replicates, 6), device=device)
    state = CrystalFlowState(
        type_template.repeat(2, 1, 1).reshape(-1, matcher.atom_types),
        coord_template.repeat(2, 1, 1).reshape(-1, 3),
        lattice_template.repeat(2, 1),
    )
    target = matcher.target_state(batch)
    return CrystalFlowState(
        state.type_state if "type" in matcher.active_heads else target.type_state,
        state.frac_coords if "coord" in matcher.active_heads else target.frac_coords,
        state.lattice_log if "lattice" in matcher.active_heads else target.lattice_log,
    )


def _coupling_audit(raw_batch, protocol: dict[str, Any]):
    settings = protocol["a5_0_path_and_coupling_audit"]
    matcher = RiemannianCrystalFlowMatcher(
        type_path="riemannian_simplex", target_coupling="optimal_transport", coordinate_gauge="no_drift"
    )
    target = matcher.target_state(raw_batch)
    rows = []
    for repeat in range(settings["coupling_replicates"]):
        _seed(settings["initial_noise_seed"] + repeat)
        base = matcher.random_state(raw_batch)
        for graph, material_id in enumerate(protocol["material_ids"]):
            nodes = torch.nonzero(raw_batch.batch == graph, as_tuple=False).flatten()
            identity = torch.arange(nodes.numel(), device=nodes.device)
            all_ot = periodic_assignment(base.frac_coords[nodes], target.frac_coords[nodes])
            type_ot = periodic_assignment(
                base.frac_coords[nodes], target.frac_coords[nodes],
                source_types=raw_batch.atom_types[nodes], target_types=raw_batch.atom_types[nodes],
            )
            velocity = torus_logmap(base.frac_coords[nodes], target.frac_coords[nodes][type_ot])
            quotient_terminal = wrap01(base.frac_coords[nodes] + velocity - velocity.mean(0, keepdim=True))
            rows.append({
                "repeat": repeat,
                "material_id": material_id,
                "identity_periodic_cost": float(periodic_assignment_cost(base.frac_coords[nodes], target.frac_coords[nodes], identity)),
                "all_atom_ot_periodic_cost": float(periodic_assignment_cost(base.frac_coords[nodes], target.frac_coords[nodes], all_ot)),
                "typewise_ot_periodic_cost": float(periodic_assignment_cost(base.frac_coords[nodes], target.frac_coords[nodes], type_ot)),
                "typewise_ot_not_worse_than_identity": bool(periodic_assignment_cost(base.frac_coords[nodes], target.frac_coords[nodes], type_ot) <= periodic_assignment_cost(base.frac_coords[nodes], target.frac_coords[nodes], identity) + 1e-8),
                "no_drift_mean_abs_max": float(velocity.sub(velocity.mean(0, keepdim=True)).mean(0).abs().max()),
                "translation_quotient_terminal_rms": float(translation_aligned_torus_rms(quotient_terminal, target.frac_coords[nodes][type_ot])),
                "simplex_nonnegative": bool((base.type_state[nodes] >= 0).all()),
                "simplex_unit_sum_max_error": float((base.type_state[nodes].sum(-1) - 1).abs().max()),
                "simplex_tangent_sum_max_error": float(matcher._type_velocity(torch.randn_like(base.type_state[nodes])).sum(-1).abs().max()),
            })
    return pd.DataFrame(rows)


def _head_audit(model, matcher: RiemannianCrystalFlowMatcher, batch, seed: int):
    _seed(seed)
    model.zero_grad(set_to_none=True)
    terms = matcher.loss(model, batch)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows = []
    for head in HEADS:
        loss = terms[head]
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        grad_norm = math.sqrt(sum(float(grad.detach().square().sum()) for grad in grads if grad is not None))
        rows.append({
            "head": head,
            "active": head in matcher.active_heads,
            "raw_mse": float(loss.detach()),
            "flow_objective": float(terms[f"{head}_objective"].detach()),
            "gradient_norm": grad_norm,
            "endpoint_type_nll": float(terms["endpoint_type_nll"].detach()),
        })
    return pd.DataFrame(rows)


def _train(batch, variant: dict[str, Any], settings: dict[str, Any], seed: int):
    _seed(seed)
    model = GaugeFlowVectorField(
        hidden_dim=settings["hidden_dim"], layers=settings["layers"], atom_types=119,
        conditioning_mode="endpoint_id",
    ).to(batch.batch.device)
    matcher = RiemannianCrystalFlowMatcher(
        atom_types=119, active_heads=tuple(variant["active_heads"]), type_path=variant["type_path"],
        target_coupling=variant["target_coupling"], coordinate_gauge=variant["coordinate_gauge"],
        loss_normalization=variant["loss_normalization"], endpoint_type_nll_weight=variant["endpoint_type_nll_weight"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    last: dict[str, float] = {}
    for _ in range(settings["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        terms = matcher.loss(model, batch)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {name: float(value.detach()) for name, value in terms.items()}
    model.eval()
    return model, matcher, last, _head_audit(model, matcher, batch, seed + 9000)


def _trajectory(model, matcher, batch, initial, steps: int, variant: str):
    state = _state_copy(initial)
    atoms = int((batch.batch == 0).sum())
    rows = []
    dt = 1.0 / steps
    with torch.no_grad():
        for step in range(steps + 1):
            for name, value, torus in (
                ("type_probability", state.type_state, False),
                ("fractional_coordinate", state.frac_coords, True),
                ("lattice_log", state.lattice_log, False),
            ):
                left, right = value[:atoms], value[atoms:2 * atoms] if value.shape[0] != batch.num_graphs else value[1:2]
                if value.shape[0] == batch.num_graphs:
                    left = value[:1]
                difference = torus_logmap(left, right) if torus else left - right
                rows.append({"variant": variant, "time": step / steps, "quantity": f"state_{name}", "pairwise_rms": _rms(difference)})
            if step == steps:
                break
            time = torch.full((batch.num_graphs,), step / steps, device=batch.batch.device)
            raw = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[:3]
            velocity = (
                matcher._type_velocity(raw[0]) if "type" in matcher.active_heads else torch.zeros_like(raw[0]),
                matcher._coordinate_velocity(raw[1], batch) if "coord" in matcher.active_heads else torch.zeros_like(raw[1]),
                raw[2] if "lattice" in matcher.active_heads else torch.zeros_like(raw[2]),
            )
            for name, value in zip(("type_probability", "fractional_coordinate", "lattice_log"), velocity):
                left, right = (value[:1], value[1:2]) if value.shape[0] == batch.num_graphs else (value[:atoms], value[atoms:2 * atoms])
                rows.append({"variant": variant, "time": step / steps, "quantity": f"velocity_{name}", "pairwise_rms": _rms(left - right)})
            next_type = state.type_state + dt * velocity[0]
            if matcher.type_path == "riemannian_simplex":
                next_type = project_simplex(next_type)
            state = CrystalFlowState(next_type, wrap01(state.frac_coords + dt * velocity[1]), state.lattice_log + dt * velocity[2])
    return pd.DataFrame(rows)


def _write_report(path: Path, protocol: dict[str, Any], audit: pd.DataFrame, results: pd.DataFrame):
    criterion = protocol["endpoint_id_substrate"]["criteria"]
    lines = ["# Gate A5 quotient substrate repair", ""]
    lines += [
        "A5 is a new, fixed-budget endpoint-ID substrate test. It does not alter the negative A4 conclusion or constitute a tensor-conditioned result.",
        "", "## A5.0 path/coupling invariants", "",
        f"All typewise OT costs were no worse than identity: `{bool(audit.typewise_ot_not_worse_than_identity.all())}`.",
        f"Maximum simplex unit-sum error: `{audit.simplex_unit_sum_max_error.max():.3e}`; maximum tangent-sum error: `{audit.simplex_tangent_sum_max_error.max():.3e}`.",
        f"Maximum no-drift graph-mean residual: `{audit.no_drift_mean_abs_max.max():.3e}`.",
        "", "## Fixed endpoint-ID result", "",
        "| Variant | type composition | geometry retrieval | joint retrieval | between/within | failures | qualifies |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        qualified = (
            row.type_decoded_composition_accuracy >= criterion["type_decoded_composition_accuracy_min"]
            and row.geometry_endpoint_retrieval_accuracy >= criterion["geometry_endpoint_retrieval_accuracy_min"]
            and row.joint_endpoint_retrieval_accuracy >= criterion["joint_endpoint_retrieval_accuracy_min"]
            and row.generated_between_within_ratio >= criterion["generated_between_within_ratio_min"]
            and row.sampling_failure_count <= criterion["sampling_failure_count_max"]
        )
        lines.append(f"| {row.variant} | {row.type_decoded_composition_accuracy:.3f} | {row.geometry_endpoint_retrieval_accuracy:.3f} | {row.joint_endpoint_retrieval_accuracy:.3f} | {row.generated_between_within_ratio:.3f} | {int(row.sampling_failure_count)} | {qualified} |")
    lines += ["", "Only an all-criteria pass would permit a separately versioned tensor-conditioned gate. A5 itself never passes Gate A."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a5_quotient_substrate_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a5_quotient_substrate_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_two_endpoint_substrate_repair":
        raise ValueError("A5 requires its versioned pre-registered protocol")
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    raw = _load_panel(protocol, ROOT, device, preprocessed_cache=_resolve(protocol["data"]["preprocessed_cache"]))
    if raw.num_graphs != 2 or int((raw.batch == 0).sum()) != int((raw.batch == 1).sum()):
        raise ValueError("A5 is restricted to the frozen equal-atom-count InN/BN pair")
    audit = _coupling_audit(raw, protocol)
    audit.to_csv(output / "a5_0_path_coupling_audit.csv", index=False)
    settings = protocol["endpoint_id_substrate"]
    endpoint_batch = _localize_types(_set_endpoint_ids(raw), list(range(119)))
    atoms = int((raw.batch == 0).sum())
    rows, decoded_frames, trajectories, head_audits = [], [], [], []
    for index, variant in enumerate(settings["variants"]):
        model, matcher, last, head = _train(endpoint_batch, variant, settings, settings["seed"] + index)
        generated_batch, labels = _generated_batch(raw, settings["sample_replicates"], list(range(119)), device)
        initial = _common_initial(matcher, generated_batch, atoms, settings["sample_replicates"], settings["common_noise_seed"])
        sampled = matcher.sample(model, generated_batch, steps=settings["sampler_steps"], initial_state=initial)
        decoded, summary = _decoded_metrics(sampled, generated_batch, labels, raw, list(range(119)), variant["id"])
        decoded_frames.append(decoded)
        features = _state_features(sampled, generated_batch)
        distribution = _distance_diagnostics(features, labels)
        rows.append({
            "variant": variant["id"], "active_heads": json.dumps(variant["active_heads"]),
            "type_path": variant["type_path"], "target_coupling": variant["target_coupling"],
            "coordinate_gauge": variant["coordinate_gauge"], "loss_normalization": variant["loss_normalization"],
            "endpoint_type_nll_weight": variant["endpoint_type_nll_weight"], **last, **summary,
            "generated_between_within_ratio": distribution["between_within_distance_ratio"],
            "generated_nearest_centroid_accuracy": distribution["leave_one_out_nearest_centroid_accuracy"],
            "sampling_failure_count": _finite_state(sampled),
        })
        head.insert(0, "variant", variant["id"])
        head_audits.append(head)
        trajectory_batch, _ = _generated_batch(raw, 1, list(range(119)), device)
        trajectory_initial = _common_initial(matcher, trajectory_batch, atoms, 1, settings["common_noise_seed"])
        trajectories.append(_trajectory(model, matcher, trajectory_batch, trajectory_initial, settings["sampler_steps"], variant["id"]))
    results = pd.DataFrame(rows)
    results.to_csv(output / "endpoint_id_results.csv", index=False)
    pd.concat(decoded_frames, ignore_index=True).to_csv(output / "decoded_state_audit.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(output / "common_noise_trajectory.csv", index=False)
    pd.concat(head_audits, ignore_index=True).to_csv(output / "head_loss_gradient_audit.csv", index=False)
    (output / "literature_basis.json").write_text(json.dumps(protocol["evidence"]["literature"], indent=2) + "\n", encoding="utf-8")
    _write_report(output / "gate_a5_quotient_substrate_report.md", protocol, audit, results)
    files = [
        "a5_0_path_coupling_audit.csv", "endpoint_id_results.csv", "decoded_state_audit.csv",
        "common_noise_trajectory.csv", "head_loss_gradient_audit.csv", "literature_basis.json",
        "gate_a5_quotient_substrate_report.md",
    ]
    criteria = settings["criteria"]
    passed = []
    for _, row in results.iterrows():
        passed.append(bool(
            row.type_decoded_composition_accuracy >= criteria["type_decoded_composition_accuracy_min"]
            and row.geometry_endpoint_retrieval_accuracy >= criteria["geometry_endpoint_retrieval_accuracy_min"]
            and row.joint_endpoint_retrieval_accuracy >= criteria["joint_endpoint_retrieval_accuracy_min"]
            and row.generated_between_within_ratio >= criteria["generated_between_within_ratio_min"]
            and row.sampling_failure_count <= criteria["sampling_failure_count_max"]
        ))
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "protocol_sha256": _sha256(protocol_path),
        "status": "endpoint_id_substrate_qualified_but_tensor_gate_not_started" if all(passed) else "endpoint_id_substrate_not_qualified",
        "a5_0_passed": bool(audit.typewise_ot_not_worse_than_identity.all() and audit.simplex_unit_sum_max_error.max() < 1e-5 and audit.simplex_tangent_sum_max_error.max() < 1e-5 and audit.no_drift_mean_abs_max.max() < 1e-5),
        "variant_passes": dict(zip(results.variant, passed)),
        "device": str(device),
        "historical_gate_evidence_modified": False,
        "tensor_conditioned_gate_started": False,
        "report_sha256": {name: _sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
