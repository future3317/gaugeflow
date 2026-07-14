"""Evaluate the pre-registered Gate A3 two-target early-branching screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audit_gate_a1 import _common_initial_state, _dummy_batch, _pairwise_rms, _record_pairs, _seed  # noqa: E402
from evaluate_gate_a import _distance_diagnostics, _load_model, _load_panel, _state_features  # noqa: E402
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher  # noqa: E402
from gaugeflow.manifold import log_vector_to_lattice, torus_logmap, wrap01  # noqa: E402
from gaugeflow.tensor import normalize_isotypic  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _per_graph_error(prediction: torch.Tensor, target: torch.Tensor, batch, graphs: int) -> torch.Tensor:
    node = (prediction - target).square().reshape(prediction.shape[0], -1).mean(dim=-1)
    return torch_geometric_scatter_mean(node, batch.batch, graphs)


def torch_geometric_scatter_mean(value: torch.Tensor, index: torch.Tensor, graphs: int) -> torch.Tensor:
    from torch_geometric.utils import scatter

    return scatter(value, index, dim=0, dim_size=graphs, reduce="mean")


def _interpolant(batch, time: torch.Tensor) -> tuple[CrystalFlowState, tuple[torch.Tensor, ...]]:
    matcher = RiemannianCrystalFlowMatcher()
    target = matcher.target_state(batch)
    base = matcher.random_state(batch)
    node_time = time[batch.batch].unsqueeze(-1)
    velocity_type = target.type_state - base.type_state
    velocity_coord = torus_logmap(base.frac_coords, target.frac_coords)
    velocity_lattice = target.lattice_log - base.lattice_log
    state = CrystalFlowState(
        type_state=base.type_state + node_time * velocity_type,
        frac_coords=wrap01(base.frac_coords + node_time * velocity_coord),
        lattice_log=base.lattice_log + time.unsqueeze(-1) * velocity_lattice,
    )
    return state, (velocity_type, velocity_coord, velocity_lattice)


def _all_negative_ranking(model, batch, times: list[float], seed: int, variant: str) -> pd.DataFrame:
    """Evaluate s_ij for every own interpolant i and every condition j."""
    graphs = batch.num_graphs
    if graphs != 2:
        raise ValueError("Gate A3 evaluator is intentionally restricted to two targets")
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, value in enumerate(times):
            _seed(seed + index)
            time = torch.full((graphs,), value, device=batch.batch.device)
            state, velocity = _interpolant(batch, time)
            errors = []
            for condition in range(graphs):
                conditions = batch.piezo_irreps[condition:condition + 1].expand(graphs, -1)
                outputs = model(
                    state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                    conditions, batch.condition_present,
                )[:3]
                errors.append(
                    _per_graph_error(outputs[0], velocity[0], batch, graphs)
                    + _per_graph_error(outputs[1], velocity[1], batch, graphs)
                    + (outputs[2] - velocity[2]).square().mean(dim=-1)
                )
            error = torch.stack(errors, dim=-1)
            score = -error
            chosen = score.argmax(dim=-1)
            for own in range(graphs):
                wrong = 1 - own
                rows.append({
                    "variant": variant,
                    "time": value,
                    "target": own,
                    "own_velocity_error": float(error[own, own]),
                    "all_negative_velocity_error": float(error[own, wrong]),
                    "own_target_margin": float(error[own, wrong] - error[own, own]),
                    "retrieved_condition": int(chosen[own]),
                    "own_target_retrieval": bool(chosen[own] == own),
                    "score_own": float(score[own, own]),
                    "score_negative": float(score[own, wrong]),
                })
    return pd.DataFrame(rows)


def _common_noise_trajectory(model, conditions: torch.Tensor, variant: str, steps: int, atoms: int, seed: int, device: torch.device):
    batch, _ = _dummy_batch(conditions, 1, atoms, device)
    state = _common_initial_state(conditions.shape[0], 1, atoms, device, seed)
    rows: list[dict[str, Any]] = []
    dt = 1.0 / steps
    for name, value, torus in (
        ("state_type_logit", state.type_state, False),
        ("state_fractional_coordinate", state.frac_coords, True),
        ("state_lattice_log", state.lattice_log, False),
    ):
        _record_pairs(rows, method=variant, time=0.0, kind=name, values=_pairwise_rms(
            value, 2, 1, atoms, torus=torus
        ))
    with torch.no_grad():
        for step in range(steps):
            time = torch.full((2,), step / steps, device=device)
            outputs = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present,
            )[:3]
            for name, value, torus in (
                ("velocity_type_logit", outputs[0], False),
                ("velocity_fractional_coordinate", outputs[1], False),
                ("velocity_lattice_log", outputs[2], False),
            ):
                _record_pairs(rows, method=variant, time=float(step / steps), kind=name, values=_pairwise_rms(
                    value, 2, 1, atoms, torus=torus
                ))
            state = CrystalFlowState(
                type_state=state.type_state + dt * outputs[0],
                frac_coords=wrap01(state.frac_coords + dt * outputs[1]),
                lattice_log=state.lattice_log + dt * outputs[2],
            )
            for name, value, torus in (
                ("state_type_logit", state.type_state, False),
                ("state_fractional_coordinate", state.frac_coords, True),
                ("state_lattice_log", state.lattice_log, False),
            ):
                _record_pairs(rows, method=variant, time=float((step + 1) / steps), kind=name, values=_pairwise_rms(
                    value, 2, 1, atoms, torus=torus
                ))
    return pd.DataFrame(rows), state, batch


def _branch_summary(trajectory: pd.DataFrame, variant: str, early_time: float, retention: float) -> tuple[pd.DataFrame, bool]:
    states = trajectory[trajectory.quantity.str.startswith("state_")]
    rows = []
    for quantity, group in states.groupby("quantity", sort=False):
        curve = group.groupby("time", as_index=False).pairwise_rms.mean().sort_values("time")
        early = float(curve.loc[(curve.time - early_time).abs().idxmin(), "pairwise_rms"])
        terminal = float(curve.iloc[-1].pairwise_rms)
        rows.append({
            "variant": variant,
            "state_head": quantity.removeprefix("state_"),
            "early_time": early_time,
            "early_pairwise_rms": early,
            "terminal_pairwise_rms": terminal,
            "terminal_over_early": terminal / max(early, 1e-12),
            "early_positive_finite": bool(math.isfinite(early) and early > 0),
            "terminal_retained": bool(math.isfinite(terminal) and terminal >= retention * early),
        })
    summary = pd.DataFrame(rows)
    passed = bool(summary.early_positive_finite.all() and summary.terminal_retained.all())
    return summary, passed


def _generated_samples(model, conditions: torch.Tensor, evaluation: dict[str, Any], device: torch.device):
    matcher = RiemannianCrystalFlowMatcher()
    batch, labels = _dummy_batch(
        conditions, evaluation["sample_replicates"], evaluation["sample_atoms"], device
    )
    _seed(evaluation["seed"])
    state = matcher.sample(model, batch, steps=evaluation["sample_steps"], guidance_scale=0.0)
    features = _state_features(state, batch)
    atoms = evaluation["sample_atoms"]
    finite = torch.stack((
        torch.isfinite(state.type_state).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
        torch.isfinite(state.frac_coords).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
        torch.isfinite(state.lattice_log).all(dim=1),
    )).all(dim=0)
    return state, batch, labels, {
        **_distance_diagnostics(features, labels),
        "sampling_failure_count": int((~finite).sum()),
    }


def _pair_distance_spectrum(coords: torch.Tensor) -> torch.Tensor:
    delta = torus_logmap(coords.unsqueeze(1), coords.unsqueeze(0))
    distance = torch.linalg.vector_norm(delta, dim=-1)
    upper = distance[torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)]
    return upper.sort().values


def _neighbor_signature(coords: torch.Tensor, lattice: torch.Tensor) -> tuple[torch.Tensor, int]:
    atoms = coords.shape[0]
    if atoms < 2:
        return coords.new_zeros((atoms,)), 0
    left, right = torch.triu_indices(atoms, atoms, offset=1, device=coords.device)
    delta = coords[right] - coords[left]
    axis = torch.arange(-1, 2, device=coords.device, dtype=coords.dtype)
    shifts = torch.cartesian_prod(axis, axis, axis)
    cart = torch.einsum("psj,jk->psk", delta.unsqueeze(1) + shifts.unsqueeze(0), lattice)
    distance = torch.linalg.vector_norm(cart, dim=-1).amin(dim=1)
    threshold = 1.25 * distance.amin().clamp_min(1e-8)
    edges = distance <= threshold
    degree = torch.zeros(atoms, device=coords.device, dtype=coords.dtype)
    degree.index_add_(0, left[edges], torch.ones_like(left[edges], dtype=coords.dtype))
    degree.index_add_(0, right[edges], torch.ones_like(right[edges], dtype=coords.dtype))
    return degree.sort().values, int(edges.sum())


def _decoded_descriptor(type_logits: torch.Tensor, coords: torch.Tensor, lattice_log: torch.Tensor) -> dict[str, Any]:
    types = type_logits.argmax(dim=-1)
    composition = torch.bincount(types, minlength=119).float() / max(types.numel(), 1)
    lattice = log_vector_to_lattice(lattice_log.unsqueeze(0))[0]
    metric = lattice @ lattice.T
    shape = (metric / metric.trace().clamp_min(1e-12)).flatten()
    volume = float(torch.linalg.det(lattice).abs())
    spectrum = _pair_distance_spectrum(coords)
    degree, edges = _neighbor_signature(coords, lattice)
    return {
        "argmax_types": types,
        "composition": composition,
        "lattice_shape": shape,
        "lattice_volume": volume,
        "coordinate_spectrum": spectrum,
        "neighbor_degrees": degree,
        "neighbor_edge_count": edges,
    }


def _descriptor_distance(value: dict[str, Any], reference: dict[str, Any], scale: dict[str, float]) -> float:
    composition = float((value["composition"] - reference["composition"]).abs().sum()) / scale["composition"]
    shape = float(torch.linalg.vector_norm(value["lattice_shape"] - reference["lattice_shape"])) / scale["shape"]
    volume = abs(math.log(max(value["lattice_volume"], 1e-12)) - math.log(max(reference["lattice_volume"], 1e-12))) / scale["volume"]
    coordinates = float(torch.sqrt(((value["coordinate_spectrum"] - reference["coordinate_spectrum"]) ** 2).mean())) / scale["coordinates"]
    topology = (
        float((value["neighbor_degrees"] - reference["neighbor_degrees"]).abs().mean())
        + abs(value["neighbor_edge_count"] - reference["neighbor_edge_count"]) / max(value["argmax_types"].numel(), 1)
    ) / scale["topology"]
    return composition + shape + volume + coordinates + topology


def _decoded_audit(state, generated_batch, labels: torch.Tensor, endpoint_batch, common_terminal, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    atoms = int((generated_batch.batch == 0).sum())
    endpoints = []
    for graph in range(endpoint_batch.num_graphs):
        nodes = torch.nonzero(endpoint_batch.batch == graph, as_tuple=False).flatten()
        target = RiemannianCrystalFlowMatcher().target_state(endpoint_batch)
        endpoints.append(_decoded_descriptor(
            target.type_state[nodes], target.frac_coords[nodes], target.lattice_log[graph]
        ))
    scale = {
        "composition": max(float((endpoints[0]["composition"] - endpoints[1]["composition"]).abs().sum()), 0.05),
        "shape": max(float(torch.linalg.vector_norm(endpoints[0]["lattice_shape"] - endpoints[1]["lattice_shape"])), 0.05),
        "volume": max(abs(math.log(endpoints[0]["lattice_volume"]) - math.log(endpoints[1]["lattice_volume"])), 0.05),
        "coordinates": max(float(torch.sqrt(((endpoints[0]["coordinate_spectrum"] - endpoints[1]["coordinate_spectrum"]) ** 2).mean())), 0.05),
        "topology": max(
            float((endpoints[0]["neighbor_degrees"] - endpoints[1]["neighbor_degrees"]).abs().mean())
            + abs(endpoints[0]["neighbor_edge_count"] - endpoints[1]["neighbor_edge_count"]) / atoms,
            0.05,
        ),
    }
    rows = []
    for graph in range(generated_batch.num_graphs):
        nodes = torch.nonzero(generated_batch.batch == graph, as_tuple=False).flatten()
        descriptor = _decoded_descriptor(state.type_state[nodes], state.frac_coords[nodes], state.lattice_log[graph])
        distance = [_descriptor_distance(descriptor, endpoint, scale) for endpoint in endpoints]
        predicted = int(torch.tensor(distance).argmin())
        rows.append({
            "variant": variant,
            "sample": graph,
            "target": int(labels[graph]),
            "retrieved_endpoint": predicted,
            "endpoint_retrieval_correct": bool(predicted == int(labels[graph])),
            "argmax_atom_types": json.dumps(descriptor["argmax_types"].tolist()),
            "composition": json.dumps(descriptor["composition"].nonzero().flatten().tolist()),
            "composition_fractions": json.dumps(descriptor["composition"][descriptor["composition"] > 0].tolist()),
            "lattice_volume": descriptor["lattice_volume"],
            "lattice_shape": json.dumps(descriptor["lattice_shape"].tolist()),
            "permutation_invariant_coordinate_spectrum": json.dumps(descriptor["coordinate_spectrum"].tolist()),
            "neighbor_degree_sequence": json.dumps(descriptor["neighbor_degrees"].tolist()),
            "neighbor_edge_count": descriptor["neighbor_edge_count"],
            "distance_to_endpoint_0": distance[0],
            "distance_to_endpoint_1": distance[1],
        })
    common_compositions = []
    for graph in range(2):
        nodes = torch.arange(graph * atoms, (graph + 1) * atoms, device=common_terminal.type_state.device)
        common_compositions.append(torch.bincount(common_terminal.type_state[nodes].argmax(dim=-1), minlength=119))
    continuous_difference = float((common_terminal.type_state[:atoms] - common_terminal.type_state[atoms:]).square().mean().sqrt())
    same_discrete_composition = bool(torch.equal(common_compositions[0], common_compositions[1]))
    return pd.DataFrame(rows), {
        "decoded_endpoint_retrieval_accuracy": float(pd.DataFrame(rows).endpoint_retrieval_correct.mean()),
        "common_noise_terminal_type_logit_rms": continuous_difference,
        "common_noise_argmax_composition_equal": same_discrete_composition,
        "continuous_control_without_discrete_branch_change": bool(
            continuous_difference > 1e-6 and same_discrete_composition
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a3_early_branching_v1.json"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/gate_a3_early_branching_v1/two_target"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a3_early_branching_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_two_target_only":
        raise ValueError("A3 evaluation requires the frozen two-target protocol")
    selection_path = _resolve(protocol["selection"]["selection_manifest"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("A3 selection manifest does not match the evaluated protocol")
    device = torch.device(args.device)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.checkpoint_root if args.checkpoint_root.is_absolute() else ROOT / args.checkpoint_root
    raw_batch = _load_panel(
        {"data": protocol["data"], "material_ids": protocol["material_ids"]}, ROOT, device,
        train_csv=_resolve(protocol["data"]["train_csv"]),
        split_manifest=_resolve(protocol["data"]["split_manifest"]),
        target_cache_dir=_resolve(protocol["data"]["target_cache_dir"]),
        preprocessed_cache=_resolve(protocol["data"]["preprocessed_cache"]),
    )
    evaluation = protocol["evaluation"]
    gate = protocol["two_target_gate"]
    metric_rows, ranking_frames, trajectory_frames, decoded_frames, branch_frames = [], [], [], [], []
    checkpoint_hashes: dict[str, str] = {}
    for variant in protocol["variants"]:
        for step in protocol["training"]["checkpoint_steps"]:
            checkpoint = root / variant["id"] / f"step_{step:04d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model, scales, mode = _load_model(checkpoint, device)
            if mode != "direct_irrep":
                raise ValueError(f"{checkpoint} is not a direct-irrep checkpoint")
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            saved = payload["config"]
            for name in ("identification_weight", "identification_temperature", "identification_early_sigma"):
                if saved.get(name) != variant[name]:
                    raise ValueError(f"{checkpoint} has a mismatched {name}")
            batch = raw_batch.clone()
            batch.piezo_irreps = normalize_isotypic(batch.piezo_irreps, scales)
            ranking = _all_negative_ranking(model, batch, evaluation["ranking_times"], evaluation["seed"], variant["id"])
            ranking["checkpoint_step"] = step
            ranking_frames.append(ranking)
            early = ranking[ranking.time <= evaluation["early_time_max"]]
            trajectory, terminal, _ = _common_noise_trajectory(
                model, batch.piezo_irreps, variant["id"], evaluation["trajectory_steps"],
                evaluation["sample_atoms"], evaluation["seed"] + 5000, device,
            )
            trajectory["checkpoint_step"] = step
            trajectory_frames.append(trajectory)
            branch, branch_pass = _branch_summary(
                trajectory, variant["id"], gate["common_noise_early_branch"]["time"],
                gate["common_noise_early_branch"]["terminal_retention_fraction_min"],
            )
            branch["checkpoint_step"] = step
            branch_frames.append(branch)
            state, generated_batch, labels, generated = _generated_samples(model, batch.piezo_irreps, evaluation, device)
            decoded, decoded_summary = _decoded_audit(
                state, generated_batch, labels, batch, terminal, variant["id"]
            )
            decoded["checkpoint_step"] = step
            decoded_frames.append(decoded)
            early_accuracy = float(early.own_target_retrieval.mean())
            all_accuracy = float(ranking.own_target_retrieval.mean())
            mean_margin = float(ranking.own_target_margin.mean())
            own_error = float(ranking.own_velocity_error.mean())
            negative_error = float(ranking.all_negative_velocity_error.mean())
            eligible = variant["id"] == gate["advancement_variant"]
            passed = all((
                eligible,
                early_accuracy >= gate["early_time_own_target_retrieval_accuracy_min"],
                all_accuracy >= gate["all_time_own_target_retrieval_accuracy_min"],
                mean_margin > gate["mean_margin_min_exclusive"],
                branch_pass,
                generated["between_within_distance_ratio"] >= gate["generated_between_within_distance_ratio_min"],
                generated["leave_one_out_nearest_centroid_accuracy"] >= gate["generated_nearest_centroid_accuracy_min"],
                generated["sampling_failure_count"] <= gate["sampling_failure_count_max"],
                own_error <= negative_error,
                decoded_summary["decoded_endpoint_retrieval_accuracy"] >= gate["decoded_endpoint_retrieval_accuracy_min"],
            ))
            metric_rows.append({
                "variant": variant["id"], "checkpoint_step": step,
                "final_training_loss": payload.get("last_terms", {}).get("loss", math.nan),
                "identification_training_loss": payload.get("last_terms", {}).get("identification", 0.0),
                "early_own_target_retrieval_accuracy": early_accuracy,
                "all_time_own_target_retrieval_accuracy": all_accuracy,
                "mean_own_target_margin": mean_margin,
                "mean_own_flow_error": own_error,
                "mean_all_negative_flow_error": negative_error,
                "own_not_worse_than_all_negatives": own_error <= negative_error,
                "common_noise_early_branch_pass": branch_pass,
                "generated_between_within_ratio": generated["between_within_distance_ratio"],
                "generated_nearest_centroid_accuracy": generated["leave_one_out_nearest_centroid_accuracy"],
                "sampling_failure_count": generated["sampling_failure_count"],
                **decoded_summary,
                "eligible_for_expansion": eligible,
                "two_target_pass": passed,
                "checkpoint_sha256": _sha256(checkpoint),
            })
            checkpoint_hashes[f"{variant['id']}/step_{step:04d}"] = _sha256(checkpoint)
    metrics = pd.DataFrame(metric_rows)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    decoded = pd.concat(decoded_frames, ignore_index=True)
    branches = pd.concat(branch_frames, ignore_index=True)
    metrics.to_csv(output_dir / "two_target_metrics.csv", index=False)
    rankings.to_csv(output_dir / "all_negative_tangent_ranking.csv", index=False)
    trajectories.to_csv(output_dir / "common_noise_trajectory.csv", index=False)
    branches.to_csv(output_dir / "early_branch_summary.csv", index=False)
    decoded.to_csv(output_dir / "decoded_state_audit.csv", index=False)
    final_step = max(protocol["training"]["checkpoint_steps"])
    advance = metrics[(metrics.checkpoint_step == final_step) & metrics.two_target_pass]
    status = "two_target_passed_expansion_still_requires_new_versioned_protocol" if not advance.empty else "two_target_not_passed"
    report = f"""# Gate A3 early-branching two-target mechanism screen

## Status

`{status}`. This is a two-target direct-irrep mechanism test only. It does not
change Gate A v1/A1/A2, start S2, launch a 4/8-target extension, or claim Gate
A passage.

## Fixed results

{metrics.round(5).to_markdown(index=False)}

## Decoded-state boundary

`decoded_state_audit.csv` records argmax atom types, composition, lattice
shape/volume, a permutation-invariant fractional pair-distance spectrum,
nearest-neighbor topology, and endpoint retrieval. A true value in
`continuous_control_without_discrete_branch_change` means that a terminal
continuous type-logit difference did not change the matched-noise argmax
composition; it is not evidence of a discrete generative branch.

## Advancement boundary

Only `{gate['advancement_variant']}` can satisfy this protocol's criteria.
Even a passing two-target result requires a new versioned protocol before a
4-target or 8-target screen. A failure requires review of the probability path,
atom-type manifold, decoder, and flow-target definition rather than another
conditional-module search.
"""
    (output_dir / "gate_a3_two_target_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": status,
        "protocol_sha256": _sha256(protocol_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "checkpoint_sha256": checkpoint_hashes,
        "s2_started": False,
        "four_target_started": False,
        "eight_target_started": False,
        "full_benchmark_started": False,
        "historical_gate_evidence_modified": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
